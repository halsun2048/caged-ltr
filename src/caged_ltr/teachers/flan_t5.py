"""Batched FLAN-T5 backend for the published PRP likelihood prompt."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from caged_ltr.teachers.prp import (
    PRPCandidate,
    PRPQuery,
    TeacherMetadata,
    TeacherResponse,
    prompt_sha256,
)

AUTHOR_PRP_PROMPT = (
    """Given a query "{query}", which of the following two passages is more """
    """relevant to the query?

Passage A: "{first}"
Passage B: "{second}"

Output Passage A or Passage B:"""
)
TARGET_FIRST = "Passage A"
TARGET_SECOND = "Passage B"
DEFAULT_FLAN_T5_XL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
ScoringMode = Literal["likelihood", "generation"]


@dataclass(frozen=True, slots=True)
class OrderedPairRequest:
    """One ordered pair presented as Passage A then Passage B."""

    request_id: str
    query_id: str
    query: str
    first_id: str
    first: str
    second_id: str
    second: str

    def __post_init__(self) -> None:
        required = (
            self.request_id,
            self.query_id,
            self.query,
            self.first_id,
            self.first,
            self.second_id,
            self.second,
        )
        if not all(required):
            raise ValueError("ordered pair fields must not be empty")
        if self.first_id == self.second_id:
            raise ValueError("ordered pair candidate IDs must differ")

    @property
    def key(self) -> str:
        return f"{self.request_id}\0{self.first_id}\0{self.second_id}"

    def prompt(self, template: str = AUTHOR_PRP_PROMPT) -> str:
        return template.format(
            query=self.query,
            first=self.first,
            second=self.second,
        )


def parse_prp_generation(output: str) -> Literal["first", "second", "tie"]:
    """Parse only the two outputs allowed by the author prompt."""
    normalized = " ".join(output.strip().split())
    if normalized == TARGET_FIRST:
        return "first"
    if normalized == TARGET_SECOND:
        return "second"
    return "tie"


def _torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"unsupported inference dtype: {name}") from error


class FlanT5PairwiseTeacher:
    """Score or generate author-format pair preferences in GPU batches."""

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        model_name: str,
        model_revision: str,
        tokenizer_name: str,
        tokenizer_revision: str,
        device: str,
        dtype: str,
        scoring_mode: ScoringMode,
        max_input_tokens: int,
        tie_margin: float = 0.0,
        prompt_template: str = AUTHOR_PRP_PROMPT,
    ) -> None:
        if scoring_mode not in {"likelihood", "generation"}:
            raise ValueError(f"unsupported scoring mode: {scoring_mode}")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if tie_margin < 0:
            raise ValueError("tie_margin must be non-negative")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.scoring_mode = scoring_mode
        self.max_input_tokens = int(max_input_tokens)
        self.tie_margin = float(tie_margin)
        self.prompt_template = prompt_template
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._metadata = TeacherMetadata(
            backend="transformers_flan_t5",
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            quantization=f"{dtype}_unquantized",
            prompt_name="prp_author_pairwise",
            prompt_version="naacl2024_appendix_e1",
            prompt_sha256=prompt_sha256(prompt_template),
            generation_parameters={
                "scoring_mode": scoring_mode,
                "max_input_tokens": self.max_input_tokens,
                "tie_margin": self.tie_margin,
                "do_sample": False,
                "targets": [TARGET_FIRST, TARGET_SECOND],
            },
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str = "google/flan-t5-xl",
        model_revision: str = DEFAULT_FLAN_T5_XL_REVISION,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        scoring_mode: ScoringMode = "likelihood",
        max_input_tokens: int = 512,
        tie_margin: float = 0.0,
        cache_dir: str | None = None,
    ) -> FlanT5PairwiseTeacher:
        """Load an immutable model/tokenizer revision for inference."""
        resolved_tokenizer_name = tokenizer_name or model_name
        resolved_tokenizer_revision = tokenizer_revision or model_revision
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_name,
            revision=resolved_tokenizer_revision,
            cache_dir=cache_dir,
            use_fast=False,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            revision=model_revision,
            cache_dir=cache_dir,
            torch_dtype=_torch_dtype(dtype),
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_name=resolved_tokenizer_name,
            tokenizer_revision=resolved_tokenizer_revision,
            device=device,
            dtype=dtype,
            scoring_mode=scoring_mode,
            max_input_tokens=max_input_tokens,
            tie_margin=tie_margin,
        )

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def compare(
        self,
        query: PRPQuery,
        first: PRPCandidate,
        second: PRPCandidate,
    ) -> TeacherResponse:
        request = OrderedPairRequest(
            request_id=query.query_id,
            query_id=query.query_id,
            query=query.text,
            first_id=first.candidate_id,
            first=first.text,
            second_id=second.candidate_id,
            second=second.text,
        )
        return self.compare_many((request,))[0]

    def compare_many(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> list[TeacherResponse]:
        if not requests:
            return []
        if self.scoring_mode == "likelihood":
            return self._score_likelihood(requests)
        return self._score_generation(requests)

    def runtime_diagnostics(self) -> dict[str, object]:
        """Report the loaded device and peak CUDA allocator footprint."""
        diagnostics: dict[str, object] = {
            "device": str(self.device),
            "dtype": self.dtype,
        }
        if self.device.type == "cuda":
            diagnostics.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(self.device),
                    "cuda_total_memory_bytes": torch.cuda.get_device_properties(
                        self.device
                    ).total_memory,
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                        self.device
                    ),
                }
            )
        return diagnostics

    def _tokenize_prompts(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> tuple[dict[str, torch.Tensor], list[bool]]:
        prompts = [
            request.prompt(self.prompt_template)
            for request in requests
        ]
        untruncated = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=False,
        )
        truncated = [
            len(input_ids) > self.max_input_tokens
            for input_ids in untruncated["input_ids"]
        ]
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        return (
            {
                key: value.to(self.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            },
            truncated,
        )

    def _score_likelihood(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> list[TeacherResponse]:
        encoded, truncated = self._tokenize_prompts(requests)
        targets = self.tokenizer(
            [TARGET_FIRST, TARGET_SECOND],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        target_ids = targets["input_ids"].to(self.device)
        target_mask = targets["attention_mask"].to(self.device).bool()
        labels = target_ids.masked_fill(~target_mask, -100).repeat(len(requests), 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            encoder = self.model.get_encoder()
            encoder_output = encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                return_dict=True,
            )
            expanded_encoder_output = BaseModelOutput(
                last_hidden_state=encoder_output.last_hidden_state.repeat_interleave(
                    2,
                    dim=0,
                )
            )
            expanded_attention_mask = encoded["attention_mask"].repeat_interleave(
                2,
                dim=0,
            )
            output = self.model(
                encoder_outputs=expanded_encoder_output,
                attention_mask=expanded_attention_mask,
                labels=labels,
            )
            log_probabilities = torch.log_softmax(output.logits.float(), dim=-1)
            safe_labels = labels.clamp_min(0)
            token_scores = log_probabilities.gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            sequence_scores = (
                token_scores * labels.ne(-100)
            ).sum(dim=-1).reshape(len(requests), 2)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        target_lengths = target_mask.sum(dim=1).tolist()
        score_values = sequence_scores.detach().cpu().tolist()
        responses: list[TeacherResponse] = []
        for index, (score_first, score_second) in enumerate(score_values):
            margin = float(score_first) - float(score_second)
            if abs(margin) <= self.tie_margin:
                choice: Literal["first", "second", "tie"] = "tie"
                raw_output = "TIE"
                output_tokens = 0
            elif margin > 0:
                choice = "first"
                raw_output = TARGET_FIRST
                output_tokens = int(target_lengths[0])
            else:
                choice = "second"
                raw_output = TARGET_SECOND
                output_tokens = int(target_lengths[1])
            responses.append(
                TeacherResponse(
                    choice=choice,
                    input_tokens=int(input_lengths[index]),
                    output_tokens=output_tokens,
                    latency_ms=latency_ms / len(requests),
                    raw_output=raw_output,
                    score_first=float(score_first),
                    score_second=float(score_second),
                    input_truncated=truncated[index],
                )
            )
        return responses

    def _score_generation(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> list[TeacherResponse]:
        encoded, truncated = self._tokenize_prompts(requests)
        decoder_prefix = self.tokenizer(
            "Passage",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        start_ids = torch.full(
            (1, 1),
            int(self.model.config.decoder_start_token_id),
            dtype=torch.long,
            device=self.device,
        )
        decoder_input_ids = torch.cat((start_ids, decoder_prefix), dim=1).repeat(
            len(requests),
            1,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                decoder_input_ids=decoder_input_ids,
                do_sample=False,
                max_new_tokens=2,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_outputs = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        responses = []
        for index, raw_output in enumerate(raw_outputs):
            responses.append(
                TeacherResponse(
                    choice=parse_prp_generation(raw_output),
                    input_tokens=int(input_lengths[index]),
                    output_tokens=int(output_ids[index].numel()),
                    latency_ms=latency_ms / len(requests),
                    raw_output=raw_output,
                    input_truncated=truncated[index],
                )
            )
        return responses


def likelihood_margin(response: TeacherResponse) -> float | None:
    """Return log P(A)-log P(B) when likelihood scores are available."""
    if response.score_first is None or response.score_second is None:
        return None
    margin = response.score_first - response.score_second
    if not math.isfinite(margin):
        raise ValueError("likelihood margin must be finite")
    return margin
