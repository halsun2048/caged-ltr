"""Launch the R16 FastAPI demo with cached or real model adapters."""

import os

from caged_ltr.r16_service import MiniLMBackend, ReplayFirstBackend, SearchService, create_app


def build_service() -> SearchService:
    mode = os.getenv("R16_BACKEND", "cached")
    if mode != "real":
        return SearchService(first_budget=float(os.getenv("R16_FIRST_BUDGET", "0.4")))
    student = MiniLMBackend(
        os.environ["R16_MODEL_PATH"],
        os.environ["R16_STUDENT_CHECKPOINT"],
        os.getenv("R16_DEVICE", "cuda"),
    )
    first = ReplayFirstBackend(os.environ["R16_FIRST_RESULTS"])
    return SearchService(
        student=student, first=first, first_budget=float(os.getenv("R16_FIRST_BUDGET", "0.4"))
    )


app = create_app(build_service())


if app is None:
    raise SystemExit(
        "Install requirements-app.txt first, then run: uvicorn scripts.run_r16_api:app"
    )
