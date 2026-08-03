"""Launch the R16 FastAPI demo with cached or real model adapters."""

import os

from caged_ltr.r16_service import (
    MiniLMBackend,
    PostStudentGateRouter,
    ReplayFirstBackend,
    SearchService,
    create_app,
)


def build_service() -> SearchService:
    mode = os.getenv("R16_BACKEND", "cached")
    route_mode = os.getenv("R16_ROUTE_MODE", "demo_hash")
    router = None
    if route_mode == "post_student_gate":
        router = PostStudentGateRouter(os.environ["R16_GATE_MANIFEST"])
    if mode == "cached":
        return SearchService(
            first_budget=float(os.getenv("R16_FIRST_BUDGET", "0.4")),
            route_mode=route_mode,
            router=router,
        )
    if mode == "replay":
        return SearchService(
            first=ReplayFirstBackend(os.environ["R16_FIRST_RESULTS"]),
            first_budget=float(os.getenv("R16_FIRST_BUDGET", "0.4")),
            route_mode=route_mode,
            router=router,
        )
    if mode not in {"cpu", "real"}:
        raise ValueError("R16_BACKEND must be cached, replay, cpu, or real")
    student = MiniLMBackend(
        os.environ["R16_MODEL_PATH"],
        os.environ["R16_STUDENT_CHECKPOINT"],
        "cpu" if mode == "cpu" else os.getenv("R16_DEVICE", "cuda"),
    )
    first_results = os.getenv("R16_FIRST_RESULTS")
    first = ReplayFirstBackend(first_results) if first_results else None
    return SearchService(
        student=student,
        first=first,
        first_budget=float(os.getenv("R16_FIRST_BUDGET", "0.4")),
        route_mode=route_mode,
        router=router,
    )


app = create_app(build_service())


if app is None:
    raise SystemExit(
        "Install requirements-app.txt first, then run: uvicorn scripts.run_r16_api:app"
    )
