"""Launch the R16 FastAPI demo."""

from caged_ltr.r16_service import app


if app is None:
    raise SystemExit("Install requirements-app.txt first, then run: uvicorn scripts.run_r16_api:app")

