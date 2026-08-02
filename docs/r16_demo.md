# R16 大模型应用演示

## 本地离线演示

```bash
uv run --frozen python -m pip install -r requirements-app.txt
PYTHONPATH=src uvicorn scripts.run_r16_api:app --host 127.0.0.1 --port 8000
PYTHONPATH=src streamlit run app/streamlit_app.py
```

API 提供 `/health`、`/search`、`/route`、`/ab/search`、`/understand`、`/explain`、`/metrics` 和 `/metrics/prometheus`。FIRST 调用包含超时、一次重试、连续失败熔断和 Student 自动降级；这些事件会进入 metrics。默认 cached backend 保证无 GPU 时仍能演示完整链路；设置 `R16_BACKEND=real` 并提供 `R16_MODEL_PATH`、`R16_STUDENT_CHECKPOINT`、`R16_FIRST_RESULTS` 后可加载真实 MiniLM 与冻结 FIRST replay。

## Docker

```bash
docker compose -f docker-compose.demo.yml up --build
```

API 在 8000 端口，Streamlit 在 8501 端口。Prometheus 抓取配置位于 `monitoring/prometheus.demo.yml`。

## 压力测试

```bash
PYTHONPATH=src python scripts/stress_r16.py --requests 100 --workers 8
```

## GPU 结果

R16 在 RTX 4090 上测得 MiniLM 单请求 P50 4.62ms、P95 4.80ms、P99 4.90ms，吞吐约 216.6 QPS；矩阵结果见 `reports/experiments/r16_gpu_matrix.json`。冷启动约 540ms。模型和 tokenizer 位于本地 `artifacts/r16_runtime/`，该目录被 Git 忽略，仅保留本地并由资产清单校验。

FIRST 的原始权重不在当前服务器资产中，因此演示使用已有 dev FIRST logits/ranking replay；这不是新的 FIRST 训练或 untouched-test 评估。
