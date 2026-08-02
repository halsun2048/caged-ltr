# ruff: noqa: RUF001, E501
"""R17 portfolio dashboard for the CAGED-LTR serving demo."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import streamlit as st

from caged_ltr.r16_llm_app import as_json, explain_result
from caged_ltr.r16_service import Candidate, SearchService

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_resource
def get_service(budget: float) -> SearchService:
    return SearchService(first_budget=budget)


def candidate_input(raw: str) -> list[Candidate]:
    result = []
    for line in raw.splitlines():
        if line.strip():
            parts = line.split("\t", 1)
            result.append(Candidate(parts[0], parts[1] if len(parts) > 1 else parts[0]))
    return result


def api_search(base_url: str, query: str, candidates: list[Candidate], backend: str) -> dict:
    payload = {
        "query": query,
        "backend": backend,
        "candidates": [item.__dict__ for item in candidates],
    }
    request = Request(
        f"{base_url.rstrip('/')}/search",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def metric_card(report: dict) -> None:
    student, first = report["student"], report["first_recorded_model_inference"]
    gate = report["hard_tail_gate_replay"]
    columns = st.columns(5)
    columns[0].metric("MiniLM P99", f"{student['p99_ms']:.2f} ms")
    columns[1].metric("FIRST P99", f"{first['p99_ms']:.2f} ms")
    columns[2].metric("Gate 平均延迟", f"{gate['mean_ms']:.2f} ms")
    columns[3].metric("FIRST 调用率", f"{gate['first_call_rate']:.0%}")
    columns[4].metric("Student QPS", f"{student['throughput_qps']:.1f}")


st.set_page_config(page_title="CAGED-LTR R17", page_icon="🔎", layout="wide")
st.sidebar.title("CAGED-LTR R17")
st.sidebar.caption("成本感知的大模型搜索重排")
page = st.sidebar.radio("展示页面", ["项目总览", "智能搜索", "A/B 实验", "性能与成本", "部署说明"])
budget = st.sidebar.slider("FIRST 调用预算", 0.0, 1.0, 0.4, 0.05)
api_url = st.sidebar.text_input("FastAPI URL（可选）", os.getenv("R18_API_URL", ""))
service = get_service(budget)
benchmark = load_json(str(ROOT / "reports/experiments/r16_gpu_service_benchmark.json"))
ab_report = load_json(str(ROOT / "reports/experiments/mind_r15_offline_ab.json"))

if page == "项目总览":
    st.title("🔎 CAGED-LTR：成本感知大模型搜索重排")
    st.write(
        "用轻量 Student 处理大多数请求，只把高价值或困难请求路由给 FIRST，展示质量、成本和延迟之间的工程权衡。"
    )
    metric_card(benchmark)
    left, right = st.columns(2)
    with left:
        st.subheader("系统能力")
        st.markdown("""
        - MiniLM CUDA 在线重排
        - Gate 稳定预算路由
        - FIRST 超时、重试、熔断和降级
        - Query 意图识别与结构化解释
        - 在线 A/B 分流与 Prometheus 指标
        """)
    with right:
        st.subheader("研究证据")
        st.metric(
            "A/B Gate NDCG@10",
            f"{ab_report['ab']['randomized_replay']['treatment_gate_ndcg10']:.4f}",
        )
        st.metric(
            "相对全量 FIRST 提升", f"{ab_report['ab']['randomized_replay']['difference']:+.4f}"
        )
        st.metric("FIRST 调用减少", f"{ab_report['ab']['first_call_reduction']:.1%}")
    st.info("当前页面默认使用 deterministic cached backend；连接真实服务时可切换到 GPU API。")

elif page == "智能搜索":
    st.title("智能搜索与路由解释")
    query = st.text_input("搜索需求", "best restaurants near downtown")
    raw = st.text_area(
        "候选文档（每行 item_id<TAB>文本）",
        "A\tBest restaurants near downtown\nB\tA guide to home cooking\nC\tDowntown travel and dining",
    )
    backend = st.selectbox("策略", ["gate", "student", "first"])
    if st.button("开始检索", type="primary"):
        candidates = candidate_input(raw)
        if api_url.strip():
            try:
                result = api_search(api_url, query, candidates, backend)
                st.success("实时 FastAPI 模式")
            except Exception as error:
                st.warning(f"API 不可用，已回退离线模式: {type(error).__name__}")
                result = service.search(query, candidates, backend)
        else:
            result = service.search(query, candidates, backend)
        understanding = as_json(query)
        st.subheader("Query 理解")
        st.json(understanding)
        st.subheader("排序结果")
        for item in result["results"]:
            explanation = explain_result(
                query, item["item_id"], item["text"], item["score"], result["backend"]
            )
            with st.expander(f"#{item['rank']} {item['item_id']} · score={item['score']:.4f}"):
                st.write(item["text"])
                st.json(explanation)
        st.subheader("路由与延迟")
        st.json(
            {
                "backend": result["backend"],
                "route": result["route"],
                "latency_ms": result["latency_ms"],
            }
        )

elif page == "A/B 实验":
    st.title("在线 A/B 实验面板")
    st.subheader("单请求稳定分流演示")
    user_id = st.text_input("匿名用户 ID", "demo-user-001")
    ab_query = st.text_input("A/B Query", "best restaurants downtown")
    if st.button("执行 A/B 请求"):
        ab_candidates = [
            Candidate("A", "best restaurants downtown"),
            Candidate("B", "home cooking guide"),
        ]
        if api_url.strip():
            payload = {
                "query": ab_query,
                "user_id": user_id,
                "backend": "gate",
                "candidates": [item.__dict__ for item in ab_candidates],
            }
            try:
                request = Request(
                    f"{api_url.rstrip('/')}/ab/search",
                    data=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"},
                )
                with urlopen(request, timeout=10) as response:
                    live_ab = json.loads(response.read())
            except Exception as error:
                live_ab = {"error": type(error).__name__}
        else:
            live_ab = service.ab_search(ab_query, ab_candidates, user_id)
        st.json(live_ab)
    ab = ab_report["ab"]
    left, right = st.columns(2)
    with left:
        st.subheader("随机回放结果")
        st.metric("Treatment Gate", f"{ab['randomized_replay']['treatment_gate_ndcg10']:.4f}")
        st.metric("Control FIRST", f"{ab['randomized_replay']['control_first_ndcg10']:.4f}")
        st.metric("差值", f"{ab['randomized_replay']['difference']:+.4f}")
        st.write(f"95% CI: {ab['randomized_replay']['bootstrap_95ci']}")
    with right:
        st.subheader("实验护栏")
        st.json(ab["acceptance"])
        st.metric("FIRST 调用减少", f"{ab['first_call_reduction']:.1%}")
        st.metric("平均延迟降低", f"{ab['latency']['mean_reduction']:.1%}")
    st.caption("这是 dev 上的离线随机回放，不代表真实线上 CTR/CVR；真实线上指标仍需接入广告日志。")

elif page == "性能与成本":
    st.title("质量—成本—延迟面板")
    metric_card(benchmark)
    st.subheader("相对成本模型")
    st.write("以下成本是相对单位：Student=1，FIRST=20；用于展示路由决策，不是云厂商报价。")
    calls = st.slider("模拟请求数", 100, 10000, 1000, 100)
    first_calls = round(calls * budget)
    student_calls = calls - first_calls
    st.bar_chart({"Student": [student_calls], "FIRST": [first_calls]})
    st.write(
        {
            "requests": calls,
            "student_calls": student_calls,
            "first_calls": first_calls,
            "relative_cost": student_calls + 20 * first_calls,
        }
    )
    st.subheader("GPU 基准")
    st.json(
        {
            "device": benchmark["device"],
            "peak_gpu_memory_mib": benchmark["peak_gpu_memory_mib"],
            "student": benchmark["student"],
            "first": benchmark["first_recorded_model_inference"],
        }
    )

else:
    st.title("部署与复现")
    st.code("docker compose -f docker-compose.demo.yml up --build", language="bash")
    st.code("PYTHONPATH=src streamlit run app/streamlit_app.py", language="bash")
    st.markdown("""
    **演示模式**：默认使用本地 deterministic cache，不需要 GPU。

    **真实模式**：设置 `R16_BACKEND=real`、MiniLM checkpoint 和 FIRST replay 路径，启动 FastAPI 后由页面或外部客户端调用。

    **边界**：当前 FIRST 使用冻结 dev replay，不是本地 FIRST 原始权重；在线 CTR/CVR 尚未接入真实业务日志。
    """)
    st.success("R17 页面已准备好用于实习展示。")
