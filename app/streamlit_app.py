"""Interactive R16 demo UI. Run with: streamlit run app/streamlit_app.py"""

import streamlit as st

from caged_ltr.r16_service import Candidate, SearchService

st.set_page_config(page_title="CAGED-LTR 智能搜索", layout="wide")
st.title("CAGED-LTR: 成本感知大模型搜索重排")
st.caption("MiniLM 轻量排序 + Gate 动态路由 + FIRST 大模型重排")
service = SearchService(first_budget=st.sidebar.slider("FIRST 调用预算", 0.0, 1.0, 0.4, 0.05))
query = st.text_input("输入搜索需求", "best restaurants near downtown")
raw = st.text_area(
    "候选文档 (每行: item_id<TAB>文本)",
    "A\tBest restaurants near downtown\nB\tA guide to home cooking\nC\tDowntown travel and dining",
)
backend = st.sidebar.selectbox("策略", ["gate", "student", "first"])

if st.button("开始检索", type="primary"):
    candidates = [
        Candidate((parts := line.split("\t", 1))[0], parts[1] if len(parts) > 1 else parts[0])
        for line in raw.splitlines()
        if line.strip()
    ]
    result = service.search(query, candidates, backend)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("排序结果")
        for item in result["results"]:
            st.write(
                f"#{item['rank']}  **{item['item_id']}** — {item['text']} "
                f"`score={item['score']:.4f}`"
            )
    with right:
        st.subheader("路由与性能")
        st.json(
            {
                "backend": result["backend"],
                "route": result["route"],
                "latency_ms": result["latency_ms"],
            }
        )
        st.metric("FIRST 调用率", f"{service.metrics()['first_call_rate']:.1%}")

st.info(
    "演示模式默认使用 deterministic cached backend; GPU 服务可通过同一 API 接入真实 MiniLM/FIRST。"
)
