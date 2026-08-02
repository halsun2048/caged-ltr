#!/usr/bin/env python3
"""Build the reproducible R9.0 closeout bundle from already consumed artifacts.

This is a reporting-only script: it never loads labels or evaluates a split.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
DATA = ROOT / "reports" / "data"
TABLES = ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    final = load(EXP / "mind_r8_9_tail_final_once.json")
    student = load(EXP / "mind_r8_9_tail_student.json")
    first = load(ROOT / "runs/mind_r8_9_tail/large_test_first/report.json")
    attr = load(EXP / "mind_r8_10_tail_attribution.json")
    floors = load(EXP / "mind_r8_11_tail_floor.json")
    guard = load(ROOT / "artifacts/mind_r8_0_large_test_guard.json")
    ckpt = ROOT / "artifacts/mind_r8_2_hard_random.pt"

    actual_gate_hash = "56bdab96fb1d43331308860a2de32439d013eb7d72acca598a13b6c3e9cfbcd1"
    checkpoint_hash = sha256(ckpt) if ckpt.exists() else student.get("checkpoint_sha256")
    # Keep the source metrics intact, but replace the compatibility placeholder in
    # the closeout metadata with the hash of the frozen gate artifact.
    policy = dict(final.get("policy", {}))
    policy["gate_model_sha256"] = actual_gate_hash

    rows = []
    for name in ("student", "first", "gate"):
        x = final["overall"][name]
        rows.append({"setting": name, "split": "large_test", "ndcg10": x["ndcg10"],
                     "hit10": x["hit10"], "mrr": x["mrr"],
                     "first_call_rate": x.get("first_call_rate", 0.0),
                     "latency_ms": x.get("latency_ms"),
                     "throughput_qps": 1000.0 / x["latency_ms"] if x.get("latency_ms") else None})

    ablation = []
    current = attr["current_route"]["overall"]
    ablation.append({"policy": "ordinary_gain_gate", "budget": current["first_call_rate"],
                     "tail_floor": 0.0, "ndcg10": current["ndcg10"],
                     "tail_ndcg10": attr["current_route"]["by_bucket"]["tail"]["gate"]["ndcg10"],
                     "latency_ms": current["latency_ms"], "source": "R8.10 fresh-confirm"})
    for c in floors["candidates"]:
        ablation.append({"policy": "tail_floor_gate", "budget": c["budget"],
                         "tail_floor": c["tail_floor"], "ndcg10": c["overall"]["ndcg10"],
                         "tail_ndcg10": c["by_bucket"]["tail"]["gate"]["ndcg10"],
                         "latency_ms": c["overall"]["latency_ms"], "source": "R8.11 fresh-confirm"})

    pareto = [{"policy": "MiniLM", "budget": 0.0, "ndcg10": final["overall"]["student"]["ndcg10"],
               "latency_ms": final["overall"]["student"]["latency_ms"]},
              {"policy": "FIRST", "budget": 1.0, "ndcg10": final["overall"]["first"]["ndcg10"],
               "latency_ms": final["overall"]["first"]["latency_ms"]},
              {"policy": "Tail-floor gate (locked)", "budget": final["overall"]["gate"]["first_call_rate"],
               "ndcg10": final["overall"]["gate"]["ndcg10"],
               "latency_ms": final["overall"]["gate"]["latency_ms"]}]
    pareto += [{"policy": f"tail-floor {r['tail_floor']:.2f}@{r['budget']:.2f}",
                "budget": r["budget"], "ndcg10": r["ndcg10"], "latency_ms": r["latency_ms"]}
               for r in ablation if r["policy"] == "tail_floor_gate"]

    for path, data in [(TABLES / "mind_r9_0_main_results.csv", rows),
                       (TABLES / "mind_r9_0_ablation.csv", ablation),
                       (TABLES / "mind_r9_0_pareto.csv", pareto)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)

    closeout = {
        "schema": "mind_r9_0_closeout_v1",
        "status": "locked",
        "final_split": "large_test",
        "final_metrics": final["overall"],
        "frequency_buckets": final["frequency_buckets"],
        "r9_0_ablation": ablation,
        "pareto": pareto,
        "efficiency": {
            "student": student["efficiency"],
            "first": {"cuda_used": first["cuda_used"], "completed_records": first["completed_records"],
                      "model": first["model"], "protocol_fingerprint": first["protocol_fingerprint"]},
            "gate_throughput_qps": 1000.0 / final["overall"]["gate"]["latency_ms"],
        },
        "hashes": {"student_checkpoint_sha256": checkpoint_hash,
                   "frozen_gate_sha256": actual_gate_hash,
                   "first_protocol_fingerprint": first["protocol_fingerprint"],
                   "large_test_guard": guard["query_id_sha256"]},
        "guard": guard,
        "acceptance": final["acceptance"],
        "limitations": [
            "R9.0 消融复用已消费的 independent fresh-confirm 数据；没有再次访问 large-test。",
            "最终 large-test 只执行一次，guard 已为 consumed_closed/evaluation_count=1。",
            "Torso 分桶的 gate 低于 FIRST，整体和 Tail 仍满足预注册质量差距门槛。",
            "报告未将单次 large-test 结果表述为跨种子统计显著性；置信区间需另建不触碰该 test 的 bootstrap 分析。",
        ],
    }
    (EXP / "mind_r9_0_closeout.json").write_text(json.dumps(closeout, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# R9.0 项目收尾报告\n\n## 锁定主结果（large-test，唯一一次）\n\n| 方法 | NDCG@10 | Hit@10 | MRR | FIRST 调用率 | 延迟 ms/query | 吞吐 q/s |\n|---|---:|---:|---:|---:|---:|---:|\n"""
    for r in rows:
        md += f"| {r['setting']} | {r['ndcg10']:.6f} | {r['hit10']:.6f} | {r['mrr']:.6f} | {r['first_call_rate']:.3f} | {r['latency_ms']:.3f} | {r['throughput_qps']:.2f} |\n"
    md += "\n## Head/Torso/Tail\n\n"
    md += "| 分桶 | MiniLM | FIRST | Tail-floor Gate | Gate 调用率 |\n|---|---:|---:|---:|---:|\n"
    for bucket in ("head", "torso", "tail"):
        b = final["frequency_buckets"][bucket]
        md += f"| {bucket} | {b['student']['ndcg10']:.6f} | {b['first']['ndcg10']:.6f} | {b['gate']['ndcg10']:.6f} | {b['route_rate']:.3f} |\n"
    md += "\n## R9.0 消融与 Pareto\n\n详见 `reports/tables/mind_r9_0_ablation.csv` 和 `reports/tables/mind_r9_0_pareto.csv`。普通 gain gate 在 Tail 上为 %.6f；锁定 Tail-floor gate 将 Tail 提升至 %.6f，同时保持 overall %.6f。\n" % (ablation[0]["tail_ndcg10"], final["frequency_buckets"]["tail"]["gate"]["ndcg10"], final["overall"]["gate"]["ndcg10"])
    md += "\n## 结论、失败结果与限制\n\n- 锁定 gate 的 overall 与 Tail 相对 FIRST 的 NDCG@10 差距均不超过 0.003，FIRST 调用率为 40%，减少约 60%。\n- Tail-floor 是必要消融：普通 gain gate 的 Tail 明显低于 FIRST；提高 Tail 保底后恢复准入。\n- Head 上学生已经强于 FIRST；Torso 仍是主要质量损失来源。\n- large-test guard 已关闭，`evaluation_count=1`；不得重新评估或调参。\n- 大型 checkpoint 仅本地保存，报告记录 SHA-256。\n\n## 复现命令\n\n```bash\npython scripts/analyze_mind_r8_10_tail.py --progress\npython scripts/select_mind_r8_11_tail_floor.py --progress\npython scripts/build_r9_0_closeout.py\n# R8.9 final 命令仅供审计，large-test guard 已 consumed_closed，禁止再次运行\npython scripts/evaluate_mind_r8_9_tail_final_once.py --progress\n```\n\n## 哈希与 guard\n\n- student checkpoint: `{checkpoint_hash}`\n- frozen gate: `{actual_gate_hash}`\n- large-test query-id hash: `{guard['query_id_sha256']}`\n- guard 状态: `{guard['status']}`, evaluation_count=`{guard['evaluation_count']}`\n"""
    md = md.replace("{checkpoint_hash}", checkpoint_hash).replace("{actual_gate_hash}", actual_gate_hash)
    md = md.replace("{guard['query_id_sha256']}", guard["query_id_sha256"])
    md = md.replace("{guard['status']}", guard["status"]).replace("{guard['evaluation_count']}", str(guard["evaluation_count"]))
    (EXP / "mind_r9_0_closeout.md").write_text(md, encoding="utf-8")
    print(json.dumps({"stage": "complete", "report": str(EXP / "mind_r9_0_closeout.json"),
                      "tables": [str(TABLES / "mind_r9_0_main_results.csv"), str(TABLES / "mind_r9_0_ablation.csv"), str(TABLES / "mind_r9_0_pareto.csv")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
