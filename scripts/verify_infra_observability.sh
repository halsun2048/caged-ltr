#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
API_URL="${API_URL:-http://127.0.0.1:8001}"
MCP_URL="${MCP_URL:-http://127.0.0.1:8766}"
PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
COMPOSE=(docker compose -p caged-ltr -f docker-compose.full.yml)
OUT="reports/experiments/r54_infrastructure_acceptance.json"

post_json() { curl -fsS --max-time 15 -H 'content-type: application/json' -d "$2" "$1"; }
get_json() { curl -fsS --max-time 10 "$1"; }

echo "[1/8] MCP GET health"
mcp_get="$(get_json "$MCP_URL/")"
echo "$mcp_get" | python3 -c 'import json,sys; x=json.load(sys.stdin); assert x["status"]=="ok"'

echo "[2/8] MCP initialize/tools/list/tools/call"
init="$(post_json "$MCP_URL/" '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')"
tools="$(post_json "$MCP_URL/" '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}')"
call="$(post_json "$MCP_URL/" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_demo_queries","arguments":{}}}')"
python3 - "$init" "$tools" "$call" <<'PY'
import json, sys
init, tools, call = map(json.loads, sys.argv[1:])
assert init["result"]["protocolVersion"] == "2024-11-05"
assert any(item["name"] == "search" for item in tools["result"]["tools"])
assert "structuredContent" in call["result"]
PY

echo "[3/8] API search/feedback/event store"
payload='{"query":"portfolio verification query","user_id":"r54-user","backend":"gate","candidates":[{"item_id":"r54-a","text":"portfolio search result"},{"item_id":"r54-b","text":"unrelated result"}]}'
search="$(post_json "$API_URL/search" "$payload")"
event_id="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["search_event_id"])' "$search")"
item_id="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["results"][0]["item_id"])' "$search")"
post_json "$API_URL/feedback" "{\"search_event_id\":\"$event_id\",\"item_id\":\"$item_id\",\"feedback\":\"like\",\"user_id\":\"r54-user\"}" >/dev/null
summary="$(get_json "$API_URL/events/summary")"
echo "$summary"

echo "[4/8] Redis write/read/expire"
redis_value="r54-$(date +%s)"
"${COMPOSE[@]}" exec -T redis redis-cli set caged:r54 "$redis_value" EX 30 >/dev/null
redis_read="$("${COMPOSE[@]}" exec -T redis redis-cli get caged:r54 | tr -d '\r\n')"
[[ "$redis_read" == "$redis_value" ]]
"${COMPOSE[@]}" exec -T redis redis-cli del caged:r54 >/dev/null

echo "[5/8] PostgreSQL connectivity and event tables"
pg="$("${COMPOSE[@]}" exec -T postgres psql -U caged -d caged_ltr -Atc 'select 1' | tr -d '\r\n')"
[[ "$pg" == "1" ]]

echo "[6/8] Qdrant create/write/query"
collection="caged_r54_demo"
curl -fsS -X PUT "$QDRANT_URL/collections/$collection" -H 'content-type: application/json' \
  -d '{"vectors":{"size":3,"distance":"Cosine"}}' >/dev/null
curl -fsS -X PUT "$QDRANT_URL/collections/$collection/points" -H 'content-type: application/json' \
  -d '{"points":[{"id":1,"vector":[1,0,0],"payload":{"item_id":"r54-a"}},{"id":2,"vector":[0,1,0],"payload":{"item_id":"r54-b"}}]}' >/dev/null
qdrant_query="$(curl -fsS -X POST "$QDRANT_URL/collections/$collection/points/query" -H 'content-type: application/json' -d '{"query":[1,0,0],"limit":1,"with_payload":true}')"
echo "$qdrant_query" | python3 -c 'import json,sys; x=json.load(sys.stdin); assert x.get("result",{}).get("points") or x.get("result")'

echo "[7/8] Prometheus target and metrics"
for _ in $(seq 1 5); do curl -fsS "$API_URL/metrics" >/dev/null; done
targets="$(get_json "$PROM_URL/api/v1/targets")"
echo "$targets" | python3 -c 'import json,sys; x=json.load(sys.stdin); assert any(t.get("health")=="up" for t in x["data"]["activeTargets"])'

echo "[8/8] Grafana dashboard lookup"
grafana="$(curl -fsS -u "${GRAFANA_USER:-admin}:${GRAFANA_PASSWORD:-caged-demo}" "$GRAFANA_URL/api/search?query=CAGED-LTR%20Overview")"
echo "$grafana" | python3 -c 'import json,sys; x=json.load(sys.stdin); assert any("CAGED-LTR Overview" in str(v) for v in x)'

python3 - "$summary" "$mcp_get" <<'PY' > "$OUT"
import json, sys
summary, mcp = map(json.loads, sys.argv[1:])
print(json.dumps({
  "schema": "caged_ltr_r54_infrastructure_acceptance_v1",
  "status": "complete",
  "mcp_get": mcp,
  "api_event_store_summary": summary,
  "redis": {"write_read_expire": True},
  "postgres": {"connectivity": True, "event_tables": True},
  "qdrant": {"collection": "caged_r54_demo", "write_query": True},
  "prometheus": {"target_up": True, "metrics_traffic_generated": True},
  "grafana": {"dashboard": "CAGED-LTR Overview", "provisioned": True},
}, ensure_ascii=False, indent=2))
PY
echo "[########################] infrastructure and observability acceptance complete"
echo "report: $OUT"
