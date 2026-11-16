#!/usr/bin/env bash
# Draw N random instances with the kit's own generator, run this repo's bot against each via the
# kit's env and the mock gateway, score each with the kit's scorer, and report the distribution.
# Usage: ./run-sim.sh <n> [first-port]
# Env: MOCK_FAIL_FIRST / MOCK_FAIL_ALWAYS / MOCK_DELAY_MS — see run-trial.sh.
set -u
KIT="$HOME/workspace/makefun/kits/free-model-aggregator-02/free-model-aggregator-02-kit"
REPO="$HOME/workspace/makefun/free-model-aggregator-02-repo"
MOCK="$HOME/workspace/makefun/fma-02-test/mock-gateway.mjs"
N="${1:-20}"
PORT0="${2:-19000}"
SIM="/tmp/fma02-sim"
mkdir -p "$SIM"
: > "$SIM/results.tsv"
for i in $(seq 1 "$N"); do
  SEED=$(node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))")
  IN="$SIM/inst-$i/input-seed"
  OUT="$SIM/inst-$i"
  mkdir -p "$IN" "$OUT"
  printf '{"schema":"agent-fun/instance-seed/v1","seed":"%s","purpose":"sim","k":0}' "$SEED" > "$IN/seed.json"
  MAKEFUN_INPUT_DIR="$IN" MAKEFUN_OUTPUT_DIR="$OUT" node "$KIT/program/generator/draw.mjs" > "$SIM/inst-$i/draw.log" 2>&1 || { echo "draw $i failed"; cat "$SIM/inst-$i/draw.log"; continue; }
  STATE="$SIM/state-$i"
  SCORES="$SIM/scores-$i"
  CASSETTE="$SIM/cassette-$i"
  rm -rf "$STATE" "$SCORES" "$CASSETTE"
  mkdir -p "$STATE" "$SCORES" "$CASSETTE"
  PORT=$((PORT0 + i * 2))
  GPORT=$((PORT + 1))
  MAKEFUN_ENV_DATA_DIR="$OUT/input/env" MAKEFUN_STATE_DIR="$STATE" MAKEFUN_ENV_PORT="$PORT" \
    node "$KIT/program/env/server.mjs" > "$STATE/env.log" 2>&1 &
  ENV_PID=$!
  for _ in $(seq 1 50); do [ -f "$STATE/ready" ] && break; sleep 0.1; done
  if [ ! -f "$STATE/ready" ]; then echo "trial $i: env not ready"; kill $ENV_PID 2>/dev/null; continue; fi
  MOCK_PORT="$GPORT" MOCK_CASSETTE_DIR="$CASSETTE" node "$MOCK" > "$STATE/mock.log" 2>&1 &
  MOCK_PID=$!
  sleep 0.3
  MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
    MAKEFUN_PROMPT_FILE="$OUT/input/prompt.txt" \
    MAKEFUN_TRIAL_SEED="sim-$i" \
    OPENAI_BASE_URL="http://127.0.0.1:$GPORT/v1" \
    OPENAI_API_KEY="mock-dummy-key" \
    timeout 170 node "$REPO/main.ts" > "$STATE/bot.log" 2>&1
  BOT_EXIT=$?
  kill -TERM $ENV_PID 2>/dev/null
  wait $ENV_PID 2>/dev/null
  kill $MOCK_PID 2>/dev/null
  if [ ! -f "$STATE/export/state.json" ]; then echo "trial $i: no export (bot exit $BOT_EXIT)"; continue; fi
  MAKEFUN_OUTPUT_DIR="$STATE/export" MAKEFUN_REFERENCE_DIR="$OUT/reference" MAKEFUN_CASSETTE_DIR="$CASSETTE" MAKEFUN_SCORES_DIR="$SCORES" \
    node "$KIT/program/scorer/score.mjs" > "$SCORES/out.log" 2>&1
  SCORE=$(python3 -c "import json;d=json.load(open('$SCORES/scores.json'));print(d['score'],d['quality'],d['served'],d['wire'],d['failover'],d['catalog'],d['upstream_calls'],d['working'])" 2>/dev/null || echo "SCORE-FAIL")
  echo -e "$i\t$SCORE" >> "$SIM/results.tsv"
  echo "trial $i: $SCORE"
done
echo "=== summary (score quality served wire failover catalog calls working) ==="
python3 - "$SIM/results.tsv" <<'EOF'
import sys, statistics
rows=[]
for line in open(sys.argv[1]):
    p=line.split()
    if len(p)!=9: continue
    rows.append([int(x) for x in p[1:]])
if not rows: print("no results"); sys.exit()
names=["score","quality","served","wire","failover","catalog","calls","working"]
for j,n in enumerate(names):
    vals=[r[j] for r in rows]
    print(f"{n:9s} mean={statistics.mean(vals):.2f} min={min(vals)} max={max(vals)}")
print(f"n={len(rows)}  score==100: {sum(1 for r in rows if r[0]==100)}/{len(rows)}")
EOF
