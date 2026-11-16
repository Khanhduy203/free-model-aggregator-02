# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Khanhduy203/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Khanhduy203/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Entry for the [make.fun free-model-aggregator-02 bounty](https://make.fun/bounty/free-model-aggregator-02).

One server, two client wires, fourteen live free models — an entry for the make.fun competition
**free-model-aggregator-02** ("The best live free-model aggregator", opened 2026-11-15T03:30Z).

The server speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
server-sent events) and answers every prompt with a real completion from one of OpenRouter's
fourteen free models, dialled through the grader's gateway on the trial's key. For each request
it picks the best-rated free model that meets the request's requirements, fails over when a
model refuses, and reports the refusals it recovered from so the failover column can see them.

This entry adapts the -01 v3 design (quality-aware shift-cost failover) to -02's live
architecture: one provider instead of ten simulated ones, no catalog probing (the brief IS the
catalog), per-model failure tracking instead of provider budgets, and a 10 s upstream timeout
for real models.

## How it works

- **Brief + catalog.** The server reads the trial brief (`$MAKEFUN_PROMPT_FILE`: the free model
  list with the house's ratings, the six prompts' manifest) and builds one catalog document
  from it: one provider, `openrouter`, the fourteen dialable ids. `node main.ts --list` prints
  that same document; the register body carries it byte for byte.
- **Quality-ranked planning.** The whole workload is planned up front: requests are assigned
  scarcest-first (fewest eligible models) then heaviest-first, each getting its best-rated
  eligible model. With six prompts and fourteen models the plan is a pure quality order.
- **One upstream wire.** Every model is dialled on the gateway's Chat Completions route: an
  Anthropic-shaped client request is translated into a Chat Completions body and the answer
  translated back. Every upstream call is non-streaming JSON; the client-facing stream is
  synthesised from the answer in the exact event order each wire wants. The `model` field of
  every answer names the model really dialled.
- **Failover without same-prompt retries.** A refused call is never retried for the same
  prompt — a real 429 is a rate limit, and an immediate retry usually refuses again — but
  failed over to the next candidate: proven models first, then untouched ones, then models
  that refused before. The planned model stays the first choice of a later prompt, because a
  refusal may be transient and a retry costs only a gateway call where a needless failover
  costs quality on every later prompt. Recovered refusals are reported in the
  `x-makefun-upstream-failures` header, the only way the failover column can see them.
- **Deterministic.** Every choice derives from the brief, the requests and the gateway's
  answers so far: no wall clock, no unseeded randomness, fixed ids and timestamps, so a replay
  leaves the same state export byte for byte.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, `/work` poll, serve-with-failover
- `lib/providers.ts` — the brief's documents, the synthetic single-provider probe, the catalog
- `lib/wire.ts` — the client-wire translations, answer parsing, JSON/SSE response synthesis
- `lib/plan.ts` — the quality-aware assignment planner and the failover order
- `run-trial.sh` — run one practice trial locally, no Docker (env + mock gateway + bot + scorer)
- `run-sim.sh` — draw N random instances with the kit's generator and score them all

## Test results

Local runs against the kit's own env and scorer (no Docker on this machine, so the harness
scripts drive `program/env/server.mjs` and `program/scorer/score.mjs` directly). No OpenRouter
key is available on this machine, so the upstream is a mock gateway
(`~/workspace/makefun/fma-02-test/mock-gateway.mjs`, not part of the entry) that answers like
a compliant free model and records a real cassette for the scorer; everything except live
provider behaviour is exercised end to end.

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| practice/trial-0 | 100 | 100 | 100 | 100 | 100 | 100 | 6 |
| practice/trial-1 | 100 | 100 | 100 | 100 | 100 | 100 | 6 |
| practice/trial-2 | 100 | 100 | 100 | 100 | 100 | 100 | 6 |
| 10 random draws | 100 (10/10) | 100 mean | 100 | 100 | 100 | 100 | 6.0 mean |
| trial-0, best model 429s once | 98 | 96 | 100 | 100 | 100 | 100 | 6 |
| 5 draws, best model 429s once | 98.2 mean | 96.4 mean | 100 | 100 | 100 | 100 | 6.0 mean |
| trial-0, best model 429s always | 94 | 90 | 100 | 100 | 100 | 100 | 6 |
| trial-0, 2 s upstream latency | 100 | 100 | 100 | 100 | 100 | 100 | 6 |

Replay check: two runs of practice/trial-0 leave byte-identical state exports.

Score 100 is the ceiling when the models answer: the plan serves every prompt with its
best-rated eligible model in one call each. The remaining risk is live-provider behaviour the
mock cannot reproduce — real 429s, slow models, and models that do not echo the token.

## Running

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve a trial (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL)
```

Node 22+, standard library only; nothing is installed.
