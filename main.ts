// free-model-aggregator-02: one server, two client wires, fourteen live free models.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone. The brief (at $MAKEFUN_PROMPT_FILE)
// carries the free model list (fourteen OpenRouter `:free` models with the house's ratings) and
// the six prompts' manifest; the env (at $MAKEFUN_ENV_URL) drives the six prompts one at a time;
// every completion comes from a real free model dialled through the grader's gateway
// ($OPENAI_BASE_URL) on the trial's key. This server:
//
//   1. reads the brief and builds the one catalog document from its model list (`--list` prints
//      it, the register body carries it, byte for byte) — one provider, `openrouter`, the
//      fourteen dialable ids, no probing: the brief IS the catalog;
//   2. plans the workload: each manifest request gets the best-rated free model that meets its
//      requirements, scarcest-first then heaviest-first (lib/plan.ts);
//   3. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout) and
//      POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   4. serves the client's requests: GET /v1/models (OpenAI shape for Authorization: Bearer,
//      Anthropic shape for x-api-key), POST /v1/chat/completions and POST /v1/messages, each in
//      JSON or server-sent events, answered by one gateway call on the Chat Completions route —
//      always a non-streaming upstream call, with the client-facing stream synthesised from the
//      answer (lib/wire.ts) — failing over to the next-best eligible model when a model refuses,
//      and reporting the refusals it recovered from in the `x-makefun-upstream-failures` header
//      so the failover column can see them;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
//
// The `model` field of every answer names the model really dialled: the house scores the model
// its gateway record shows answered, and a field naming anything else earns the prompt nothing.
//
// Determinism: every choice derives from the brief, the requests and the gateway's answers so
// far. No wall clock, no unseeded randomness, fixed ids and timestamps in every reply, so a
// replay leaves the same state export byte for byte.
//
// Submission note: this file is the entrypoint; lib/providers.ts reads the brief and builds the
// catalog, lib/plan.ts plans the workload and orders failovers, lib/wire.ts translates the two
// client wires to and from Chat Completions. The routing is the -01 v3 design (quality-aware
// shift-cost failover) re-cut for one provider and fourteen live models: the provider-level
// budget and failure-schedule machinery of -01's simulation has no counterpart here, so
// failures are tracked per model, a model that refused is never tried twice for the same
// prompt, and it stays the planned first choice of a later prompt (a refusal may be transient,
// and a retry costs only a gateway call where a needless failover costs quality).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  buildProbe,
  isRecord,
  modelFree,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicModelList,
  anthropicSSE,
  anthropicToOpenAI,
  openAIError,
  openAIJson,
  openAIModelList,
  openAISSE,
  openAIToOpenAI,
  parseOpenAIAnswer,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { candidatesFor, failoverOrder, planAssignments, type Candidate } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one upstream inference call may take, in milliseconds: a real free model answers in
 * seconds, and the env's client waits 30 s per prompt, so 10 s still leaves room for a failover. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most provider calls one client request tries before it gives up. */
const MAX_ATTEMPTS = 4;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The header that reports refusals a served answer recovered from, for the failover column. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let plan = new Map<number, Candidate>();
/** The best eligible rating per manifest request, by index: the scorer's per-request ceiling. */
let bestRatingByIndex: number[] = [];
/** The manifest weight per request, by index. */
let weightByIndex: number[] = [];
/** The weighted quality loss the trial can afford before quality drops below 98: floor(2% of
 * the sum of weight x best rating). For this workload 229 milli-points. */
let qualityBudget = 0;
/** The weighted quality loss already spent this trial: weight x (best - served rating) per
 * served request, weight x best per unserved one. */
let lossSoFar = 0;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference calls made this trial, by model id. */
const callsMade = new Map<string, number>();
/** Failures seen this trial, by model id. A model that refused is never tried twice for the
 * same prompt, and drops to the last tier of the failover order — but it stays the planned
 * first choice of a later prompt, because a refusal may be transient and a retry costs only
 * a gateway call where a needless failover costs quality on every later prompt. */
const failureCounts = new Map<string, number>();

/** One refusal, as the `x-makefun-upstream-failures` header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The failover rating guard for the request now arriving: how many milli-points below the
 * tier's best rating a cheaper failover candidate may sit. The guard spends quality only
 * the trial can afford: slack is the quality budget minus the loss already spent minus the
 * loss the plan still expects from the requests not yet served. Wide (50) while slack is
 * over 100, narrow (25) while it is over 50, closed (0) below that — at guard 0 the
 * failover still prefers the cheapest candidate among the tied-best, which costs no
 * quality at all. A request outside the manifest gets guard 0: no budget is known for it.
 */
function guardFor(index: number): number {
  if (bestRatingByIndex[index] === undefined) return 0;
  const manifest = brief.workload?.manifest ?? [];
  let remainingPlannedLoss = 0;
  for (let j = index; j < manifest.length; j++) {
    const w = weightByIndex[j] ?? 0;
    const b = bestRatingByIndex[j] ?? 0;
    const planned = plan.get(j);
    remainingPlannedLoss += w * (b - (planned?.rating ?? 0));
  }
  const slack = qualityBudget - lossSoFar - remainingPlannedLoss;
  if (slack > 100) return 50;
  if (slack > 50) return 25;
  return 0;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** Build the catalog from the brief and plan the workload. No network: the brief IS the catalog. */
function load(): void {
  if (brief.error !== null) log(brief.error);
  const probe = buildProbe(brief);
  probes = [probe];
  if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = planAssignments(brief, probes);
  const manifest = brief.workload?.manifest ?? [];
  bestRatingByIndex = manifest.map((request) => {
    const cands = candidatesFor(probes, brief.ratings, request.requires, request.api, {});
    return cands.length > 0 ? cands[0]!.rating : 0;
  });
  weightByIndex = manifest.map((request) => request.weight);
  const bound = manifest.reduce((sum, request, i) => sum + request.weight * (bestRatingByIndex[i] ?? 0), 0);
  qualityBudget = Math.floor(0.02 * bound);
  log(`planned ${plan.size} of ${brief.workload?.manifest.length ?? 0} requests over ${probes.filter((p) => p.ok).length} providers`);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(index: number, body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string, headers: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The failures header for a served answer, or no header when nothing refused. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(failures.slice(0, 8)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call's outcome: an answer, or the refusal's status and code for the trail. */
type UpstreamResult = { ok: true; answer: NormalizedAnswer } | { ok: false; status: number; code: string };

/** The error code of a gateway refusal, as a string: the body's code or type, else the HTTP status. */
function errorCode(status: number, body: unknown): string {
  if (isRecord(body)) {
    const error = body.error;
    if (isRecord(error)) {
      if (typeof error.code === 'string' && error.code !== '') return error.code.slice(0, 64);
      if (typeof error.code === 'number') return `http_${error.code}`;
      if (typeof error.type === 'string' && error.type !== '') return error.type.slice(0, 64);
    }
  }
  return `http_${status}`;
}

/**
 * One inference call to one free model, on the gateway's Chat Completions route. The client wire
 * decides the translation; the upstream call is always non-streaming JSON. Returns the normalized
 * answer, or the refusal with its status and code. An answer with no text and no tool calls is a
 * failure too: it cannot carry the prompt's token, so the prompt is lost unless another model
 * answers it.
 */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const url = chatCompletionsUrl();
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (GATEWAY_KEY !== '') headers.authorization = `Bearer ${GATEWAY_KEY}`;
  const upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, candidate.model.id) : anthropicToOpenAI(body, candidate.model.id);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    const timeout = err instanceof Error && err.name === 'TimeoutError';
    log(`upstream ${candidate.model.id} ${timeout ? 'timed out' : 'unreachable'}: ${(err as Error).message}`);
    return { ok: false, status: 502, code: timeout ? 'upstream_timeout' : 'upstream_unreachable' };
  }
  if (!res.ok) {
    let code = `http_${res.status}`;
    try {
      const text = await res.text();
      code = errorCode(res.status, JSON.parse(text));
      log(`upstream ${candidate.model.id} answered HTTP ${res.status}: ${code}`);
    } catch {
      log(`upstream ${candidate.model.id} answered HTTP ${res.status}`);
    }
    return { ok: false, status: res.status, code };
  }
  let answer: unknown;
  try {
    answer = await res.json();
  } catch {
    log(`upstream ${candidate.model.id} answered a body that is not JSON`);
    return { ok: false, status: 502, code: 'upstream_bad_json' };
  }
  if (!isRecord(answer)) {
    log(`upstream ${candidate.model.id} answered a JSON value that is not one object`);
    return { ok: false, status: 502, code: 'upstream_bad_json' };
  }
  const parsed = parseOpenAIAnswer(answer);
  if (parsed.text === '' && parsed.toolCalls.length === 0) {
    log(`upstream ${candidate.model.id} answered with no text and no tool calls`);
    return { ok: false, status: 502, code: 'upstream_empty' };
  }
  return { ok: true, answer: parsed };
}

/**
 * Serve one client request: the planned candidate first (when it is still eligible for the live
 * body) — even when it refused an earlier prompt, because a refusal may be transient and a retry
 * costs only a gateway call where a needless failover costs quality on every later prompt — then
 * the failover order: proven models first, then untouched ones, then models that refused before.
 * A model that refuses during this request is not tried again for it: a real 429 is a rate
 * limit, and an immediate retry usually refuses again where a failover to another model answers
 * at once. Returns the answer, the candidate that served it and the refusals it recovered from,
 * or null when no candidate answered.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<{ answer: NormalizedAnswer; candidate: Candidate; failures: FailureRow[] } | null> {
  const index = manifestIndex;
  manifestIndex += 1;
  const requires = requirementsFor(index, body);
  const weight = weightByIndex[index] ?? 0;
  const best = bestRatingByIndex[index] ?? 0;
  const eligible = candidatesFor(probes, brief.ratings, requires, clientApi, body);
  if (eligible.length === 0) {
    lossSoFar += weight * best;
    return null;
  }

  const tried = new Set<string>();
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const attempts: Candidate[] = [];
  const planned = plan.get(index);
  if (planned !== undefined && eligible.some((c) => keyOf(c) === keyOf(planned))) attempts.push(planned);
  for (const next of failoverOrder(eligible, tried, callsMade, failureCounts, guardFor(index))) {
    if (attempts.some((c) => keyOf(c) === keyOf(next))) continue;
    attempts.push(next);
  }

  const failures: FailureRow[] = [];
  const remaining = attempts.slice(0, MAX_ATTEMPTS);
  let made = 0;
  while (remaining.length > 0 && made < MAX_ATTEMPTS) {
    const candidate = remaining.shift()!;
    tried.add(keyOf(candidate));
    made += 1;
    callsMade.set(candidate.model.id, (callsMade.get(candidate.model.id) ?? 0) + 1);
    const result = await callUpstream(candidate, clientApi, body);
    if (result.ok) {
      lossSoFar += weight * (best - candidate.rating);
      return { answer: result.answer, candidate, failures };
    }
    failureCounts.set(candidate.model.id, (failureCounts.get(candidate.model.id) ?? 0) + 1);
    failures.push({ status: result.status, code: result.code, model: candidate.model.id });
  }
  lossSoFar += weight * best;
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by one free model via the gateway. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model answered this request');
  const model = served.candidate.model.id;
  const headers = failuresHeader(served.failures);
  if (body.stream === true) sendText(res, 200, 'text/event-stream; charset=utf-8', openAISSE(served.answer, model), headers);
  else sendJson(res, 200, openAIJson(served.answer, model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by one free model via the gateway. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model answered this request');
  const model = served.candidate.model.id;
  const headers = failuresHeader(served.failures);
  if (body.stream === true) sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicSSE(served.answer, model), headers);
  else sendJson(res, 200, anthropicJson(served.answer, model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
