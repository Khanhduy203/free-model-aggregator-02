// The brief's two documents, the free model list, and the catalog.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON documents in it, found
// by their `schema` member; the prose around them does not matter:
//
//   - the free model list (`free-model-aggregator-02/models/v1`): one row a model, keyed by the id
//     the gateway routes (`openrouter/<vendor>/<model>:free`), with its published rating in
//     milli-points, what it can do (tools, vision, JSON mode, context length) and its free flag.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts the client will send,
//     in order, each with its id, its api (openai or anthropic), whether it streams, its weight
//     (1 or 2) and the requirements a model must meet to serve it.
//
// Unlike the sibling -01 contest there is no provider directory to probe: every free model hangs
// off the one provider, `openrouter`, and is dialled on the grader's gateway Chat Completions
// route. The catalog is therefore built straight from the brief — one provider row, the fourteen
// free ids — and the "probe" is a synthetic one, built from the brief's own rows with no HTTP.
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or `manifest`
// (the sibling's names), and one document may carry both; a document is found by the arrays it
// carries, never by a name alone.
//
// Everything here is deterministic: no clock, no randomness, so `node main.ts --list` and the
// register body build the same document, byte for byte, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';
/** The one provider id this contest dials. */
export const OPENROUTER_PROVIDER_ID = 'openrouter';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. This contest has exactly one: openrouter. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as read. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as the brief lists it; null members are what the brief did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. In -02 the
 * probe is synthetic, built from the brief's own model rows — there is no catalog route to ask. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** The one provider this contest dials, as a directory entry. */
export function openRouterProvider(): ProviderDirectoryEntry {
  return {
    id: OPENROUTER_PROVIDER_ID,
    name: 'OpenRouter',
    wire: 'openai',
    auth: 'bearer',
    base: OPENROUTER_PUBLIC_BASE,
    key_env: OPENROUTER_KEY_ENV,
    key: '',
  };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row that
 * gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free unless it
 * says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    model: id,
    provider: named === '' ? OPENROUTER_PROVIDER_ID : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free !== false,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: MODELS_SCHEMA, providers: [openRouterProvider()] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const ratings: RatingsFile = {
    schema: MODELS_SCHEMA,
    ratings: (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
      const r = modelOf(v);
      return r === null ? [] : [r];
    }),
    best: [],
  };
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  const workload: WorkloadFile | null =
    promptsDoc === null
      ? null
      : {
          schema: PROMPTS_SCHEMA,
          manifest: (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
            const w = workloadRequestOf(v);
            return w === null ? [] : [w];
          }),
        };
  if (workload === null) {
    return { directory, ratings, workload: null, error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  return { directory, ratings, workload, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(ratings: RatingsFile | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A model's rating in milli-points, 0 when the quality table has no row for it. */
export function ratingMilliOf(ratings: RatingsFile | null, providerId: string, modelId: string): number {
  return ratingOfModel(ratings, providerId, modelId)?.rating_milli ?? 0;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The synthetic probe for the one provider: every free model the brief lists, with the brief's own
 * capabilities. There is no catalog route to ask in -02 — the brief IS the catalog — so this never
 * touches the network and never fails while the brief parsed.
 */
export function buildProbe(brief: Brief): Probe {
  const provider = brief.directory.providers[0] ?? openRouterProvider();
  if (brief.ratings === null) return { provider, ok: false, status: 0, models: [], error: 'the brief holds no model list' };
  const models: ProbedModel[] = brief.ratings.ratings.map((row) => ({
    id: row.model,
    display_name: row.display_name,
    free: row.free,
    context_length: row.context_length,
    tools: row.tools,
    vision: row.vision,
    json: row.json,
  }));
  return { provider, ok: true, status: 200, models, error: null };
}

/**
 * The catalog the probe and the quality table make: the one provider, every free model it serves.
 *
 * Its base_url is OpenRouter's public endpoint, never the gateway's trial address: the gateway's
 * origin differs between a live pass and its replay, and the catalog is part of the state export a
 * replay must reproduce byte for byte. The scorer compares model-id sets only: the free ids here
 * must be exactly the fourteen dialable ids, once each.
 */
export function buildCatalog(_envUrl: string, probes: readonly Probe[], ratings: RatingsFile | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
