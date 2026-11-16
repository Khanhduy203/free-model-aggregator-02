// Reading the brief, and turning it into the catalog the gateway dials.
//
// A trial hands this entry one text file (MAKEFUN_PROMPT_FILE). Buried in
// its prose sit two JSON documents, each carrying a `schema` field:
//   * the model table, schema free-model-aggregator-02/models/v1 —
//     fourteen rows, one per free model, with the id the gateway routes,
//     its rating in milli-points, and what it supports (tools, vision,
//     JSON mode, context length);
//   * the prompt manifest, schema free-model-aggregator-02/prompts/v1 —
//     the six requests the trial will drive, in order, with their weights
//     and the capabilities a model needs to answer them.
// The prose is decoration; only the documents matter. A document is
// recognised by the arrays it carries, so a brief that spells its lists
// under the sibling contest's names (ratings, manifest) still parses, and
// one document may carry both lists.
//
// There is no provider directory to discover here: every free model hangs
// off the single provider openrouter and is dialled on the gateway's Chat
// Completions route. The catalog is therefore the brief itself, reshaped
// — one provider row holding the fourteen ids — and the "probe" is
// synthetic, assembled from the brief's rows without touching the
// network. Nothing here reads a clock or rolls dice, so --list and the
// register body agree byte for byte on a live pass and on its replay.

import * as nodeFs from 'node:fs';

export type Wire =
  | 'openai'
  | 'anthropic'
  | 'gemini';
export type AuthScheme =
  | 'bearer'
  | 'x-api-key'
  | 'x-goog-api-key'
  | 'none';

export const MODELS_SCHEMA =
  'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA =
  'free-model-aggregator-02/prompts/v1';

/** Both brief documents name this prefix in their schema field. */
const SCHEMA_PREFIX =
  'free-model-aggregator-02/';

/** Where the free models live in public: the endpoint the gateway reaches. */
export const OPENROUTER_PUBLIC_BASE =
  'https://openrouter.ai/api/v1';
/** The env var the entrant's OpenRouter key arrives under (an entry secret). */
export const OPENROUTER_KEY_ENV =
  'OPENROUTER_API_KEY';
/** The single provider id this contest dials. */
export const OPENROUTER_PROVIDER_ID =
  'openrouter';

export type Json =
  Record<string, unknown>;
export const isRecord = (
  v: unknown,
): v is Json =>
  v !== null &&
  typeof v === 'object' &&
  Array.isArray(v) === false;

/** One provider row of the brief's directory. This contest has one: openrouter. */
export interface ProviderDirectoryEntry {
  id: string; name: string; wire: Wire; auth: AuthScheme;
  base: string; key_env: string; key: string;
}

export interface ProviderDirectory {
  schema: string; providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string; free: boolean; context_length: number | null;
}

export interface CatalogProvider {
  id: string; base_url: string; key_env: string;
  wire: Wire; models: CatalogModel[];
}

/** What --list prints and /register receives: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the brief's quality table. */
export interface RatingRow {
  model: string; provider: string; display_name: string;
  rating_milli: number; context_length: number | null;
  tools: boolean; vision: boolean; json: boolean; free: boolean;
}

export interface RatingsFile {
  schema: string; ratings: RatingRow[];
  best: Array<{ id: string; rating_milli: number }>;
}

/** What one request demands of a model; a model lacking it cannot serve. */
export interface Requirement {
  tools?: boolean; vision?: boolean; json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string; api: 'openai' | 'anthropic';
  stream: boolean; weight: 1 | 2; requires: Requirement;
}

export interface WorkloadFile {
  schema: string; manifest: WorkloadRequest[];
}

/** The brief, parsed. `error` is set only when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory; error: string | null;
  ratings: RatingsFile | null; workload: WorkloadFile | null;
}

/** One model as the brief lists it; a null member is one the brief left unsaid. */
export interface ProbedModel {
  id: string; display_name: string;
  free: boolean | null; context_length: number | null;
  tools: boolean | null; vision: boolean | null; json: boolean | null;
}

/**
 * One provider's catalog probe: what its route answered, and the models it
 * listed. In -02 the probe is synthetic, built from the brief's own rows —
 * there is no catalog route to ask.
 */
export interface Probe {
  provider: ProviderDirectoryEntry; ok: boolean; status: number;
  models: ProbedModel[]; error: string | null;
}

/** What a model can do, listing first, quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null; tools: boolean | null;
  vision: boolean | null; json: boolean | null;
}

/** The one provider this contest dials, as a directory entry. */
export function openRouterProvider(): ProviderDirectoryEntry {
  const entry: ProviderDirectoryEntry = {
    id: OPENROUTER_PROVIDER_ID, name: 'OpenRouter', wire: 'openai',
    auth: 'bearer', base: OPENROUTER_PUBLIC_BASE,
    key_env: OPENROUTER_KEY_ENV, key: '',
  };
  return entry;
}

/** A base path joined onto the env's URL; an absolute base_url stands as-is. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  const stripTrailing = (s: string): string => s.replace(/\/+$/, '');
  const lower = trimmed.toLowerCase();
  const absolute = lower.startsWith('http://') || lower.startsWith('https://');
  if (absolute) return stripTrailing(trimmed);
  const root = stripTrailing(envUrl);
  if (trimmed.length === 0) return root;
  const sep = trimmed.startsWith('/') ? '' : '/';
  return root + sep + stripTrailing(trimmed);
}

/** The headers a provider wants its key carried in. */
export function authHeaders(
  auth: AuthScheme,
  key: string,
): Record<string, string> {
  if (key.length === 0) return {};
  switch (auth) {
    case 'bearer':
      return { authorization: `Bearer ${key}` };
    case 'x-api-key':
      return { 'x-api-key': key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': key };
    default:
      return {};
  }
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function collectSchemaDocs(text: string): Json[] {
  const found: Json[] = [];
  let cursor = 0;
  for (;;) {
    const open = text.indexOf('{', cursor);
    if (open < 0) break;
    cursor = open + 1;
    const close = matchingClose(text, open);
    if (close < 0) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text.slice(open, close + 1));
    } catch {
      continue;
    }
    if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
  }
  return found;
}

/** Index of the `}` closing the object that opens at `start`, or -1; string bodies are skipped. */
function matchingClose(text: string, start: number): number {
  let depth = 0, quoted = false;
  let i = start;
  while (i < text.length) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') {
        i += 2;
        continue;
      }
      if (ch === '"') quoted = false;
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
    i += 1;
  }
  return -1;
}

function asText(v: unknown): string {
  if (typeof v === 'string') return v;
  return '';
}

function asFiniteNumber(v: unknown): number | null {
  if (typeof v !== 'number' || Number.isFinite(v) === false) return null;
  return v;
}

function asBoolOrNull(v: unknown): boolean | null {
  if (typeof v === 'boolean') return v;
  return null;
}

/**
 * The first document under this contest's prefix carrying one of `keys` as
 * an array: the document and the key it was found under. A brief spelling
 * its lists in the sibling's names still reads.
 */
function findArrayDoc(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (asText(doc.schema).includes(SCHEMA_PREFIX) === false) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the
 * provider naming it: a row giving the bare OpenRouter id
 * (`vendor/model:free`) and naming `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). The brief lists free models,
 * so a row is free unless it says `free: false`; a paid row is never routed.
 */
function parseRatingRow(v: unknown): RatingRow | null {
  if (isRecord(v) === false) return null;
  let raw = asText(v.id);
  if (raw === '') raw = asText(v.model);
  if (raw.length === 0) return null;
  let named = asText(v.provider);
  if (named === '') named = asText(v.wire);
  let id = raw;
  if (named === 'openrouter' && raw.toLowerCase().startsWith('openrouter/') === false) {
    id = `openrouter/${raw}`;
  }
  let display = asText(v.display_name);
  if (display === '') display = raw;
  const row: RatingRow = {
    model: id, provider: named === '' ? OPENROUTER_PROVIDER_ID : named,
    display_name: display, rating_milli: asFiniteNumber(v.rating_milli) ?? 0,
    context_length: asFiniteNumber(v.context_length),
    tools: v.tools === true, vision: v.vision === true,
    json: v.json === true, free: v.free !== false,
  };
  return row;
}

function parseManifestEntry(v: unknown): WorkloadRequest | null {
  if (isRecord(v) === false) return null;
  const id = asText(v.id);
  if (id.length === 0) return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const entry: WorkloadRequest = {
    id, api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true, weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: req.tools === true, vision: req.vision === true,
      json: req.json === true,
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
        ? { min_context: req.min_context }
        : {}),
    },
  };
  return entry;
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(
  path: string,
): Brief {
  const directory: ProviderDirectory = {
    schema: MODELS_SCHEMA,
    providers: [openRouterProvider()],
  };
  const unread = (why: string): Brief => ({
    directory, ratings: null, workload: null, error: why,
  });
  if (path.length === 0) return unread('MAKEFUN_PROMPT_FILE is not set');
  let text: string;
  try {
    text = nodeFs.readFileSync(path, 'utf8');
  } catch (err) {
    return unread(`cannot read ${path}: ${(err as Error).message}`);
  }
  const docs = collectSchemaDocs(text);
  const modelsDoc = findArrayDoc(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return unread(`the brief at ${path} holds no ${MODELS_SCHEMA} document`);
  }
  const rows: RatingRow[] = [];
  for (const item of modelsDoc.doc[modelsDoc.key] as unknown[]) {
    const parsed = parseRatingRow(item);
    if (parsed !== null) rows.push(parsed);
  }
  const ratings: RatingsFile = { schema: MODELS_SCHEMA, ratings: rows, best: [] };
  const promptsDoc = findArrayDoc(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    const missing: Brief = {
      directory, ratings, workload: null,
      error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document`,
    };
    return missing;
  }
  const manifest: WorkloadRequest[] = [];
  for (const item of promptsDoc.doc[promptsDoc.key] as unknown[]) {
    const parsed = parseManifestEntry(item);
    if (parsed !== null) manifest.push(parsed);
  }
  const workload: WorkloadFile = { schema: PROMPTS_SCHEMA, manifest };
  const done: Brief = { directory, ratings, workload, error: null };
  return done;
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(
  ratings: RatingsFile | null,
  providerId: string,
  modelId: string,
): RatingRow | null {
  if (ratings === null) return null;
  for (const row of ratings.ratings) {
    if (row.provider === providerId && row.model === modelId) return row;
  }
  return null;
}

/** A model's rating in milli-points, 0 when the quality table has no row for it. */
export function ratingMilliOf(
  ratings: RatingsFile | null,
  providerId: string,
  modelId: string,
): number {
  const row = ratingOfModel(ratings, providerId, modelId);
  if (row === null) return 0;
  return row.rating_milli;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is
 * the sealed truth), the listing's own word where it does not, and paid
 * when neither says. A paid model is never routed.
 */
export function modelFree(
  model: ProbedModel,
  row: RatingRow | null,
): boolean {
  if (row !== null) return row.free;
  if (model.free === null) return false;
  return model.free;
}

/** What a model can do: the quality table's row first, the listing's fields second. */
export function capabilitiesOf(
  model: ProbedModel,
  row: RatingRow | null,
): Capabilities {
  return {
    context_length: row !== null && row.context_length !== null
      ? row.context_length
      : model.context_length,
    tools: row !== null ? row.tools : model.tools,
    vision: row !== null ? row.vision : model.vision,
    json: row !== null ? row.json : model.json,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(
  requires: Requirement,
  caps: Capabilities,
): boolean {
  const fails = (wanted: boolean | undefined, have: boolean | null): boolean =>
    wanted === true && have !== true;
  if (fails(requires.tools, caps.tools)) return false;
  if (fails(requires.vision, caps.vision)) return false;
  if (fails(requires.json, caps.json)) return false;
  const floor = requires.min_context;
  if (floor !== undefined) {
    if (caps.context_length === null) return false;
    if (caps.context_length < floor) return false;
  }
  return true;
}

/**
 * The synthetic probe for the one provider: every free model the brief
 * lists, with the brief's own capabilities. There is no catalog route to
 * ask in -02 — the brief IS the catalog — so this never touches the
 * network and never fails while the brief parsed.
 */
export function buildProbe(
  brief: Brief,
): Probe {
  const provider = brief.directory.providers[0] ?? openRouterProvider();
  const table = brief.ratings;
  if (table === null) {
    return { provider, ok: false, status: 0, models: [], error: 'the brief holds no model list' };
  }
  const models: ProbedModel[] = [];
  for (const row of table.ratings) {
    models.push({
      id: row.model, display_name: row.display_name, free: row.free,
      context_length: row.context_length, tools: row.tools,
      vision: row.vision, json: row.json,
    });
  }
  const probe: Probe = { provider, ok: true, status: 200, models, error: null };
  return probe;
}

/**
 * The catalog the probe and the quality table make: the one provider, every
 * free model it serves.
 *
 * Its base_url is OpenRouter's public endpoint, never the gateway's trial
 * address: the gateway's origin differs between a live pass and its
 * replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte. The scorer compares model-id sets only: the
 * free ids here must be exactly the fourteen dialable ids, once each.
 */
export function buildCatalog(
  _envUrl: string,
  probes: readonly Probe[],
  ratings: RatingsFile | null,
): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (probe.ok === false) continue;
    const ordered = [...probe.models].sort(compareModelIds);
    const emitted: Set<string> = new Set();
    const models: CatalogModel[] = [];
    for (const model of ordered) {
      if (emitted.has(model.id)) continue;
      emitted.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      const ctx = row?.context_length ?? model.context_length ?? null;
      models.push({ id: model.id, free: modelFree(model, row), context_length: ctx });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

function compareModelIds(a: ProbedModel, b: ProbedModel): number {
  if (a.id === b.id) return 0;
  return a.id < b.id ? -1 : 1;
}
