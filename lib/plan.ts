// Quality-aware routing: which free model serves each request.
//
// Quality is 60% of the score and it is the rating of the model that
// answered each request, so the router wants the best-rated free model
// that meets a request's requirements. Six prompts share fourteen models
// on one provider, so nothing forces spreading: the plan hands every
// request its best eligible model, and the real work is the failover — a
// live free model refuses with a 429 or a 5xx at its own moment, and the
// router must serve the prompt anyway without spending quality the trial
// cannot afford.
//
//   1. per request, the eligible candidates: free models meeting its
//      requirements, best-rated first;
//   2. requests are assigned scarcest-first (fewest eligible candidates),
//      then heaviest-first, so the JSON and tools requests claim their
//      models before the plain requests take the shared best ones, and
//      weight-2 requests outrank weight-1 ones;
//   3. each model is planned up to its cap (below): with six prompts and
//      fourteen models the cap never binds, so every request plans its
//      best candidate and the plan is a pure quality order.
//
// The plan guides, it does not gate: at serve time a planned model that
// refuses is failed over, never retried — a live 429 is a rate limit, and
// an immediate retry usually refuses again where a failover to another
// model answers at once. A model that has refused is tried again only
// when every eligible model has refused. The failover order prefers
// proven models (called before, never failed), then untouched ones, then
// refused ones, and within a tier it spends only quality the trial's
// budget allows (the guard the caller passes). Every call costs one
// gateway call whatever model it names, so there is no per-model cost to
// minimise: within a tier the order is by rating, and the guard is what
// keeps a cheap-looking failover from costing quality the trial cannot
// afford. Everything is deterministic: the same brief gives the same plan.

import {
  capabilitiesOf, isRecord, meetsRequirements, modelFree,
  ratingMilliOf, ratingOfModel,
  type Brief, type Json, type Probe, type ProbedModel,
  type RatingsFile, type Requirement, type WorkloadRequest,
} from './providers.ts';

/** One routable (provider, model) pair, with its rating. */
export interface Candidate {
  probe: Probe; model: ProbedModel; rating: number;
}

/**
 * The per-provider planning cap: how many requests the plan assigns to
 * each provider. This contest has one provider and six prompts, so the
 * cap never binds: every request plans its best candidate. The cap is a
 * planning number only: the router still fails over onto a provider past
 * its cap rather than leave a request unserved.
 */
export const PLANNED_CALL_CAP: Record<string, number> = Object.fromEntries([
  ['openrouter', 7],
]);
export const DEFAULT_CALL_CAP =
  4;

export function callCapOf(providerId: string): number {
  const cap = PLANNED_CALL_CAP[providerId];
  if (cap === undefined) return DEFAULT_CALL_CAP;
  return cap;
}

/** Whether a request body asks for JSON mode on the OpenAI wire (response_format json_object/json_schema). */
export function wantsJsonWire(body: Json): boolean {
  const format = body.response_format;
  if (isRecord(format) === false) return false;
  return format.type === 'json_object' || format.type === 'json_schema';
}

/**
 * The eligible candidates for one request, best-rated first (ties:
 * provider id, then model id, so the order is deterministic). A candidate
 * is a free model on a probed provider that meets the requirements. An
 * OpenAI JSON-mode request additionally excludes anthropic-wire
 * providers: the Anthropic wire cannot carry response_format, so its
 * echo would report json_mode absent.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Candidate[] {
  const jsonWire = clientApi === 'openai' && wantsJsonWire(body);
  const found: Candidate[] = [];
  for (const probe of probes) {
    if (probe.ok === false) continue;
    if (jsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (modelFree(model, row) === false) continue;
      const caps = capabilitiesOf(model, row);
      if (meetsRequirements(requires, caps) === false) continue;
      found.push({
        probe, model,
        rating: ratingMilliOf(ratings, probe.provider.id, model.id),
      });
    }
  }
  found.sort(compareCandidates);
  return found;
}

function compareCandidates(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.probe.provider.id !== b.probe.provider.id) {
    return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  }
  if (a.model.id === b.model.id) return 0;
  return a.model.id < b.model.id ? -1 : 1;
}

/**
 * The planned candidate for each manifest request, by manifest index.
 * Greedy, scarcest-first then heaviest-first, each provider planned up
 * to its cap; a request whose candidates are all planned out still gets
 * its best candidate (over cap) rather than none.
 */
export function planAssignments(
  brief: Brief,
  probes: readonly Probe[],
): Map<number, Candidate> {
  const workload = brief.workload;
  const manifest: WorkloadRequest[] = workload === null ? [] : workload.manifest;
  const plan: Map<number, Candidate> = new Map();
  if (manifest.length < 1) return plan;
  const assigned = new Map<string, number>();
  const usedOf = (providerId: string): number => assigned.get(providerId) ?? 0;

  // The manifest carries no bodies, so the JSON-wire exclusion cannot be
  // read off it: the manifest's own json requirement already restricts
  // JSON requests to json-capable models, and the wire exclusion is
  // applied again at serve time from the live body.
  const scored = manifest.map((request, index) => ({
    index, weight: request.weight,
    candidates: candidatesFor(
      probes, brief.ratings, request.requires, request.api, {},
    ),
  }));
  scored.sort((a, b) => {
    if (a.candidates.length !== b.candidates.length) return a.candidates.length - b.candidates.length;
    if (a.weight !== b.weight) return b.weight - a.weight;
    return a.index - b.index;
  });
  for (const row of scored) {
    if (row.candidates.length < 1) continue;
    let pick: Candidate | undefined;
    for (const c of row.candidates) {
      if (usedOf(c.probe.provider.id) < callCapOf(c.probe.provider.id)) {
        pick = c;
        break;
      }
    }
    if (pick === undefined) pick = row.candidates[0];
    if (pick === undefined) continue;
    assigned.set(pick.probe.provider.id, usedOf(pick.probe.provider.id) + 1);
    plan.set(row.index, pick);
  }
  return plan;
}

/**
 * The marginal failure risk of a model's next call. In -01 this was a
 * per-provider table fitted to the simulation's failure schedule; -02's
 * models are live, and no per-model risk data exists before the trial
 * runs, so every model costs the same and the failover order falls to
 * rating within a tier. The functions stay so the order's shape is the
 * v3 one.
 */
export const DEFAULT_NEXT_CALL_RISK =
  0.3;

/** The expected failures of a model's next call: uniform across models in -02. */
export function nextCallRisk(
  _modelId: string,
  _callsMade: number,
): number {
  const risk = DEFAULT_NEXT_CALL_RISK;
  return risk;
}

/**
 * The shift cost of spending an untouched model's first call on a
 * failover. Uniform in -02 for the same reason: no model carries planned
 * calls that shift.
 */
export const DEFAULT_SHIFT_COST =
  0.4;

export function shiftCostOf(
  _modelId: string,
): number {
  const cost = DEFAULT_SHIFT_COST;
  return cost;
}

/**
 * The failover order for one request: its eligible candidates,
 * already-tried (provider, model) pairs removed, in three tiers — proven
 * models first (called before, never failed: a model that just answered
 * is the likeliest to answer again), then untouched models, then models
 * that have refused this trial (a last resort: a rate limit may have
 * lifted, and a retry beats leaving the prompt unserved). Within a tier,
 * the cheapest candidate goes first — but only among the candidates
 * within `guard` milli-points of the tier's best rating, so a cheap
 * failover never costs more quality than the trial's budget allows.
 * Costs are uniform in -02, so within a tier the order is by rating; the
 * guard is what keeps the order honest. The guard is quality-aware: the
 * caller passes a wide guard (50) while the trial's quality slack is
 * large and narrows it (25, then 0) as the slack runs out, so cheap
 * failovers are bought only with quality the trial can afford.
 */
export function failoverOrder(
  candidates: readonly Candidate[],
  tried: ReadonlySet<string>,
  callsMade: ReadonlyMap<string, number>,
  failureCounts: ReadonlyMap<string, number>,
  guard: number,
): Candidate[] {
  const pairKey = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const tierOf = (c: Candidate): number => {
    const failures = failureCounts.get(c.model.id) ?? 0;
    if (failures >= 1) return 2;
    const made = callsMade.get(c.model.id) ?? 0;
    if (made === 0) return 1;
    return 0;
  };
  const callCost = (c: Candidate): number => {
    const made = callsMade.get(c.model.id) ?? 0;
    if (made === 0) return shiftCostOf(c.model.id);
    return nextCallRisk(c.model.id, made);
  };
  const byCost = (a: Candidate, b: Candidate): number => {
    const diff = callCost(a) - callCost(b);
    if (diff !== 0) return diff;
    return compareByRating(a, b);
  };
  const remaining = candidates.filter((c) => tried.has(pairKey(c)) === false);
  const ordered: Candidate[] = [];
  const tiers = [0, 1, 2];
  for (const tier of tiers) {
    const inTier = remaining.filter((c) => tierOf(c) === tier);
    if (inTier.length < 1) continue;
    let best = inTier[0].rating;
    for (const c of inTier) if (c.rating > best) best = c.rating;
    const near = inTier.filter((c) => c.rating + guard >= best).sort(byCost);
    const far = inTier.filter((c) => c.rating < best - guard).sort(compareByRating);
    ordered.push(...near, ...far);
  }
  return ordered;
}

function compareByRating(a: Candidate, b: Candidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.model.id === b.model.id) return 0;
  return a.model.id < b.model.id ? -1 : 1;
}
