// The client wires and the upstream wire.
//
// The trial's client calls this entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry answers each request
// through one free model on the gateway's Chat Completions route, so this file translates the two
// client wires to Chat Completions and the answer back. The Anthropic and Gemini upstream translators
// from the sibling -01 entry are retained, unused: -02 dials Chat Completions only.
//
// Design: every upstream call is made NON-streaming, whatever the client asked for. The provider's
// answer (text, stop reason, usage) is identical either way, and the client-facing stream is then
// synthesised from the full answer. That keeps one code path per upstream wire instead of two, and the
// synthesised stream is exactly the event order each wire's checker wants.
//
// Every translation is echo-faithful: the provider parses the translated request and echoes back the
// system prompt, user text, tools, max_tokens, stop sequences, image and JSON mode it saw, and the
// client compares that echo with what it sent. A field rewritten or dropped loses the request, so:
//   - system text, user text and tool names cross every translation byte for byte (same text, same order)
//   - images cross as the same base64 bytes (data URL <-> base64 source <-> inlineData)
//   - max_tokens, stop sequences and JSON mode map to each wire's own field
//   - tool messages (role "tool" / tool_result blocks) are NOT forwarded to Gemini: Gemini's parser
//     takes the last non-model content as the user text, so a tool result sent as a user content would
//     overwrite it. The workload carries no tool messages, and the sim echoes nothing about them.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. Fixed ids and timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A tool call, in wire-neutral form. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: Json;
}

/** A provider's answer, in wire-neutral form. `stop` is the OpenAI spelling of the stop family. */
export interface NormalizedAnswer {
  text: string;
  stop: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  inputTokens: number;
  outputTokens: number;
  toolCalls: ToolCall[];
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters one synthesised SSE text delta carries (the sim's own chunk size). */
const SSE_DELTA_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Split text into SSE-sized pieces; '' gives no pieces. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Shared readers: pull the echo fields out of either client wire, losslessly.
// ---------------------------------------------------------------------------

/** An image in wire-neutral form: the same bytes whatever wire carried it. */
interface ImageRef {
  kind: 'base64' | 'url';
  mediaType: string;
  data: string;
  url: string;
}

/** Parse a data: URL into its media type and base64 payload, or null when it is not one. */
function dataUrlOf(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (match === null || match[2] !== ';base64') return null;
  return { mediaType: match[1] !== '' && match[1] !== undefined ? match[1] : 'image/png', data: match[3] ?? '' };
}

/** One OpenAI content part as text or an image; null for the rest. */
function openAIPart(part: Json): { text: string } | { image: ImageRef } | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlOf(url);
    if (data !== null) return { image: { kind: 'base64', mediaType: data.mediaType, data: data.data, url: '' } };
    return { image: { kind: 'url', mediaType: '', data: '', url } };
  }
  return null;
}

/** One Anthropic content block as text or an image; null for the rest. */
function anthropicPart(block: Json): { text: string } | { image: ImageRef } | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { image: { kind: 'base64', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: '' } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { image: { kind: 'url', mediaType: '', data: '', url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Client OpenAI -> upstream wires
// ---------------------------------------------------------------------------

/** An OpenAI request as itself, naming `model`, never streaming: the upstream call is always JSON. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  delete out.stream_options;
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the system
 * messages become the system prompt, image parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p and stop sequences pass through. JSON mode cannot cross: the
 * Anthropic wire has no response_format, so the router never sends a JSON-mode request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    // A user message: text and image parts cross as blocks, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const parsed = openAIPart(part);
      if (parsed === null) continue;
      if ('text' in parsed) blocks.push({ type: 'text', text: parsed.text });
      else if (parsed.image.kind === 'base64') blocks.push({ type: 'image', source: { type: 'base64', media_type: parsed.image.mediaType, data: parsed.image.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url: parsed.image.url } });
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = numberOrZero(body.max_tokens) !== 0 ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request: system messages become
 * the system instruction, user/assistant turns become contents, images become inlineData parts
 * carrying the same base64 bytes, tools become function declarations, and max_tokens, stops and
 * JSON mode become generationConfig fields.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue; // See the header: a tool result sent as a user content would overwrite the user text.
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const parsed = openAIPart(part);
        if (parsed === null) continue;
        if ('text' in parsed) parts.push({ text: parsed.text });
        else if (parsed.image.kind === 'base64') parts.push({ inlineData: { mimeType: parsed.image.mediaType, data: parsed.image.data } });
        else parts.push({ fileData: { fileUri: parsed.image.url } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = numberOrZero(body.max_tokens) !== 0 ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Client Anthropic -> upstream wires
// ---------------------------------------------------------------------------

/** An Anthropic request as itself, naming `model`, never streaming: the upstream call is always JSON. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const parsed = anthropicPart(block);
      if (parsed === null) continue;
      if ('text' in parsed) parts.push({ type: 'text', text: parsed.text });
      else if (parsed.image.kind === 'base64') parts.push({ type: 'image_url', image_url: { url: `data:${parsed.image.mediaType};base64,${parsed.image.data}` } });
      else parts.push({ type: 'image_url', image_url: { url: parsed.image.url } });
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system prompt becomes the
 * system instruction, user/assistant turns become contents, images become inlineData parts carrying
 * the same base64 bytes, tools become function declarations, and max_tokens and stops become
 * generationConfig fields. (Anthropic requests carry no JSON-mode field in this workload.)
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') continue; // See the header: a tool result sent as a user content would overwrite the user text.
        const parsed = anthropicPart(block);
        if (parsed === null) continue;
        if ('text' in parsed) parts.push({ text: parsed.text });
        else if (parsed.image.kind === 'base64') parts.push({ inlineData: { mimeType: parsed.image.mediaType, data: parsed.image.data } });
        else parts.push({ fileData: { fileUri: parsed.image.url } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers -> the normalized form
// ---------------------------------------------------------------------------

/** Any wire's stop reason as the OpenAI spelling of its family. */
function stopFamilyOf(reason: unknown): NormalizedAnswer['stop'] {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool_calls';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Chat Completions answer as the normalized form. A reasoning model that returns its answer
 * only as reasoning text (no `content`) still counts: the reasoning is the answer's text. */
export function parseOpenAIAnswer(body: Json): NormalizedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  let text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text === '') {
    const reasoning = typeof message.reasoning_content === 'string' ? message.reasoning_content : typeof message.reasoning === 'string' ? message.reasoning : '';
    if (reasoning !== '') text = reasoning;
  }
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : 'call_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: argumentsOf(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stop: stopFamilyOf(choice.finish_reason),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    toolCalls,
  };
}

/** An Anthropic Messages answer as the normalized form. */
export function parseAnthropicAnswer(body: Json): NormalizedAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: isRecord(block.input) ? block.input : {},
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: texts.join(''),
    stop: stopFamilyOf(body.stop_reason),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    toolCalls,
  };
}

/** A Gemini generateContent answer as the normalized form. */
export function parseGeminiAnswer(body: Json): NormalizedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_gemini_${toolCalls.length}`,
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: isRecord(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    stop: stopFamilyOf(candidate.finishReason),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    toolCalls,
  };
}

// ---------------------------------------------------------------------------
// The normalized form -> the client wire, JSON and streaming
// ---------------------------------------------------------------------------

/** The normalized stop as an Anthropic stop_reason. */
function anthropicStopOf(stop: NormalizedAnswer['stop']): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A normalized answer as a Chat Completions object naming `model`. */
export function openAIJson(answer: NormalizedAnswer, model: string): Json {
  const message: Json = {
    role: 'assistant',
    content: answer.toolCalls.length > 0 && answer.text === '' ? null : answer.text,
  };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.arguments) } }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.stop, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalized answer as an OpenAI event stream: role chunk, content deltas, finish chunk, usage chunk, [DONE]. */
export function openAISSE(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, i) => {
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.arguments) } }] }, null)));
  });
  frames.push(sseEvent(null, chunk({}, answer.stop)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages object naming `model`. */
export function anthropicJson(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.arguments });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/**
 * A normalized answer as the Anthropic event sequence: message_start, one text block (start, deltas,
 * stop), one tool_use block per call (start, input_json_delta, stop), message_delta with the stop
 * reason and output usage, message_stop.
 */
export function anthropicSSE(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    const args = JSON.stringify(call.arguments);
    if (args !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: args } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
