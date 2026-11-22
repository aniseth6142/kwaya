/**
 * OpenRouter agent integration — sends chat messages to DeepSeek V4
 * with score context and tool definitions for synthesis and TTS.
 *
 * Supports:
 * - Standard chat with score context
 * - <synthesize> tags for formant-based audio
 * - <tts> tags for human-like neural TTS
 * - OMR workflow guidance
 */

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
const MODEL = 'deepseek/deepseek-v4-flash';
const SITE_URL = 'https://kwaya.app';
const SITE_NAME = 'Kwaya';

export interface AgentMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface AgentRequest {
  messages: AgentMessage[];
  scoreContext?: string;
}

export interface AgentResponse {
  content: string;
  synthesisRequest?: SynthesisRequest;
  ttsRequest?: TTSRequest;
}

export interface SynthesisRequest {
  action: 'synthesize';
  notes: Array<{
    pitch: string;
    duration: number;
    vowel: string;
    voice: string;
  }>;
}

export interface TTSRequest {
  action: 'tts';
  notes: Array<{
    pitch: string;
    duration: number;
    lyric: string;
    voice: string;
  }>;
}

function getApiKey(): string {
  const key = process.env['OPENROUTER_API_KEY'] || '';
  if (!key || key.startsWith('sk-or-v1-') === false) return '';
  return key;
}

export function hasApiKey(): boolean {
  const key = getApiKey();
  return key.length > 20 && key.startsWith('sk-or-v1-');
}

/**
 * Call OpenRouter with score context injected into system prompt.
 * Parses structured requests from the response.
 */
export async function callAgent(req: AgentRequest): Promise<AgentResponse> {
  const apiKey = getApiKey();
  if (!apiKey) {
    return {
      content: "⚠️ OpenRouter API key not configured. Set `OPENROUTER_API_KEY` in `.env` to enable the AI agent. You can get one at https://openrouter.ai/keys",
    };
  }

  const systemPrompt = buildSystemPrompt(req.scoreContext || '');
  
  const messages: AgentMessage[] = [
    { role: 'system', content: systemPrompt },
    ...req.messages,
  ];

  const body = {
    model: MODEL,
    messages,
    temperature: 0.7,
    max_tokens: 4096,
    stream: false,
  };

  const resp = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': SITE_URL,
      'X-Title': SITE_NAME,
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text();
    const status = resp.status;
    if (status === 401) {
      return { content: '⚠️ Invalid OpenRouter API key. Please check your `.env` file.' };
    }
    if (status === 429) {
      return { content: '⚠️ Rate limited by OpenRouter. Please wait a moment and try again.' };
    }
    return { content: `⚠️ API error (${status}): ${text.substring(0, 300)}` };
  }

  const json = await resp.json() as any;
  const choice = json?.choices?.[0];
  const rawContent = choice?.message?.content || '';
  
  // Check response for structured requests
  return parseAgentResponse(rawContent);
}

function buildSystemPrompt(scoreContext: string): string {
  let prompt = `You are Kwaya, a Swahili choral score assistant. You help users understand and learn from their uploaded choral scores.

Your purpose:
- Explain music theory concepts in simple, clear terms
- Answer questions about the uploaded score (notes, harmony, voice leading, key signatures, time signatures)
- Help users who have NO music background understand how to read sheet music
- When users want to hear how something sounds, offer audio options
- Guide users through the OMR (image-to-score transcription) workflow

CAPABILITIES:
1. **OMR (Optical Music Recognition)**: Users can upload a photo/scan/PDF of sheet music. The system will transcribe it into playable notation automatically. Tell users to upload an image.

2. **Formant Synthesis** (robotic, for precise pitch demonstration):
   Use <synthesize> tags with JSON. Good for showing exact note relationships.
   Format: {"notes": [{"pitch": "C4", "duration": 1.0, "vowel": "a", "voice": "Soprano"}, ...]}

3. **Human Neural TTS** (real human-like voices!):
   Use <tts> tags with JSON for natural singing voices.
   Format: {"notes": [{"pitch": "C4", "duration": 1.0, "lyric": "Ah", "voice": "Soprano"}, ...]}
   Available voices: Soprano (bright female), Alto (warm female), Tenor (warm male), Bass (deep male)
   Use this for realistic choral playback.

TEACHING APPROACH:
- Assume the user may know nothing about music — be patient and encouraging
- NEVER mock or shame
- Use analogies from everyday life
- Reference specific notes and parts from the score
- Always ask if they'd like to hear a passage when relevant

KEY: You can trigger human-like TTS by wrapping JSON in <tts> tags, or formant synthesis by wrapping in <synthesize> tags. Prefer <tts> for human voice requests.`;

  if (scoreContext) {
    prompt += `\n\nHere is the current score:\n\n${scoreContext}\n\nRefer to this when answering questions. Use voice names (Soprano, Alto, Tenor, Bass) and reference specific notes.`;
  } else {
    prompt += `\n\nThe user hasn't uploaded a score yet. Suggest they upload a MusicXML file, or an image/PDF of sheet music (the OMR system will transcribe it automatically).`;
  }

  prompt += `\n\nKeep responses concise but warm. You are a friendly choir tutor, not a textbook.`;

  return prompt;
}

/**
 * Parse the agent's response for structured tags.
 */
function parseAgentResponse(content: string): AgentResponse {
  let cleanContent = content;
  let synthesisRequest: SynthesisRequest | undefined;
  let ttsRequest: TTSRequest | undefined;

  // Extract <synthesize> blocks (formant audio)
  const synthRegex = /<synthesize>\s*([\s\S]*?)\s*<\/synthesize>/g;
  let match;
  while ((match = synthRegex.exec(content)) !== null) {
    try {
      const parsed = JSON.parse(match[1]) as SynthesisRequest;
      if (parsed && parsed.notes && parsed.notes.length > 0) {
        synthesisRequest = parsed;
        cleanContent = cleanContent.replace(match[0], '');
      }
    } catch {
      // Invalid JSON, leave visible
    }
  }

  // Extract <tts> blocks (human neural TTS)
  const ttsRegex = /<tts>\s*([\s\S]*?)\s*<\/tts>/g;
  while ((match = ttsRegex.exec(content)) !== null) {
    try {
      const parsed = JSON.parse(match[1]) as TTSRequest;
      if (parsed && parsed.notes && parsed.notes.length > 0) {
        ttsRequest = parsed;
        cleanContent = cleanContent.replace(match[0], '');
      }
    } catch {
      // Invalid JSON
    }
  }

  return {
    content: cleanContent.trim(),
    synthesisRequest,
    ttsRequest,
  };
}