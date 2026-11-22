/**
 * OMR — Optical Music Recognition for Kwaya.
 * Reads PDF/PNG/JPG sheet music images and produces structured score data
 * using AI vision transcription via OpenRouter.
 *
 * Flow: image → sharp preprocessing → AI vision → MusicXML → parsed score
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import sharp from 'sharp';
import { parseMusicXML, ParsedScore } from './music-parser';

// ── AI Vision Model ────────────────────────────────────────────────
const VISION_MODEL = 'google/gemini-2.0-flash-001';

// ── Types ──────────────────────────────────────────────────────────

export interface OMRResult {
  ok: boolean;
  score?: ParsedScore;
  rawMusicXML?: string;
  confidence?: number;
  error?: string;
  verification?: VerificationResult;
}

export interface VerificationResult {
  passed: boolean;
  issues: string[];
  summary: string;
}

const SUPPORTED_MIME: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.tiff': 'image/tiff',
  '.tif': 'image/tiff',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
  '.pdf': 'application/pdf',
};

// ── Find ffmpeg (Windows-safe) ─────────────────────────────────────

let _ffmpegPath: string | null = null;

function resolveFfmpeg(): string {
  if (_ffmpegPath) return _ffmpegPath;

  // Try where command (Windows)
  try {
    const out = execSync('where ffmpeg 2>NUL', { stdio: 'pipe', timeout: 5000, encoding: 'utf-8' });
    const line = out.split('\n')[0].trim();
    if (line) {
      _ffmpegPath = line;
      return _ffmpegPath;
    }
  } catch { /* not on PATH */ }

  // Known install locations
  const candidates = [
    'C:\\Users\\HomePC\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1.2-full_build\\bin\\ffmpeg.exe',
    'ffmpeg',
  ];

  for (const c of candidates) {
    if (c === 'ffmpeg') { _ffmpegPath = 'ffmpeg'; return _ffmpegPath; }
    if (fs.existsSync(c)) { _ffmpegPath = c; return _ffmpegPath; }
  }

  _ffmpegPath = 'ffmpeg';
  return _ffmpegPath;
}

// ── Main OMR entry point ───────────────────────────────────────────

export async function transcribeImage(filePath: string): Promise<OMRResult> {
  const ext = path.extname(filePath).toLowerCase();
  const mime = SUPPORTED_MIME[ext];

  if (!mime) {
    return { ok: false, error: `Unsupported format: ${ext}. Use PNG, JPG, TIFF, BMP, WebP, or PDF.` };
  }

  try {
    let processedPath: string;
    let imageBuffer: Buffer;

    if (ext === '.pdf') {
      processedPath = await pdfToPng(filePath);
      imageBuffer = fs.readFileSync(processedPath);
      if (processedPath !== filePath) fs.unlinkSync(processedPath);
    } else {
      processedPath = await preprocessImage(filePath);
      imageBuffer = fs.readFileSync(processedPath);
      if (processedPath !== filePath) fs.unlinkSync(processedPath);
    }

    const base64 = imageBuffer.toString('base64');

    // Call AI vision for transcription
    const transcription = await callVisionModel(base64, SUPPORTED_MIME['.png']);

    if (!transcription.ok) {
      return transcription;
    }

    const musicXML = transcription.rawMusicXML;
    if (!musicXML) {
      return { ok: false, error: 'AI returned no MusicXML content' };
    }

    let parsed: ParsedScore;
    try {
      parsed = await parseMusicXML(musicXML);
    } catch (parseErr: any) {
      return {
        ok: false,
        error: `Failed to parse AI-generated MusicXML: ${parseErr.message}`,
        rawMusicXML: musicXML,
      };
    }

    return {
      ok: true,
      score: parsed,
      rawMusicXML: musicXML,
      confidence: transcription.confidence || 0.7,
    };
  } catch (err: any) {
    return { ok: false, error: `OMR failed: ${err.message}` };
  }
}

// ── Image preprocessing ────────────────────────────────────────────

async function preprocessImage(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();
  const img = sharp(filePath);
  const metadata = await img.metadata();

  if (!metadata) {
    throw new Error('Could not read image metadata');
  }

  let pipeline = img.png();

  const maxDim = 2048;
  const w = metadata.width || maxDim;
  const h = metadata.height || maxDim;
  if (w > maxDim || h > maxDim) {
    if (w >= h) {
      pipeline = pipeline.resize(maxDim, undefined, { fit: 'inside', withoutEnlargement: true });
    } else {
      pipeline = pipeline.resize(undefined, maxDim, { fit: 'inside', withoutEnlargement: true });
    }
  }

  pipeline = pipeline.normalise().sharpen();

  const tempPath = filePath.replace(ext, '_processed.png');
  await pipeline.toFile(tempPath);
  return tempPath;
}

// ── PDF → PNG (Windows ffmpeg) ────────────────────────────────────

async function pdfToPng(pdfPath: string): Promise<string> {
  const ffmpeg = resolveFfmpeg();
  const outPath = pdfPath.replace('.pdf', '_page1.png');
  const shellQuote = (s: string) => `"${s}"`;

  // Attempt 1: with density flag (better for vector PDFs)
  try {
    execSync(
      `${shellQuote(ffmpeg)} -y -density 300 -i ${shellQuote(pdfPath)} -vf "scale=2048:-1" -frames:v 1 ${shellQuote(outPath)}`,
      { stdio: 'pipe', timeout: 30000 }
    );
    if (fs.existsSync(outPath) && fs.statSync(outPath).size > 100) return outPath;
  } catch { /* fall through */ }

  // Attempt 2: without density
  execSync(
    `${shellQuote(ffmpeg)} -y -i ${shellQuote(pdfPath)} -vf "scale=2048:-1" -frames:v 1 ${shellQuote(outPath)}`,
    { stdio: 'pipe', timeout: 30000 }
  );

  if (!fs.existsSync(outPath) || fs.statSync(outPath).size < 100) {
    throw new Error(
      'Failed to convert PDF to image. Ensure ffmpeg is installed.\n' +
      'Run: winget install Gyan.FFmpeg'
    );
  }

  return outPath;
}

// ── AI Vision call ─────────────────────────────────────────────────

async function callVisionModel(
  imageBase64: string,
  mimeType: string
): Promise<{ ok: boolean; rawMusicXML?: string; confidence?: number; error?: string }> {
  const apiKey = process.env['OPENROUTER_API_KEY'] || '';
  if (!apiKey || !apiKey.startsWith('sk-or-v1-')) {
    return { ok: false, error: 'OpenRouter API key not configured in .env' };
  }

  const prompt = buildVisionPrompt();

  const body = {
    model: VISION_MODEL,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          {
            type: 'image_url',
            image_url: {
              url: `data:${mimeType};base64,${imageBase64}`,
            },
          },
        ],
      },
    ],
    temperature: 0.1,
    max_tokens: 8192,
  };

  try {
    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://kwaya.app',
        'X-Title': 'Kwaya OMR',
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text();
      if (resp.status === 402) {
        return { ok: false, error: 'OpenRouter credit required. Add credits at https://openrouter.ai/keys' };
      }
      return { ok: false, error: `Vision API error (${resp.status}): ${text.substring(0, 200)}` };
    }

    const json = await resp.json() as any;
    const content = json?.choices?.[0]?.message?.content || '';
    return extractMusicXML(content);
  } catch (err: any) {
    return { ok: false, error: `Vision API call failed: ${err.message}` };
  }
}

// ── Prompt & Response parsing ──────────────────────────────────────

function buildVisionPrompt(): string {
  return `You are a music transcription AI. Transcribe this sheet music image into **valid MusicXML** format (score-partwise).

RULES:
- Output ONLY valid MusicXML markup — no explanations, no markdown
- Use <score-partwise version="4.0"> as the root element
- Include <part-list> with each voice as a <score-part> (Soprano, Alto, Tenor, Bass)
- Include proper <attributes> per measure: <divisions>, <key> (fifths), <time> (beats, beat-type), <clef>
- For each part, include <measure> elements with <note> entries containing: <pitch> (<step>, <octave>, optional <alter>), <duration>, <type>, and <lyric> if text is present
- Include rests where appropriate
- Use standard MusicXML clefs: treble for Soprano/Alto, bass for Tenor/Bass
- Transcribe ALL notes visible in the image accurately
- Preserve lyrics/syllables under the notes where possible

IMPORTANT: Be extremely precise with pitch (C4 is middle C), duration values, and rhythm.`;
}

function extractMusicXML(
  content: string
): { ok: boolean; rawMusicXML?: string; confidence?: number; error?: string } {
  const xmlBlockMatch = content.match(/```(?:xml)?\s*([\s\S]*?)```/);
  let xml = xmlBlockMatch ? xmlBlockMatch[1].trim() : content.trim();

  if (xml.startsWith('<') && xml.includes('<score-partwise') && xml.includes('</score-partwise')) {
    return { ok: true, rawMusicXML: xml, confidence: 0.85 };
  }

  if (xml.includes('<measure') && xml.includes('<note') && xml.includes('<pitch')) {
    if (!xml.includes('<score-partwise')) {
      xml = `<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Soprano</part-name></score-part>
    <score-part id="P2"><part-name>Alto</part-name></score-part>
    <score-part id="P3"><part-name>Tenor</part-name></score-part>
    <score-part id="P4"><part-name>Bass</part-name></score-part>
  </part-list>
  <part id="P1">${xml}</part>
</score-partwise>`;
    }
    return { ok: true, rawMusicXML: xml, confidence: 0.6 };
  }

  try {
    const jsonMatch = content.match(/```(?:json)?\s*({[\s\S]*?})\s*```/);
    const jsonStr = jsonMatch ? jsonMatch[1] : content;
    const data = JSON.parse(jsonStr);
    const foundXml = data.musicxml || data.musicXML || data.xml;
    if (foundXml && (foundXml.includes('<score-partwise') || foundXml.includes('<measure'))) {
      return { ok: true, rawMusicXML: foundXml, confidence: 0.7 };
    }
  } catch { /* not JSON */ }

  return { ok: false, error: 'AI response did not contain valid MusicXML. Try a clearer image.' };
}

// ── Verification ───────────────────────────────────────────────────

export async function verifyTranscription(
  imagePath: string,
  score: ParsedScore
): Promise<VerificationResult> {
  const apiKey = process.env['OPENROUTER_API_KEY'] || '';
  if (!apiKey || !apiKey.startsWith('sk-or-v1-')) {
    return { passed: false, issues: ['No API key'], summary: 'Cannot verify without API key.' };
  }

  try {
    const ext = path.extname(imagePath).toLowerCase();
    let processed: string | null = null;
    let imageBuffer: Buffer;

    if (ext === '.pdf') {
      processed = await pdfToPng(imagePath);
      imageBuffer = fs.readFileSync(processed);
    } else {
      processed = await preprocessImage(imagePath);
      imageBuffer = fs.readFileSync(processed);
    }

    const base64 = imageBuffer.toString('base64');
    if (processed && processed !== imagePath) fs.unlinkSync(processed);

    const scoreSummary = buildScoreSummary(score);

    const body = {
      model: VISION_MODEL,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Verify this choral score transcription. Decoded:\n\n${scoreSummary}\n\nCompare against the image. List discrepancies (wrong notes, missing voices, rhythm/key errors). If correct, say "VERIFIED: OK". Be specific.`,
            },
            {
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${base64}` },
            },
          ],
        },
      ],
      temperature: 0.1,
      max_tokens: 2048,
    };

    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://kwaya.app',
        'X-Title': 'Kwaya Verification',
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text();
      return { passed: false, issues: [`Verification API error: ${resp.status}`], summary: text.substring(0, 200) };
    }

    const json = await resp.json() as any;
    const verdict = json?.choices?.[0]?.message?.content || '';
    const passed = verdict.includes('VERIFIED: OK');

    const issues: string[] = [];
    for (const line of verdict.split('\n')) {
      const trimmed = line.replace(/^[-*\d.\s]+/, '').trim();
      if (trimmed.length > 10 && !trimmed.includes('VERIFIED')) {
        if (/error|wrong|missing|incorrect/i.test(trimmed)) {
          issues.push(trimmed);
        }
      }
    }

    return { passed, issues: issues.slice(0, 10), summary: verdict.substring(0, 500) };
  } catch (err: any) {
    return { passed: false, issues: [`Verification error: ${err.message}`], summary: 'Verification failed.' };
  }
}

function buildScoreSummary(score: ParsedScore): string {
  const lines: string[] = [`Title: ${score.title}`, `Key: ${score.key}`, `Time: ${score.timeSig}`, ''];
  for (const part of score.parts) {
    lines.push(`--- ${part.name} (${part.notes.length} notes) ---`);
    for (const n of part.notes) {
      lines.push(n.rest ? `  ${n.type} rest` : `  ${n.pitch} (${n.type})${n.lyric ? ` "${n.lyric}"` : ''}`);
    }
    lines.push('');
  }
  return lines.join('\n');
}