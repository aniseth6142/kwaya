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

const VISION_MODEL = 'google/gemini-2.0-flash-001';

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
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.tiff': 'image/tiff', '.tif': 'image/tiff', '.bmp': 'image/bmp',
  '.webp': 'image/webp', '.pdf': 'application/pdf',
};

export async function transcribeImage(filePath: string): Promise<OMRResult> {
  const ext = path.extname(filePath).toLowerCase();
  const mime = SUPPORTED_MIME[ext];
  if (!mime) return { ok: false, error: `Unsupported: ${ext}. Use PNG, JPG, TIFF, BMP, WebP, or PDF.` };

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
    const transcription = await callVisionModel(base64, SUPPORTED_MIME['.png']);
    if (!transcription.ok) return transcription;

    const musicXML = transcription.rawMusicXML;
    if (!musicXML) return { ok: false, error: 'AI returned no MusicXML content' };

    let parsed: ParsedScore;
    try { parsed = await parseMusicXML(musicXML); }
    catch (e: any) { return { ok: false, error: `MusicXML parse failed: ${e.message}`, rawMusicXML: musicXML }; }

    return { ok: true, score: parsed, rawMusicXML: musicXML, confidence: transcription.confidence || 0.7 };
  } catch (e: any) {
    return { ok: false, error: `OMR failed: ${e.message}` };
  }
}

async function preprocessImage(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();
  const img = sharp(filePath);
  const metadata = await img.metadata();
  if (!metadata) throw new Error('Could not read image metadata');

  let pipeline = img.png();
  const maxDim = 2048;
  if ((metadata.width || 0) > maxDim || (metadata.height || 0) > maxDim)
    pipeline = pipeline.resize(maxDim, undefined, { fit: 'inside', withoutEnlargement: true });

  pipeline = pipeline.normalise().sharpen();
  const tempPath = filePath.replace(ext, '_processed.png');
  await pipeline.toFile(tempPath);
  return tempPath;
}

// ── PDF → PNG via Python PyMuPDF ──────────────────────────────────

async function pdfToPng(pdfPath: string): Promise<string> {
  const scriptPath = path.resolve(__dirname, '..', 'scripts', 'pdf-to-png.py');
  const outPath = pdfPath.replace('.pdf', '_page1.png');

  if (!fs.existsSync(scriptPath))
    throw new Error(`PDF helper not found at ${scriptPath}`);

  try {
    execSync(`python3 "${scriptPath}" "${pdfPath}" "${outPath}"`, { stdio: 'pipe', timeout: 30000 });
  } catch (e: any) {
    throw new Error(`PDF to PNG conversion failed: ${e.stderr?.toString() || e.message}`);
  }

  if (!fs.existsSync(outPath) || fs.statSync(outPath).size < 500)
    throw new Error('PDF to PNG produced empty output. Ensure the PDF is not encrypted.');

  return outPath;
}

// ── AI Vision call ─────────────────────────────────────────────────

async function callVisionModel(
  imageBase64: string, mimeType: string
): Promise<{ ok: boolean; rawMusicXML?: string; confidence?: number; error?: string }> {
  const apiKey = process.env['OPENROUTER_API_KEY'] || '';
  if (!apiKey || !apiKey.startsWith('sk-or-v1-'))
    return { ok: false, error: 'OpenRouter API key not configured in .env' };

  const body = {
    model: VISION_MODEL,
    messages: [{ role: 'user', content: [
      { type: 'text', text: buildVisionPrompt() },
      { type: 'image_url', image_url: { url: `data:${mimeType};base64,${imageBase64}` } },
    ] }],
    temperature: 0.1, max_tokens: 8192,
  };

  try {
    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST', headers: {
        'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json',
        'HTTP-Referer': 'https://kwaya.app', 'X-Title': 'Kwaya OMR',
      }, body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const t = await resp.text();
      if (resp.status === 402) return { ok: false, error: 'OpenRouter credit required. Add credits at https://openrouter.ai/keys' };
      return { ok: false, error: `Vision API error (${resp.status}): ${t.substring(0, 200)}` };
    }

    const json = await resp.json() as any;
    return extractMusicXML(json?.choices?.[0]?.message?.content || '');
  } catch (e: any) {
    return { ok: false, error: `Vision API call failed: ${e.message}` };
  }
}

// ── Prompt & Response parsing ──────────────────────────────────────

function buildVisionPrompt(): string {
  return `You are a music transcription AI. Transcribe this sheet music image into valid MusicXML (score-partwise version 4.0).

RULES:
- Output ONLY valid MusicXML markup, no explanations or markdown
- Include part-list with Soprano, Alto, Tenor, Bass score-parts
- Include attributes: divisions, key (fifths), time (beats/beat-type), clef
- Each note must have: pitch (step, octave, optional alter), duration, type
- Include lyric for any visible text
- Use treble clef for Soprano/Alto, bass clef for Tenor/Bass
- Transcribe ALL notes. C4 is middle C. Be precise.`;
}

function extractMusicXML(content: string): { ok: boolean; rawMusicXML?: string; confidence?: number; error?: string } {
  const block = content.match(/```(?:xml)?\s*([\s\S]*?)```/);
  let xml = block ? block[1].trim() : content.trim();

  if (xml.includes('<score-partwise') && xml.includes('</score-partwise'))
    return { ok: true, rawMusicXML: xml, confidence: 0.85 };

  if (xml.includes('<measure') && xml.includes('<note')) {
    if (!xml.includes('<score-partwise'))
      xml = `<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Soprano</part-name></score-part><score-part id="P2"><part-name>Alto</part-name></score-part><score-part id="P3"><part-name>Tenor</part-name></score-part><score-part id="P4"><part-name>Bass</part-name></score-part></part-list><part id="P1">${xml}</part></score-partwise>`;
    return { ok: true, rawMusicXML: xml, confidence: 0.6 };
  }

  try {
    const jm = content.match(/```(?:json)?\s*({[\s\S]*?})\s*```/);
    const data = JSON.parse(jm ? jm[1] : content);
    const fx = data.musicxml || data.musicXML || data.xml;
    if (fx && fx.includes('<score-partwise')) return { ok: true, rawMusicXML: fx, confidence: 0.7 };
  } catch { /* not JSON */ }

  return { ok: false, error: 'AI response did not contain valid MusicXML. Try a clearer image.' };
}

// ── Verification ───────────────────────────────────────────────────

export async function verifyTranscription(imagePath: string, score: ParsedScore): Promise<VerificationResult> {
  const apiKey = process.env['OPENROUTER_API_KEY'] || '';
  if (!apiKey || !apiKey.startsWith('sk-or-v1-'))
    return { passed: false, issues: ['No API key'], summary: 'Cannot verify without API key.' };

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

    const body = {
      model: VISION_MODEL,
      messages: [{ role: 'user', content: [
        { type: 'text', text: `Verify choral score transcription against the image:\n${buildScoreSummary(score)}\nSay "VERIFIED: OK" if correct. List discrepancies.` },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${base64}` } },
      ] }],
      temperature: 0.1, max_tokens: 2048,
    };

    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST', headers: {
        'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json',
        'HTTP-Referer': 'https://kwaya.app', 'X-Title': 'Kwaya Verification',
      }, body: JSON.stringify(body),
    });

    if (!resp.ok) return { passed: false, issues: [`API error ${resp.status}`], summary: await resp.text() };

    const json = await resp.json() as any;
    const verdict: string = json?.choices?.[0]?.message?.content || '';
    const passed = verdict.includes('VERIFIED: OK');
    const lines = verdict.split('\n').filter((l: string) => /error|wrong|missing|incorrect/i.test(l));
    const issues = lines.map((l: string) => l.trim()).slice(0, 10);

    return { passed, issues, summary: verdict.substring(0, 500) };
  } catch (e: any) {
    return { passed: false, issues: [`Verification error: ${e.message}`], summary: 'Verification failed.' };
  }
}

function buildScoreSummary(score: ParsedScore): string {
  const lines = [`Title: ${score.title}`, `Key: ${score.key}`, `Time: ${score.timeSig}`, ''];
  for (const p of score.parts) {
    lines.push(`--- ${p.name} (${p.notes.length} notes) ---`);
    for (const n of p.notes)
      lines.push(n.rest ? `  ${n.type} rest` : `  ${n.pitch} (${n.type})${n.lyric ? ` "${n.lyric}"` : ''}`);
    lines.push('');
  }
  return lines.join('\n');
}