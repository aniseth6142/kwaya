/**
 * OMR — Optical Music Recognition for Kwaya.
 * Reads PDF/PNG/JPG sheet music images and produces structured score data
 * using AI vision transcription via OpenRouter.
 *
 * Flow: image → sharp preprocessing → AI vision → MusicXML → parsed score
 */

import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { parseMusicXML, ParsedScore } from './music-parser';

// ── AI Vision Model ────────────────────────────────────────────────
// Gemini 2.0 Flash has vision + free tier on many OpenRouter providers
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

// ── Supported image formats ────────────────────────────────────────

const SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'];
const SUPPORTED_MIME = [
  'image/png', 'image/jpeg', 'image/tiff',
  'image/bmp', 'image/webp',
];

// ── Main OMR entry point ───────────────────────────────────────────

/**
 * Transcribe a sheet music image into a ParsedScore.
 * @param filePath Path to the image or PDF file
 * @returns OMRResult with parsed score
 */
export async function transcribeImage(filePath: string): Promise<OMRResult> {
  const ext = path.extname(filePath).toLowerCase();
  const mime = mimeFromExt(ext);

  if (!mime) {
    return { ok: false, error: `Unsupported format: ${ext}. Use PNG, JPG, TIFF, BMP, or WebP.` };
  }

  try {
    // Preprocess image — enhance for better OCR
    const processedPath = await preprocessImage(filePath);

    // Read and base64-encode
    const imageBuffer = fs.readFileSync(processedPath);
    const base64 = imageBuffer.toString('base64');

    // Call AI vision for transcription
    const transcription = await callVisionModel(base64, mime);

    if (!transcription.ok) {
      // Clean up temp file
      if (processedPath !== filePath) fs.unlinkSync(processedPath);
      return transcription;
    }

    // The AI should return MusicXML markup
    const musicXML = transcription.rawMusicXML;
    if (!musicXML) {
      if (processedPath !== filePath) fs.unlinkSync(processedPath);
      return { ok: false, error: 'AI returned no MusicXML content' };
    }

    // Parse MusicXML into our internal format
    let parsed: ParsedScore;
    try {
      parsed = await parseMusicXML(musicXML);
    } catch (parseErr: any) {
      if (processedPath !== filePath) fs.unlinkSync(processedPath);
      return {
        ok: false,
        error: `Failed to parse AI-generated MusicXML: ${parseErr.message}`,
        rawMusicXML: musicXML,
      };
    }

    // Clean up temp preprocessed file if different from original
    if (processedPath !== filePath) fs.unlinkSync(processedPath);

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

/**
 * Preprocess the image for better AI vision recognition:
 * - Convert to PNG
 * - Resize if too large (max 2048px on longest side)
 * - Enhance contrast
 * - Convert to grayscale
 */
async function preprocessImage(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();
  const isPdf = ext === '.pdf';

  if (isPdf) {
    // For PDFs, convert first page to PNG using ffmpeg
    return await pdfToPng(filePath);
  }

  // For images, process with sharp
  const img = sharp(filePath);
  const metadata = await img.metadata();

  if (!metadata) {
    throw new Error('Could not read image metadata');
  }

  // Build processing pipeline
  let pipeline = img;

  // Convert to PNG for consistency
  pipeline = pipeline.png();

  // Resize if larger than 2048px on longest side (vision model limits)
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

  // Normalize and sharpen for better readability
  pipeline = pipeline.normalise().sharpen();

  // Output to temp file
  const tempPath = filePath.replace(ext, '_processed.png');
  await pipeline.toFile(tempPath);

  return tempPath;
}

/**
 * Convert first page of PDF to PNG using ffmpeg.
 */
async function pdfToPng(pdfPath: string): Promise<string> {
  const outPath = pdfPath.replace('.pdf', '_page1.png');
  const { execSync } = require('child_process');

  try {
    execSync(
      `ffmpeg -y -i "${pdfPath}" -vf "scale=2048:-1" -frames:v 1 "${outPath}" 2>/dev/null`,
      { stdio: 'pipe', timeout: 30000 }
    );
  } catch {
    // Try alternative: use density flag
    execSync(
      `ffmpeg -y -density 300 -i "${pdfPath}" -vf "scale=2048:-1" -frames:v 1 "${outPath}" 2>/dev/null`,
      { stdio: 'pipe', timeout: 30000 }
    );
  }

  if (!fs.existsSync(outPath)) {
    throw new Error('Failed to convert PDF to image. Ensure the PDF is not encrypted.');
  }

  return outPath;
}

// ── AI Vision call ─────────────────────────────────────────────────

/**
 * Send the image to a vision-capable AI model for transcription.
 * The model should return MusicXML markup.
 */
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
        return { ok: false, error: `OpenRouter credit required. Add credits at https://openrouter.ai/keys` };
      }
      return { ok: false, error: `Vision API error (${resp.status}): ${text.substring(0, 200)}` };
    }

    const json = await resp.json() as any;
    const content = json?.choices?.[0]?.message?.content || '';

    // Extract MusicXML from the response
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
- For SATB choral music, use 4 parts: Soprano (staff 1), Alto (staff 2), Tenor (staff 3), Bass (staff 4)
- If there are fewer than 4 voices, still use appropriate clefs
- Transcribe ALL notes visible in the image accurately
- Preserve lyrics/syllables under the notes where visible

IMPORTANT: Be extremely precise with pitch (C4 is middle C), duration values, and rhythm. This will be used for audio synthesis.`;
}

/**
 * Extract MusicXML from AI response text.
 * The AI might wrap it in markdown code blocks or just output it directly.
 */
function extractMusicXML(
  content: string
): { ok: boolean; rawMusicXML?: string; confidence?: number; error?: string } {
  // Try to extract from markdown code blocks first
  const xmlBlockMatch = content.match(/```(?:xml)?\s*([\s\S]*?)```/);
  let xml = xmlBlockMatch ? xmlBlockMatch[1].trim() : content.trim();

  // Verify it looks like valid MusicXML
  if (xml.startsWith('<') && xml.includes('<score-partwise') && xml.includes('</score-partwise')) {
    return { ok: true, rawMusicXML: xml, confidence: 0.85 };
  }

  // If it has partial MusicXML, try to wrap it
  if (xml.includes('<measure') && xml.includes('<note') && xml.includes('<pitch')) {
    // It contains note data but no proper wrapper — try to salvage
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

  // If AI returned JSON instead, try to parse it
  try {
    const jsonMatch = content.match(/```(?:json)?\s*({[\s\S]*?})\s*```/);
    const jsonStr = jsonMatch ? jsonMatch[1] : content;
    const data = JSON.parse(jsonStr);

    // Check for MusicXML field
    if (data.musicxml || data.musicXML || data.xml) {
      const foundXml = data.musicxml || data.musicXML || data.xml;
      if (foundXml.includes('<score-partwise') || foundXml.includes('<measure')) {
        return { ok: true, rawMusicXML: foundXml, confidence: 0.7 };
      }
    }
  } catch {
    // Not JSON either
  }

  return { ok: false, error: 'AI response did not contain valid MusicXML. Try a clearer image.' };
}

// ── Verification ───────────────────────────────────────────────────

/**
 * Verify a decoded score against the original image by asking the AI
 * to cross-check. This catches transcription errors.
 */
export async function verifyTranscription(
  imagePath: string,
  score: ParsedScore
): Promise<VerificationResult> {
  const apiKey = process.env['OPENROUTER_API_KEY'] || '';
  if (!apiKey || !apiKey.startsWith('sk-or-v1-')) {
    return { passed: false, issues: ['No API key'], summary: 'Cannot verify without API key.' };
  }

  try {
    const processedPath = await preprocessImage(imagePath);
    const imageBuffer = fs.readFileSync(processedPath);
    const base64 = imageBuffer.toString('base64');
    if (processedPath !== imagePath) fs.unlinkSync(processedPath);

    // Build score summary for the verification prompt
    const scoreSummary = buildScoreSummary(score);

    const body = {
      model: VISION_MODEL,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Verify this choral score transcription. Here is what was decoded:\n\n${scoreSummary}\n\nCompare against the original image. List any discrepancies found (wrong notes, missing voices, incorrect rhythms, key/time signature errors). If correct, say "VERIFIED: OK". Be specific about measure numbers and voices.`,
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${base64}`,
              },
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

    // Extract issues
    const issues: string[] = [];
    const lines = verdict.split('\n');
    for (const line of lines) {
      const trimmed = line.replace(/^[-*\d.\s]+/, '').trim();
      if (trimmed.length > 10 && !trimmed.includes('VERIFIED') && !trimmed.includes('OK')) {
        if (trimmed.toLowerCase().includes('error') || trimmed.toLowerCase().includes('wrong') ||
            trimmed.toLowerCase().includes('missing') || trimmed.toLowerCase().includes('incorrect')) {
          issues.push(trimmed);
        }
      }
    }

    return {
      passed,
      issues: issues.slice(0, 10),
      summary: verdict.substring(0, 500),
    };
  } catch (err: any) {
    return { passed: false, issues: [`Verification error: ${err.message}`], summary: 'Verification failed.' };
  }
}

function buildScoreSummary(score: ParsedScore): string {
  const lines: string[] = [];
  lines.push(`Title: ${score.title}`);
  lines.push(`Key: ${score.key} (fifths=${score.keyAccidentals})`);
  lines.push(`Time: ${score.timeSig}`);
  lines.push('');

  for (const part of score.parts) {
    lines.push(`--- ${part.name} (${part.notes.length} notes) ---`);
    const noteLines = part.notes.map(n => {
      if (n.rest) return `  ${n.type} rest`;
      return `  ${n.pitch} (${n.type})${n.lyric ? ` "${n.lyric}"` : ''}`;
    });
    lines.push(...noteLines);
    lines.push('');
  }

  return lines.join('\n');
}

// ── Helpers ────────────────────────────────────────────────────────

function mimeFromExt(ext: string): string | null {
  const map: Record<string, string> = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    '.pdf': 'application/pdf',
  };
  return map[ext] || null;
}