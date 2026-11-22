import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import sharp from 'sharp';
import { parseMusicXML, ParsedScore } from './music-parser';

const VISION_MODEL = 'google/gemini-2.5-flash-lite';

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

const MIME: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.tiff': 'image/tiff', '.tif': 'image/tiff', '.bmp': 'image/bmp',
  '.webp': 'image/webp', '.pdf': 'application/pdf',
};

export async function transcribeImage(filePath: string): Promise<OMRResult> {
  const ext = path.extname(filePath).toLowerCase();
  if (!MIME[ext]) return { ok: false, error: `Unsupported: ${ext}` };
  try {
    const isPdf = ext === '.pdf';
    const png = isPdf ? await pdfToPng(filePath) : await preprocessImage(filePath);
    const img = fs.readFileSync(png);
    if (png !== filePath) fs.unlinkSync(png);
    const result = await callVisionModel(img.toString('base64'), 'image/png');
    if (!result.ok) return result;
    if (!result.rawMusicXML) return { ok: false, error: 'No MusicXML from AI' };
    try {
      const score = await parseMusicXML(result.rawMusicXML);
      return { ok: true, score, rawMusicXML: result.rawMusicXML, confidence: result.confidence || 0.7 };
    } catch (e: any) {
      return { ok: false, error: `MusicXML parse: ${e.message}`, rawMusicXML: result.rawMusicXML };
    }
  } catch (e: any) {
    return { ok: false, error: `OMR failed: ${e.message}` };
  }
}

async function preprocessImage(fp: string): Promise<string> {
  const ext = path.extname(fp).toLowerCase();
  const img = sharp(fp);
  const meta = await img.metadata();
  if (!meta) throw new Error('No image metadata');
  let p = img.png();
  const max = 2048;
  if ((meta.width || 0) > max || (meta.height || 0) > max)
    p = p.resize(max, undefined, { fit: 'inside', withoutEnlargement: true });
  const out = fp.replace(ext, '_proc.png');
  await p.normalise().sharpen().toFile(out);
  return out;
}

async function pdfToPng(pdfPath: string): Promise<string> {
  const script = path.resolve(process.cwd(), 'scripts', 'pdf-to-png.py');
  const out = pdfPath.replace('.pdf', '_page1.png');
  if (!fs.existsSync(script)) throw new Error(`Missing script: ${script}`);
  try {
    execSync(`python "${script}" "${pdfPath}" "${out}"`, { stdio: 'pipe', timeout: 30000 });
  } catch (e: any) {
    throw new Error(`PDF to PNG: ${e.stderr?.toString() || e.message}`);
  }
  if (!fs.existsSync(out) || fs.statSync(out).size < 500)
    throw new Error('PDF to PNG empty output');
  return out;
}

async function callVisionModel(b64: string, mime: string) {
  const key = process.env['OPENROUTER_API_KEY'] || '';
  if (!key?.startsWith('sk-or-v1-'))
    return { ok: false, error: 'No API key' } as const;
  const body = {
    model: VISION_MODEL,
    messages: [{ role: 'user', content: [
      { type: 'text', text: `Transcribe this choral sheet music into compact JSON.
ONLY raw JSON, no markdown, no backticks, no explanation text.
Format: {"key":0,"time":"4/4","parts":[{"name":"Soprano","notes":[{"p":"C4","d":4,"t":"quarter","l":"A"},...]}]}
p=pitch like C4 F#5 Bb3 (R=rest), d=duration in quarter-note units (4=quarter, 2=eighth), t=type, l=lyric.
Soprano=treble, Alto=treble, Tenor=bass, Bass=bass clef. ALL voices, ALL notes, precise.` },
      { type: 'image_url', image_url: { url: `data:${mime};base64,${b64}` } },
    ] }],
    temperature: 0.1, max_tokens: 16384,
  };
  const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json',
      'HTTP-Referer': 'https://kwaya.app', 'X-Title': 'Kwaya OMR' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const t = await resp.text();
    if (resp.status === 402) return { ok: false, error: 'OpenRouter credits needed' } as const;
    return { ok: false, error: `Vision API ${resp.status}: ${t.substring(0, 200)}` } as const;
  }
  const json: any = await resp.json();
  return extractMusicXML(json?.choices?.[0]?.message?.content || '');
}

// Compact JSON to MusicXML converter
interface CN { p?: string; d: number; t: string; l?: string }
interface CP { name: string; notes: CN[] }
interface CS { title?: string; key?: number; time?: string; parts: CP[] }

function extractMusicXML(raw: string): { ok: boolean; rawMusicXML?: string; confidence?: number; error?: string } {
  let text = raw;
  const block = raw.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
  if (block) text = block[1];
  const fb = text.indexOf('{');
  if (fb < 0) return { ok: false, error: `No JSON in AI response. First 200: ${raw.substring(0, 200)}` };
  text = text.substring(fb);

  // Auto-close truncated JSON: count brackets and add missing closing ones
  for (let attempt = 0; attempt < 5; attempt++) {
    const noStrings = text.replace(/"([^"]*)"/g, '""');
    const openC = (noStrings.match(/\{/g) || []).length;
    const closeC = (noStrings.match(/\}/g) || []).length;
    const openB = (noStrings.match(/\[/g) || []).length;
    const closeB = (noStrings.match(/\]/g) || []).length;
    const needC = openC - closeC;
    const needB = openB - closeB;
    if (needC <= 0 && needB <= 0) break;
    text += ']}'.repeat(needC) + ']'.repeat(needB);
  }

  try {
    const d: CS = JSON.parse(text);
    if (d.parts?.length) return { ok: true, rawMusicXML: jsonToXML(d), confidence: 0.7 };
  } catch {}

  return { ok: false, error: `Not JSON. First 300: ${raw.substring(0, 300)}` };
}

function jsonToXML(d: CS): string {
  const k = d.key ?? 0;
  const [b, bt] = (d.time || '4/4').split('/').map(Number);
  const pxml = d.parts.map((p, i) => {
    const meas = groupMeasures(p.notes);
    const mx = meas.map((ns, mi) => {
      const a = mi === 0
        ? `<attributes><divisions>4</divisions><key><fifths>${k}</fifths></key><time><beats>${b}</beats><beat-type>${bt}</beat-type></time><clef><sign>${p.name === 'Tenor' || p.name === 'Bass' ? 'F' : 'G'}</sign><line>${p.name === 'Tenor' || p.name === 'Bass' ? 4 : 2}</line></clef></attributes>`
        : '';
      const nx = ns.map(n => {
        if (!n.p || n.p === 'R') return `<note><rest/><duration>${Math.round(n.d * 4)}</duration><type>${n.t}</type></note>`;
        const m = n.p.match(/^([A-G])([#b]?)(\d+)$/);
        if (!m) return '';
        const al = m[2] === '#' ? '<alter>1</alter>' : m[2] === 'b' ? '<alter>-1</alter>' : '';
        const ly = n.l ? `<lyric><text>${n.l}</text></lyric>` : '';
        return `<note><pitch><step>${m[1]}</step>${al}<octave>${m[3]}</octave></pitch><duration>${Math.round(n.d * 4)}</duration><type>${n.t}</type>${ly}</note>`;
      }).join('');
      return `<measure number="${mi + 1}">${a}${nx}</measure>`;
    }).join('');
    return `<part id="P${i + 1}">${mx}</part>`;
  }).join('');
  return `<score-partwise version="4.0"><part-list>${d.parts.map((p, i) => `<score-part id="P${i + 1}"><part-name>${p.name}</part-name></score-part>`).join('')}</part-list>${pxml}</score-partwise>`;
}

function groupMeasures(notes: CN[]): CN[][] {
  const m: CN[][] = []; let cur: CN[] = [], a = 0;
  for (const n of notes) { cur.push(n); a += n.d; if (Math.abs(a - 4) < 0.01) { m.push(cur); cur = []; a = 0; } }
  if (cur.length) m.push(cur);
  return m;
}

// Verification
export async function verifyTranscription(imagePath: string, score: ParsedScore): Promise<VerificationResult> {
  const key = process.env['OPENROUTER_API_KEY'] || '';
  if (!key?.startsWith('sk-or-v1-'))
    return { passed: false, issues: ['No key'], summary: '' };
  try {
    const ext = path.extname(imagePath).toLowerCase();
    const png = ext === '.pdf' ? await pdfToPng(imagePath) : await preprocessImage(imagePath);
    const img = fs.readFileSync(png);
    if (png !== imagePath) fs.unlinkSync(png);
    const body = {
      model: VISION_MODEL, temperature: 0.1, max_tokens: 2048,
      messages: [{ role: 'user', content: [
        { type: 'text', text: `Verify this transcription:\n${buildSummary(score)}\nSay "VERIFIED: OK" if correct.` },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${img.toString('base64')}` } },
      ] }],
    };
    const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json',
        'HTTP-Referer': 'https://kwaya.app', 'X-Title': 'Kwaya Verification' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) return { passed: false, issues: [`API ${resp.status}`], summary: await resp.text() };
    const j: any = await resp.json();
    const v: string = j?.choices?.[0]?.message?.content || '';
    return { passed: v.includes('VERIFIED: OK'), issues: [], summary: v.substring(0, 500) };
  } catch (e: any) {
    return { passed: false, issues: [e.message], summary: '' };
  }
}

function buildSummary(s: ParsedScore): string {
  const ls = [`${s.title}  Key:${s.key}  Time:${s.timeSig}`];
  for (const p of s.parts) {
    ls.push(`${p.name} (${p.notes.length}):`);
    for (const n of p.notes) ls.push(n.rest ? `  ${n.type} rest` : `  ${n.pitch} (${n.type})${n.lyric ? ` "${n.lyric}"` : ''}`);
  }
  return ls.join('\n');
}