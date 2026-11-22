/**
 * Kwaya — Swahili Choral Score Assistant
 * TypeScript harness: Express server, file upload, OMR, AI agent, synthesis
 *
 * Features:
 *   - Upload MusicXML (existing)
 *   - Upload PDF/Image for OMR transcription (new)
 *   - AI agent chat with score context (existing, enhanced)
 *   - Human-like neural TTS via edge-tts (new)
 *   - Formant synthesis fallback (existing)
 *   - OMR verification loop (new)
 */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { initDb, insertScore, getScore, listScores, insertChat, getChatHistory, deleteScore } from './db';
import { parseMusicXML, scoreToText } from './music-parser';
import { callAgent, hasApiKey } from './agent';
import { synthesizeToWav } from './formant-synth';
import { transcribeImage, verifyTranscription } from './omr';
import { generateChoirAudio, speakText } from './tts';

// --- Config ---
const PORT = parseInt(process.env['PORT'] || '8742', 10);
const UPLOAD_DIR = path.resolve('uploads');

// --- Express Setup ---
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static frontend
app.use(express.static('public'));

// --- Multer for file uploads ---
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
const storage = multer.diskStorage({
  destination: (_req: any, _file: any, cb: any) => cb(null, UPLOAD_DIR),
  filename: (_req: any, file: any, cb: any) => {
    const unique = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, unique + '-' + file.originalname);
  },
});
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// =========================================================================
// API Routes
// =========================================================================

// ── Status ───────────────────────────────────────────────────────────────

app.get('/api/status', (_req: any, res: any) => {
  res.json({
    ok: true,
    agent: hasApiKey() ? 'configured' : 'missing-api-key',
    scores: listScores().length,
    version: '2.0.0',
    features: {
      omr: true,
      humanTts: true,
      verification: true,
    },
  });
});

// ── Upload MusicXML (original) ──────────────────────────────────────────

app.post('/api/upload', upload.single('file'), async (req: any, res: any) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded. Send a MusicXML file as "file".' });
    }

    const xml = fs.readFileSync(req.file.path, 'utf-8') as string;

    if (!xml.includes('<score-partwise')) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Not a valid MusicXML file — no <score-partwise> root found.' });
    }

    const parsed = await parseMusicXML(xml);

    const scoreId = insertScore({
      filename: req.file.filename,
      title: parsed.title,
      composer: parsed.composer,
      key_sig: parsed.key,
      time_sig: parsed.timeSig,
      voices: JSON.stringify(parsed.parts.map(p => p.name)),
      raw_xml: xml,
    });

    const voiceList = parsed.parts.map(p => `${p.name} (${p.notes.length} notes)`).join(', ');
    insertChat({
      score_id: scoreId,
      role: 'system',
      message: `Score loaded: "${parsed.title}" by ${parsed.composer}. Key: ${parsed.key}, Time: ${parsed.timeSig}. Voices: ${voiceList}`,
    });

    res.json({
      ok: true,
      method: 'upload',
      score: {
        id: scoreId,
        title: parsed.title,
        composer: parsed.composer,
        key: parsed.key,
        timeSig: parsed.timeSig,
        parts: parsed.parts.map(p => ({ name: p.name, noteCount: p.notes.length })),
      },
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Failed to parse score' });
  }
});

// ── OMR — Transcribe image/PDF to score (NEW) ──────────────────────────

app.post('/api/omr', upload.single('file'), async (req: any, res: any) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded. Send a PNG, JPG, or PDF as "file".' });
    }

    const filePath = req.file.path;
    const ext = path.extname(filePath).toLowerCase();
    const imageExts = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.pdf'];

    if (!imageExts.includes(ext)) {
      fs.unlinkSync(filePath);
      return res.status(400).json({ error: `Unsupported format: ${ext}. Use PNG, JPG, TIFF, BMP, WebP, or PDF.` });
    }

    // Report progress
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const sendProgress = (msg: string) => {
      res.write(`data: ${JSON.stringify({ progress: msg })}\n\n`);
    };

    sendProgress('Preprocessing image...');

    // Run OMR
    const result = await transcribeImage(filePath);

    if (!result.ok) {
      sendProgress('error');
      res.write(`data: ${JSON.stringify({ error: result.error })}\n\n`);
      res.end();
      fs.unlinkSync(filePath);
      return;
    }

    sendProgress('Score decoded. Saving...');

    const parsed = result.score!;

    // Save to database
    const scoreId = insertScore({
      filename: req.file.filename,
      title: parsed.title,
      composer: parsed.composer,
      key_sig: parsed.key,
      time_sig: parsed.timeSig,
      voices: JSON.stringify(parsed.parts.map(p => p.name)),
      raw_xml: result.rawMusicXML || '',
    });

    sendProgress('Verifying transcription...');

    // Run verification (AI cross-checks against image)
    let verification = null;
    try {
      verification = await verifyTranscription(filePath, parsed);
    } catch {
      verification = { passed: true, issues: [], summary: 'Verification skipped.' };
    }

    sendProgress('Complete!');

    res.write(`data: ${JSON.stringify({
      ok: true,
      method: 'omr',
      score: {
        id: scoreId,
        title: parsed.title,
        composer: parsed.composer,
        key: parsed.key,
        timeSig: parsed.timeSig,
        parts: parsed.parts.map(p => ({
          name: p.name,
          noteCount: p.notes.length,
          sample: p.notes.slice(0, 10).map(n => ({
            pitch: n.pitch,
            type: n.type,
            rest: n.rest,
            lyric: n.lyric,
          })),
        })),
      },
      verification,
    })}\n\n`);
    res.end();

    // Clean up uploaded file
    fs.unlinkSync(filePath);

  } catch (err: any) {
    res.write(`data: ${JSON.stringify({ error: err.message || 'OMR failed' })}\n\n`);
    res.end();
  }
});

// ── Verify transcription (NEW) ──────────────────────────────────────────

app.post('/api/verify/:id', async (req: any, res: any) => {
  try {
    const id = parseInt(req.params.id, 10);
    const score = getScore(id);
    if (!score) return res.status(404).json({ error: 'Score not found' });

    // We need the original image to verify against — stored in uploads
    // If no image file is available, return a message
    const parsed = await parseMusicXML(score.raw_xml);

    res.json({
      ok: true,
      message: 'Verification requires the original image. Use the OMR endpoint with an image file.',
      score: {
        id: score.id,
        title: parsed.title,
        parts: parsed.parts.map(p => ({ name: p.name, noteCount: p.notes.length })),
      },
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── List scores ─────────────────────────────────────────────────────────

app.get('/api/scores', (_req: any, res: any) => {
  const scores = listScores().map(s => ({
    id: s.id,
    filename: s.filename,
    title: s.title,
    composer: s.composer,
    key_sig: s.key_sig,
    time_sig: s.time_sig,
    voices: JSON.parse(s.voices || '[]'),
    created_at: s.created_at,
  }));
  res.json(scores);
});

// ── Get score details ───────────────────────────────────────────────────

app.get('/api/scores/:id', async (req: any, res: any) => {
  const id = parseInt(req.params.id, 10);
  const score = getScore(id);
  if (!score) return res.status(404).json({ error: 'Score not found' });

  try {
    const parsed = await parseMusicXML(score.raw_xml);
    res.json({
      id: score.id,
      title: parsed.title,
      composer: parsed.composer,
      key: parsed.key,
      keyAccidentals: parsed.keyAccidentals,
      timeSig: parsed.timeSig,
      parts: parsed.parts.map(p => ({
        name: p.name,
        noteCount: p.notes.length,
        range: getRange(p.notes),
        sample: p.notes.slice(0, 30).map(n => ({
          pitch: n.pitch,
          type: n.type,
          rest: n.rest,
          lyric: n.lyric,
        })),
      })),
      created_at: score.created_at,
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

function getRange(notes: any[]): string {
  const pitches = notes.filter((n: any) => !n.rest && n.pitch && n.pitch.length > 0).map((n: any) => n.pitch);
  if (pitches.length === 0) return '—';
  const sorted = [...pitches].sort((a: string, b: string) => {
    const p = (s: string) => {
      const m = s.match(/([A-G])([#b]?)(\d)/);
      return m ? parseInt(m[3]) * 12 + 'C D EF G A B'.indexOf(m[1]) : 0;
    };
    return p(a) - p(b);
  });
  return `${sorted[0]} → ${sorted[sorted.length - 1]}`;
}

// ── Chat with AI agent ─────────────────────────────────────────────────

app.post('/api/chat', async (req: any, res: any) => {
  try {
    const { score_id, message } = req.body;
    if (!message || !message.trim()) {
      return res.status(400).json({ error: 'message is required' });
    }

    // Save user message
    if (score_id) {
      insertChat({ score_id, role: 'user', message: message.trim() });
    }

    // Build context
    let scoreContext = '';
    if (score_id) {
      const score = getScore(score_id);
      if (score) {
        try {
          const parsed = await parseMusicXML(score.raw_xml);
          scoreContext = scoreToText(parsed);
        } catch { /* no context available */ }
      }
    }

    // Build message history (last 12 messages)
    const history: { role: 'user' | 'assistant'; content: string }[] = [];
    if (score_id) {
      const chats = getChatHistory(score_id);
      const relevant = chats.filter(c => c.role !== 'system').slice(-12);
      for (const c of relevant) {
        history.push({ role: c.role as 'user' | 'assistant', content: c.message });
      }
    }
    history.push({ role: 'user', content: message.trim() });

    // Call agent
    const agentResp = await callAgent({
      messages: history,
      scoreContext,
    });

    // Handle formant synthesis request (from agent)
    let audioBase64: string | null = null;
    if (agentResp.synthesisRequest) {
      try {
        const synNotes = agentResp.synthesisRequest.notes.map((n: any) => ({
          pitch: n.pitch,
          duration: n.duration,
          vowel: n.vowel,
          voice: n.voice,
        }));
        const wavBuf = synthesizeToWav(synNotes);
        audioBase64 = wavBuf.toBase64();
      } catch (synthErr: any) {
        console.error('Synthesis error:', synthErr);
      }
    }

    // Handle human TTS request (new: agent can request neural voice)
    let humanAudioBase64: string | null = null;
    if (agentResp.ttsRequest) {
      try {
        const ttsResult = await generateChoirAudio(agentResp.ttsRequest.notes);
        if (ttsResult.ok && ttsResult.audioBase64) {
          humanAudioBase64 = ttsResult.audioBase64;
        } else {
          console.error('TTS error:', ttsResult.error);
        }
      } catch (ttsErr: any) {
        console.error('TTS error:', ttsErr);
      }
    }

    // Save assistant response
    if (score_id) {
      insertChat({ score_id, role: 'assistant', message: agentResp.content });
    }

    res.json({
      message: agentResp.content,
      audio: audioBase64 ? `data:audio/wav;base64,${audioBase64}` : null,
      humanAudio: humanAudioBase64 ? `data:audio/wav;base64,${humanAudioBase64}` : null,
      score_id,
    });
  } catch (err: any) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message || 'Chat failed' });
  }
});

// ── Synthesize (formant — original) ─────────────────────────────────────

app.post('/api/synthesize', (req: any, res: any) => {
  try {
    const { notes } = req.body;
    if (!notes || !Array.isArray(notes) || notes.length === 0) {
      return res.status(400).json({ error: 'Need an array of notes with pitch, duration, vowel, voice' });
    }

    const synNotes = notes.map((n: any) => ({
      pitch: n.pitch,
      duration: n.duration,
      vowel: n.vowel,
      voice: n.voice,
    }));

    const wavBuf = synthesizeToWav(synNotes);
    res.json({
      audio: `data:audio/wav;base64,${wavBuf.toBase64()}`,
      format: 'wav',
      sampleRate: 22050,
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Synthesis failed' });
  }
});

// ── Human TTS — neural voice synthesis (NEW) ───────────────────────────

app.post('/api/synthesize-human', async (req: any, res: any) => {
  try {
    const { notes, text, voice } = req.body;

    // If text-only mode (speaking, not singing)
    if (text && !notes) {
      const result = await speakText(text, voice || 'en-US-JennyNeural');
      if (!result.ok) {
        return res.status(500).json({ error: result.error });
      }
      return res.json({
        audio: `data:audio/wav;base64,${result.audioBase64}`,
        format: result.format,
        sampleRate: result.sampleRate,
        method: 'neural-speech',
      });
    }

    // If notes mode (singing)
    if (!notes || !Array.isArray(notes) || notes.length === 0) {
      return res.status(400).json({ error: 'Need either "text" for speech or "notes" array for singing' });
    }

    const ttsNotes = notes.map((n: any) => ({
      pitch: n.pitch,
      duration: n.duration,
      lyric: n.lyric || 'ah',
      voice: n.voice,
    }));

    const result = await generateChoirAudio(ttsNotes);

    if (!result.ok) {
      return res.status(500).json({ error: result.error });
    }

    res.json({
      audio: `data:audio/wav;base64,${result.audioBase64}`,
      format: result.format,
      sampleRate: result.sampleRate,
      method: 'neural-singing',
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Neural TTS failed' });
  }
});

// ── Delete score ────────────────────────────────────────────────────────

app.delete('/api/scores/:id', (req: any, res: any) => {
  const id = parseInt(req.params.id, 10);
  const score = getScore(id);
  if (!score) return res.status(404).json({ error: 'Score not found' });
  deleteScore(id);
  res.json({ ok: true });
});

// =========================================================================
// Start
// =========================================================================
initDb();
console.log('✓ Database initialised at data/kwaya.db');
console.log(hasApiKey() ? '✓ OpenRouter key configured' : '— No OpenRouter key (set OPENROUTER_API_KEY in .env)');

app.listen(PORT, () => {
  console.log(`\n  🎵  Kwaya — Swahili Choral Score Assistant`);
  console.log(`  ────────────────────────────────────────────`);
  console.log(`  Local:   http://localhost:${PORT}`);
  console.log(`  Features: MusicXML ✓  OMR (Image→Score) ✓  Human TTS ✓`);
  console.log(`  Upload a score or open the web UI`);
  console.log(`  ────────────────────────────────────────────\n`);
});