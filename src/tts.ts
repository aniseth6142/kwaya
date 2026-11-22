/**
 * Kwaya Human-like TTS — Neural voice synthesis for choral SATB singing.
 * Uses Microsoft Edge TTS (edge-tts via Python) for human-like voices,
 * combined with formant synthesis for pitch accuracy.
 *
 * Edge TTS voices used per SATB part:
 *   Soprano — en-US-JennyNeural (female, bright)
 *   Alto    — en-US-MichelleNeural (female, warm)
 *   Tenor   — en-US-AndrewNeural (male, warm)
 *   Bass    — en-US-ChristopherNeural (male, deep)
 */

import { execSync, spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { synthesizeToWav, SynthNote } from './formant-synth';

// ── Voice mappings ─────────────────────────────────────────────────

const VOICE_MAP: Record<string, string> = {
  Soprano: 'en-US-JennyNeural',
  Alto: 'en-US-MichelleNeural',
  Tenor: 'en-US-AndrewNeural',
  Bass: 'en-US-ChristopherNeural',
};

const DEFAULT_VOICE = 'en-US-JennyNeural';

// ── Types ──────────────────────────────────────────────────────────

export interface TTSNote {
  pitch: string;       // e.g. "C4", "F#5"
  duration: number;    // in seconds
  lyric: string;       // the syllable/solfege to sing
  voice: string;       // Soprano, Alto, Tenor, Bass
}

export interface TTSResult {
  ok: boolean;
  audioBase64: string | null;
  format: string;
  sampleRate: number;
  error?: string;
}

// ── Main TTS entry point ───────────────────────────────────────────

/**
 * Generate human-like choral audio from a set of notes.
 * Uses hybrid approach: edge-tts neural voice + formant tonal backing.
 */
export async function generateChoirAudio(notes: TTSNote[]): Promise<TTSResult> {
  if (notes.length === 0) {
    return { ok: false, audioBase64: null, format: 'wav', sampleRate: 24000, error: 'No notes provided' };
  }

  // Group notes by voice
  const byVoice: Record<string, TTSNote[]> = {};
  for (const n of notes) {
    if (!byVoice[n.voice]) byVoice[n.voice] = [];
    byVoice[n.voice].push(n);
  }

  const voiceOrder = ['Soprano', 'Alto', 'Tenor', 'Bass'];
  const voiceWavs: string[] = [];

  try {
    // Generate audio for each voice
    for (const voice of voiceOrder) {
      if (!byVoice[voice] || byVoice[voice].length === 0) continue;

      const voiceNotes = byVoice[voice];
      const wav = await generateVoiceAudio(voiceNotes, voice);
      if (wav) voiceWavs.push(wav);
    }

    if (voiceWavs.length === 0) {
      return { ok: false, audioBase64: null, format: 'wav', sampleRate: 24000, error: 'No audio generated' };
    }

    // Mix all voice WAVs together
    const mixedWav = await mixAudioFiles(voiceWavs);

    // Read the mixed WAV as base64
    const buf = fs.readFileSync(mixedWav);
    const base64 = buf.toString('base64');

    // Clean up temp files
    cleanupTempFiles();

    return {
      ok: true,
      audioBase64: base64,
      format: 'wav',
      sampleRate: 24000,
    };
  } catch (err: any) {
    cleanupTempFiles();
    return { ok: false, audioBase64: null, format: 'wav', sampleRate: 24000, error: `TTS failed: ${err.message}` };
  }
}

// ── Voice audio generation ─────────────────────────────────────────

/**
 * Generate audio for a single voice part.
 * Hybrid: neural TTS for the lyric/human quality + formant for tonal accuracy.
 */
async function generateVoiceAudio(notes: TTSNote[], voiceName: string): Promise<string | null> {
  const templateDir = path.resolve('data/tmp/tts');
  fs.mkdirSync(templateDir, { recursive: true });

  const edgeVoice = VOICE_MAP[voiceName] || DEFAULT_VOICE;
  const totalDuration = notes.reduce((sum, n) => sum + n.duration, 0);

  if (totalDuration < 0.1 || notes.length === 0) return null;

  // ── Method 1: Generate lyric audio with edge-tts ──
  // Speak the lyrics/solfege as connected text with neural voice
  const lyricText = notes
    .map(n => n.lyric || 'ah')
    .join(' ');

  const lyricWav = path.join(templateDir, `${voiceName.toLowerCase()}_lyric.wav`);
  await runEdgeTts(lyricText, edgeVoice, lyricWav);

  // ── Method 2: Formant synthesis for tonal accuracy ──
  const formantNotes: SynthNote[] = notes.map(n => ({
    pitch: n.pitch,
    duration: n.duration,
    vowel: mapLyricToVowel(n.lyric),
    voice: n.voice,
  }));

  // Generate the tonal WAV
  const formantWavPath = path.join(templateDir, `${voiceName.toLowerCase()}_tone.wav`);
  try {
    // Formant synth produces WAV as a custom Buffer with .data (Uint8Array)
    const fResult: any = synthesizeToWav(formantNotes);
    const formantWavFull = Buffer.from(fResult.data.buffer, 0, fResult.data.length);
    fs.writeFileSync(path.join(templateDir, `${voiceName.toLowerCase()}_tone_raw.wav`), formantWavFull);

    execSync(
      `ffmpeg -y -i "${path.join(templateDir, `${voiceName.toLowerCase()}_tone_raw.wav`)}" ` +
      `-ar 24000 -ac 1 -filter:a "volume=0.35" "${formantWavPath}" 2>/dev/null`,
      { timeout: 15000 }
    );
  } catch {
    // If formant fails, just use edge-tts alone
    return lyricWav;
  }

  // ── Mix lyric + tone ──
  const mixedWav = path.join(templateDir, `${voiceName.toLowerCase()}_mixed.wav`);

  try {
    // Stretch/speed the lyric audio to match the total duration
    const lyricDuration = getWavDuration(lyricWav);
    const tempoRatio = totalDuration / Math.max(lyricDuration, 0.1);

    // Use ffmpeg to mix: lyric audio as primary + formant as tonal backing
    execSync(
      `ffmpeg -y ` +
      `-i "${lyricWav}" ` +
      `-i "${formantWavPath}" ` +
      `-filter_complex ` +
      `"[0:a]atempo=${Math.min(Math.max(tempoRatio, 0.5), 2.0)},volume=1.2[a]; ` +
      `[1:a]volume=0.4[b]; ` +
      `[a][b]amix=inputs=2:duration=first:dropout_transition=0[c]" ` +
      `-ac 1 -ar 24000 "${mixedWav}" 2>/dev/null`,
      { timeout: 30000 }
    );

    if (fs.existsSync(mixedWav) && fs.statSync(mixedWav).size > 1000) {
      return mixedWav;
    }
  } catch {
    // If mixing fails, return lyric-only audio
  }

  return lyricWav;
}

// ── Edge TTS execution ─────────────────────────────────────────────

/**
 * Call edge-tts Python CLI to generate audio from text.
 */
async function runEdgeTts(text: string, voice: string, outputPath: string): Promise<void> {
  // Split into manageable chunks if too long (edge-tts has limits)
  const maxChars = 500;
  if (text.length > maxChars) {
    // Generate phrase by phrase and concatenate
    const phrases = chunkText(text, maxChars);
    const chunkFiles: string[] = [];
    const baseName = path.basename(outputPath, '.wav');

    for (let i = 0; i < phrases.length; i++) {
      const chunkPath = path.join(path.dirname(outputPath), `${baseName}_chunk${i}.wav`);
      await runSingleEdgeTts(phrases[i], voice, chunkPath);
      chunkFiles.push(chunkPath);
    }

    // Concatenate chunks with ffmpeg
    if (chunkFiles.length > 1) {
      const concatList = path.join(path.dirname(outputPath), `${baseName}_list.txt`);
      fs.writeFileSync(concatList, chunkFiles.map(f => `file '${f}'`).join('\n'));
      execSync(
        `ffmpeg -y -f concat -safe 0 -i "${concatList}" -ac 1 -ar 24000 "${outputPath}" 2>/dev/null`,
        { timeout: 30000 }
      );
      // Clean chunks
      for (const f of chunkFiles) try { fs.unlinkSync(f); } catch {}
      try { fs.unlinkSync(concatList); } catch {}
    } else if (chunkFiles.length === 1) {
      fs.renameSync(chunkFiles[0], outputPath);
    }
    return;
  }

  await runSingleEdgeTts(text, voice, outputPath);
}

async function runSingleEdgeTts(text: string, voice: string, outputPath: string): Promise<void> {
  const result = spawnSync('edge-tts', [
    '--voice', voice,
    '--text', text,
    '--write-media', outputPath,
  ], {
    timeout: 60000,
    stdio: 'pipe',
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });

  if (result.error || result.status !== 0) {
    throw new Error(`edge-tts failed: ${result.error?.message || result.stderr.toString() || 'unknown error'}`);
  }

  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 100) {
    throw new Error('edge-tts produced empty output');
  }
}

// ── Audio mixing ───────────────────────────────────────────────────

/**
 * Mix multiple WAV files into one using ffmpeg.
 */
async function mixAudioFiles(wavPaths: string[]): Promise<string> {
  const templateDir = path.resolve('data/tmp/tts');
  const outputPath = path.join(templateDir, 'choir_mixed.wav');

  if (wavPaths.length === 0) {
    throw new Error('No audio files to mix');
  }

  if (wavPaths.length === 1) {
    fs.copyFileSync(wavPaths[0], outputPath);
    return outputPath;
  }

  // Build ffmpeg command to mix all files
  let cmd = 'ffmpeg -y ';
  for (const wav of wavPaths) {
    cmd += `-i "${wav}" `;
  }

  // Build filter complex for mixing with per-file volume adjustment
  const inputs = wavPaths.map((_, i) => `[${i}:a]volume=0.8[a${i}]`).join('; ');
  const mixInputs = wavPaths.map((_, i) => `[a${i}]`).join('');
  const mixCount = wavPaths.length;

  cmd += `-filter_complex "${inputs}; ${mixInputs}amix=inputs=${mixCount}:duration=first:dropout_transition=2[out]" -map "[out]" -ac 1 -ar 24000 "${outputPath}" 2>/dev/null`;

  execSync(cmd, { timeout: 60000 });

  if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 100) {
    throw new Error('Audio mixing failed — output is empty');
  }

  return outputPath;
}

/**
 * Get the duration of a WAV file in seconds using ffprobe.
 */
function getWavDuration(wavPath: string): number {
  try {
    const result = execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${wavPath}"`,
      { timeout: 5000, encoding: 'utf-8' }
    );
    return parseFloat(result.trim()) || 1.0;
  } catch {
    return 1.0;
  }
}

// ── Utilities ──────────────────────────────────────────────────────

function mapLyricToVowel(lyric: string): string {
  if (!lyric || lyric.length === 0) return 'a';
  const lastChar = lyric.toLowerCase().slice(-1);
  if ('aeiou'.includes(lastChar)) return lastChar;
  // Common solfege mappings
  const solfege: Record<string, string> = {
    do: 'o', re: 'e', mi: 'i', fa: 'a', so: 'o', la: 'a', ti: 'i',
    sol: 'o', si: 'i', le: 'e', te: 'e',
  };
  const lower = lyric.toLowerCase();
  if (solfege[lower]) return solfege[lower];
  return 'a';
}

/**
 * Split text into chunks at word boundaries.
 */
function chunkText(text: string, maxLen: number): string[] {
  const words = text.split(' ');
  const chunks: string[] = [];
  let current = '';
  for (const word of words) {
    if ((current + ' ' + word).trim().length > maxLen && current.length > 0) {
      chunks.push(current.trim());
      current = word;
    } else {
      current += ' ' + word;
    }
  }
  if (current.trim().length > 0) chunks.push(current.trim());
  return chunks;
}

/**
 * Clean up temporary TTS files.
 */
function cleanupTempFiles(): void {
  const dir = path.resolve('data/tmp/tts');
  try {
    if (fs.existsSync(dir)) {
      const files = fs.readdirSync(dir);
      for (const f of files) {
        try { fs.unlinkSync(path.join(dir, f)); } catch {}
      }
    }
  } catch {
    // Ignore cleanup errors
  }
}

// ── Direct voice synthesis for simple cases ────────────────────────

/**
 * Quick TTS — speak text with a neural voice. Used for simple
 * announcements or explanations.
 */
export async function speakText(text: string, voice: string = 'en-US-JennyNeural'): Promise<TTSResult> {
  try {
    const dir = path.resolve('data/tmp/tts');
    fs.mkdirSync(dir, { recursive: true });
    const outPath = path.join(dir, 'speech.wav');

    await runEdgeTts(text, voice, outPath);
    const buf = fs.readFileSync(outPath);

    cleanupTempFiles();

    return {
      ok: true,
      audioBase64: buf.toString('base64'),
      format: 'wav',
      sampleRate: 24000,
    };
  } catch (err: any) {
    cleanupTempFiles();
    return { ok: false, audioBase64: null, format: 'wav', sampleRate: 24000, error: err.message };
  }
}

/**
 * List available edge-tts voices.
 */
export function listVoices(): string[] {
  try {
    const result = execSync('edge-tts --list-voices', { timeout: 15000, encoding: 'utf-8' });
    const lines = result.split('\n');
    const voices: string[] = [];
    for (const line of lines) {
      const match = line.match(/^([\w-]+Neural)\s+/);
      if (match) voices.push(match[1]);
    }
    return voices;
  } catch {
    return [];
  }
}