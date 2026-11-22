/**
 * Pure-TypeScript formant synthesis engine for choral voices.
 * Generates WAV audio bytes with realistic singing tone.
 * Uses formant filtering, vibrato, harmonic partials, and breath noise.
 */

const VOWEL_FORMANTS: Record<string, [number, number, number]> = {
  a:  [800, 1200, 2500],
  e:  [500, 1800, 2500],
  i:  [300, 2300, 3000],
  o:  [500, 900,  2500],
  u:  [300, 900,  2300],
  ae: [750, 1600, 2600],
  oe: [550, 1400, 2500],
};

const DEFAULT_VOWEL: [number, number, number] = [600, 1400, 2500];

const VOICE_SCALE: Record<string, { f1: number; f2: number; f3: number }> = {
  Soprano: { f1: 1.12, f2: 1.10, f3: 1.05 },
  Alto:    { f1: 1.05, f2: 1.04, f3: 1.02 },
  Tenor:   { f1: 0.95, f2: 0.96, f3: 0.98 },
  Bass:    { f1: 0.85, f2: 0.88, f3: 0.93 },
};

const DEFAULT_SCALE = { f1: 1.0, f2: 1.0, f3: 1.0 };

export interface SynthNote {
  pitch: string;
  duration: number;
  vowel: string;
  voice: string;
}

function pitchToHz(pitch: string): number {
  const m = pitch.match(/^([A-G])([#b]?)(-?\d+)$/);
  if (!m) return 440;
  const step = m[1];
  const acc = m[2];
  const oct = parseInt(m[3]);
  const semitones: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  let n = semitones[step] + (oct - 4) * 12;
  if (acc === '#') n += 1;
  if (acc === 'b') n -= 1;
  return 440 * Math.pow(2, (n - 9) / 12);
}

function formantWeight(harmFreq: number, formantFreq: number, bw: number): number {
  const ratio = harmFreq / formantFreq;
  if (ratio < 0.01 || ratio > 10) return 0;
  const semitoneDist = 12 * Math.log2(ratio);
  const spread = bw / 12;
  return Math.exp(-0.5 * (semitoneDist / spread) ** 2);
}

export function synthesizeToWav(notes: SynthNote[]): Buffer {
  const SAMPLE_RATE = 22050;
  const NUM_HARMONICS = 12;

  if (notes.length === 0) {
    const buf = Buffer.alloc(SAMPLE_RATE * 2 + 44);
    return buildWavHeader(buf, SAMPLE_RATE, SAMPLE_RATE);
  }

  // Sort notes by (voice priority, start) for deterministic output
  let totalDur = 0;
  const voiceGroups: Record<string, SynthNote[]> = {};
  for (const n of notes) {
    if (n.duration > totalDur) totalDur = n.duration;
    if (!voiceGroups[n.voice]) voiceGroups[n.voice] = [];
    voiceGroups[n.voice].push(n);
  }

  totalDur += 0.5; // padding
  const totalSamples = Math.ceil(totalDur * SAMPLE_RATE);
  const mix = new Float64Array(totalSamples);

  const VOICE_ORDER = ['Soprano', 'Alto', 'Tenor', 'Bass'];
  const orderedVoices = Object.keys(voiceGroups).sort(
    (a, b) => VOICE_ORDER.indexOf(a) - VOICE_ORDER.indexOf(b)
  );

  for (const voice of orderedVoices) {
    const vNotes = voiceGroups[voice];
    const scale = VOICE_SCALE[voice] || DEFAULT_SCALE;
    const mixGain = voice === 'Soprano' ? 0.35 : voice === 'Alto' ? 0.30 : voice === 'Tenor' ? 0.25 : 0.25;

    for (const note of vNotes) {
      const freq = pitchToHz(note.pitch);
      if (freq < 50 || freq > 2000) continue;

      const formants = VOWEL_FORMANTS[note.vowel] || DEFAULT_VOWEL;
      const f1 = formants[0] * scale.f1;
      const f2 = formants[1] * scale.f2;
      const f3 = formants[2] * scale.f3;

      const durSamples = Math.ceil(note.duration * SAMPLE_RATE);
      const endSample = Math.min(durSamples, totalSamples);

      for (let s = 0; s < endSample; s++) {
        const t = s / SAMPLE_RATE;
        const progress = t / note.duration;
        if (progress > 1) break;

        // Envelope
        let amp = 1.0;
        const attackS = 0.025;
        const releaseS = 0.060;
        if (t < attackS) amp = t / attackS;
        else if (t > note.duration - releaseS) amp = Math.max(0, (note.duration - t) / releaseS);

        // Vibrato
        const vib = 1.0 + 0.008 * Math.sin(2 * Math.PI * 5.5 * t + Math.PI * 0.25);

        // Harmonic series + formants
        let sample = 0.0;
        for (let h = 1; h <= NUM_HARMONICS; h++) {
          const hFreq = freq * h * vib;
          const hAmp = 1.0 / h;
          const fw = Math.max(
            formantWeight(hFreq, f1, 80),
            formantWeight(hFreq, f2, 120),
            formantWeight(hFreq, f3, 160)
          );
          sample += hAmp * fw * Math.sin(2 * Math.PI * hFreq * t + h * 0.3);
        }

        // Breath noise
        const breath = (Math.random() * 2 - 1) * 0.003;

        const finalSample = (sample * 0.6 + breath) * amp * mixGain;
        mix[s] = Math.max(-0.98, Math.min(0.98, mix[s] + finalSample));
      }
    }
  }

  // Normalize
  let peak = 0.0;
  for (let i = 0; i < totalSamples; i++) {
    const abs = Math.abs(mix[i]);
    if (abs > peak) peak = abs;
  }
  if (peak > 0.0) {
    const norm = Math.min(1.0 / peak, 0.95);
    for (let i = 0; i < totalSamples; i++) mix[i] *= norm;
  }

  const dataSize = totalSamples * 2;
  const wavSize = 44 + dataSize;
  const buf = Buffer.alloc(wavSize);
  return buildWavData(buf, mix, totalSamples, SAMPLE_RATE);
}

function buildWavData(buf: Buffer, samples: Float64Array, numSamples: number, sr: number): Buffer {
  const dataSize = numSamples * 2;
  const byteRate = sr * 2;

  // RIFF header
  buf.setString(0, 'RIFF');
  buf.setI32(4, 44 + dataSize - 8);
  buf.setString(8, 'WAVE');

  // fmt sub-chunk
  buf.setString(12, 'fmt ');
  buf.setI32(16, 16);
  buf.setI16(20, 1);    // PCM
  buf.setI16(22, 1);    // mono
  buf.setI32(24, sr);
  buf.setI32(28, byteRate);
  buf.setI16(32, 2);    // block align
  buf.setI16(34, 16);   // bits per sample

  // data sub-chunk
  buf.setString(36, 'data');
  buf.setI32(40, dataSize);

  // Write samples
  for (let i = 0; i < numSamples; i++) {
    let val = Math.round(samples[i] * 32767);
    if (val > 32767) val = 32767;
    if (val < -32768) val = -32768;
    const uval = val < 0 ? 65536 + val : val;
    buf.setI16(44 + i * 2, uval);
  }

  return buf;
}

function buildWavHeader(buf: Buffer, numSamples: number, sr: number): Buffer {
  const dataSize = numSamples * 2;
  buf.setString(0, 'RIFF');
  buf.setI32(4, 44 + dataSize - 8);
  buf.setString(8, 'WAVE');
  buf.setString(12, 'fmt ');
  buf.setI32(16, 16);
  buf.setI16(20, 1);
  buf.setI16(22, 1);
  buf.setI32(24, sr);
  buf.setI32(28, sr * 2);
  buf.setI16(32, 2);
  buf.setI16(34, 16);
  buf.setString(36, 'data');
  buf.setI32(40, dataSize);
  for (let i = 0; i < numSamples; i++) {
    buf.setI16(44 + i * 2, 0);
  }
  return buf;
}

/**
 * Minimal buffer wrapper for building WAV files.
 */
class Buffer {
  data!: Uint8Array;

  static alloc(size: number): Buffer {
    const b = new Buffer();
    b.data = new Uint8Array(size);
    return b;
  }

  setString(offset: number, s: string): void {
    for (let i = 0; i < s.length; i++) {
      this.data[offset + i] = s.charCodeAt(i) & 0xFF;
    }
  }

  setI16(offset: number, val: number): void {
    this.data[offset]     = val & 0xFF;
    this.data[offset + 1] = (val >> 8) & 0xFF;
  }

  setI32(offset: number, val: number): void {
    this.data[offset]     = val & 0xFF;
    this.data[offset + 1] = (val >> 8) & 0xFF;
    this.data[offset + 2] = (val >> 16) & 0xFF;
    this.data[offset + 3] = (val >> 24) & 0xFF;
  }

  toBase64(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    let result = '';
    const bytes = this.data;
    for (let i = 0; i < bytes.length; i += 3) {
      const b1 = bytes[i];
      const b2 = i + 1 < bytes.length ? bytes[i + 1] : 0;
      const b3 = i + 2 < bytes.length ? bytes[i + 2] : 0;
      const triple = (b1 << 16) | (b2 << 8) | b3;
      result += chars[(triple >> 18) & 0x3F];
      result += chars[(triple >> 12) & 0x3F];
      result += i + 1 < bytes.length ? chars[(triple >> 6) & 0x3F] : '=';
      result += i + 2 < bytes.length ? chars[triple & 0x3F] : '='; 
    }
    return result;
  }
}

export { Buffer };