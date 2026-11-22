/**
 * Lightweight MusicXML parser — parses SATB scores into a usable data structure.
 * Uses xml2js for reliable XML parsing.
 */

// @ts-ignore — xml2js has no type declarations
import { parseStringPromise } from 'xml2js';

export interface Note {
  pitch: string;
  step: string;
  octave: number;
  alter: number;
  type: string;
  duration: number;
  rest: boolean;
  lyric?: string;
}

export interface VoicePart {
  name: string;
  id: string;
  notes: Note[];
}

export interface ParsedScore {
  title: string;
  composer: string;
  key: string;
  keyAccidentals: number;
  timeSig: string;
  divisions: number;
  parts: VoicePart[];
}

function noteName(step: string, alter: number, octave: number): string {
  const acc = alter === 1 ? '#' : alter === -1 ? 'b' : '';
  return `${step}${acc}${octave}`;
}

function keyToString(acc: number): string {
  const majorKeys = [
    'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
  ];
  const idx = Math.max(0, Math.min(14, acc + 7));
  return majorKeys[idx];
}

function val(o: any): any {
  if (typeof o === 'string') return o;
  if (Array.isArray(o) && o.length > 0) return val(o[0]);
  if (o && typeof o === 'object') {
    const keys = Object.keys(o);
    if (keys.length === 1 && (keys[0] === '$' || keys[0] === '_')) return o['_'];
    // Try to get text content
    if (o['_']) return o['_'];
    for (const k of keys) {
      if (k !== '$') return val(o[k]);
    }
  }
  return '';
}

function num(o: any, fallback: number = 0): number {
  const v = val(o);
  const n = parseInt(v, 10);
  return isNaN(n) ? fallback : n;
}

export async function parseMusicXML(xml: string): Promise<ParsedScore> {
  const doc = await parseStringPromise(xml, {
    explicitArray: false,
    ignoreAttrs: false,
    mergeAttrs: true,
    explicitChildren: false,
    preserveChildren: false,
  });

  const root = doc['score-partwise'] || doc;
  
  // --- Title & Composer ---
  const movTitle = val(root.movement?.['movement-title']) || val(root.movement?.title) || '';
  const workTitle = val(root.work?.['work-title']) || val(root.work?.title) || '';
  const movementTitle = movTitle || workTitle;
  const ident = root.identification;
  let composer = '';
  if (ident?.creator) {
    const c = ident.creator;
    composer = typeof c === 'string' ? c : val(c);
  }

  // --- Part list definitions ---
  const partList = root['part-list'] || {};
  const scoreParts = partList['score-part'] || [];
  const partDefs: any[] = Array.isArray(scoreParts) ? scoreParts : [scoreParts];

  // --- Actual part data ---
  const partsNode = root.part || [];
  const partsArr: any[] = Array.isArray(partsNode) ? partsNode : [partsNode];

  let keyAcc = 0;
  let timeSig = '4/4';
  let divisions = 1;
  const voiceParts: VoicePart[] = [];

  for (let pi = 0; pi < partsArr.length; pi++) {
    const part = partsArr[pi];
    if (!part) continue;
    
    const partId = part['$']?.id || `P${pi}`;
    
    const def = partDefs.find((d: any) => {
      const id = d?.$?.id || '';
      return id === partId;
    });
    const partName = val(def?.['part-name']) || `Voice ${pi + 1}`;
    const voiceName = mapVoiceName(partName, pi);

    const measures = part.measure || [];
    const measuresArr: any[] = Array.isArray(measures) ? measures : [measures];
    const notes: Note[] = [];

    for (const measure of measuresArr) {
      if (!measure || typeof measure !== 'object') continue;

      const attrs = measure.attributes;
      if (attrs) {
        // Divisions
        if (attrs.divisions) divisions = num(attrs.divisions, 1);
        
        // Key signature
        const keyNode = attrs.key;
        if (keyNode) {
          const fifths = num(keyNode.fifths, 0);
          keyAcc = fifths;
        }

        // Time signature
        const timeNode = attrs.time;
        if (timeNode) {
          let beats = val(timeNode.beats);
          let beatType = val(timeNode['beat-type']);
          if (beats && beatType) timeSig = `${beats}/${beatType}`;
        }
      }

      // Parse notes
      const noteNodes = measure.note;
      if (!noteNodes) continue;
      const noteArr: any[] = Array.isArray(noteNodes) ? noteNodes : [noteNodes];

      for (const ne of noteArr) {
        if (!ne || typeof ne !== 'object') continue;

        const rest = ne.rest !== undefined;
        const pitchNode = ne.pitch;
        const type = val(ne.type) || '';
        const dur = num(ne.duration, 0);
        
        let step = 'C', octave = 4, alter = 0;
        let notePitch = '';

        if (pitchNode && !rest) {
          step = val(pitchNode.step) || 'C';
          octave = num(pitchNode.octave, 4);
          alter = num(pitchNode.alter, 0);
          notePitch = noteName(step, alter, octave);
        }

        // Lyric / syllable
        let lyric: string | undefined;
        const lyricNode = ne.lyric;
        if (lyricNode) {
          const txt = val(lyricNode.text) || val(lyricNode.syllabic) || '';
          if (txt && txt !== '-' && txt !== '_') {
            // lyricNode may have text sub-element
            if (typeof lyricNode.text === 'object') {
              const t = val(lyricNode.text);
              if (t) lyric = t;
            } else if (typeof lyricNode.text === 'string') {
              lyric = lyricNode.text;
            }
          }
        }

        notes.push({
          pitch: notePitch,
          step,
          octave,
          alter,
          type,
          duration: dur,
          rest,
          lyric,
        });
      }
    }

    voiceParts.push({ name: voiceName, id: partId, notes });
  }

  const key = keyToString(keyAcc);

  return {
    title: movementTitle || 'Untitled',
    composer: composer || 'Unknown',
    key,
    keyAccidentals: keyAcc,
    timeSig,
    divisions,
    parts: voiceParts,
  };
}

function mapVoiceName(rawName: string, index: number): string {
  const lower = rawName.toLowerCase().trim();
  if (lower.includes('soprano')) return 'Soprano';
  if (lower.includes('alto')) return 'Alto';
  if (lower.includes('tenor')) return 'Tenor';
  if (lower.includes('bass') || lower.includes('bajo')) return 'Bass';
  const names = ['Soprano', 'Alto', 'Tenor', 'Bass'];
  return index < names.length ? names[index] : rawName;
}

export function scoreToText(score: ParsedScore): string {
  const lines: string[] = [];
  lines.push(`Title: ${score.title}`);
  lines.push(`Composer: ${score.composer}`);
  lines.push(`Key: ${score.key} (${score.keyAccidentals > 0 ? score.keyAccidentals + ' sharps' : score.keyAccidentals < 0 ? Math.abs(score.keyAccidentals) + ' flats' : 'no accidentals'})`);
  lines.push(`Time Signature: ${score.timeSig}`);
  lines.push('');

  for (const part of score.parts) {
    lines.push(`--- ${part.name} (${part.id}) ---`);
    lines.push(`Total notes: ${part.notes.length}`);

    const sample = part.notes.slice(0, 20);
    const noteStrs = sample.map(n => {
      if (n.rest) return `[rest:${n.type}]`;
      const s = n.pitch;
      return n.lyric ? `${s}(${n.lyric})` : s;
    });
    lines.push(`Notes: ${noteStrs.join(' ')}${part.notes.length > 20 ? ' …' : ''}`);

    const pitches = part.notes.filter(n => !n.rest).map(n => n.pitch);
    if (pitches.length > 0) {
      const sorted = [...pitches].sort((a, b) => {
        const p = (s: string) => {
          const m = s.match(/([A-G])([#b]?)(\d)/);
          if (!m) return 0;
          return parseInt(m[3]) * 12 + 'C D EF G A B'.indexOf(m[1]);
        };
        return p(a) - p(b);
      });
      lines.push(`Range: ${sorted[0]} → ${sorted[sorted.length - 1]}`);
    }
    lines.push('');
  }

  return lines.join('\n');
}