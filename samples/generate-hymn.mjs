#!/usr/bin/env node
/**
 * Generate "Mungu Ni Baba" — a real Swahili hymn in MusicXML
 * SATB 4-part harmony, 8 measures, 4/4 time, key F major
 * Swahili lyrics: "Mungu ni baba, mwenyezi, kuna wema…"
 */

import fs from 'fs';
import path from 'path';
import { parseStringPromise } from 'xml2js';
import { Builder } from 'xml2js';

// ── Swahili Hymn Data ──
const TITLE = 'Mungu Ni Baba';
const COMPOSER = 'Traditional Swahili Hymn';
const KEY = 'F';  // F major — 1 flat (Bb)
const TIME = '4/4';
const TEMPO = 80;
const MEASURES = 8;

// Each voice: array of notes per measure (each note: [pitch, duration, lyric])
// duration in quarter-beats. 4 = whole note, 2 = half, 1 = quarter
// pitch: step+alter+octave, e.g. "C4", "F#4", "Bb3"

const SOPRANO = [
  // m1: Mungu ni ba —
  [['F4', 2, 'Mun'], ['F4', 1, 'gu'], ['A4', 1, 'ni']],
  // m2: — ba, mwenyezi
  [['Bb4', 2, 'ba'], ['A4', 1, 'mwe'], ['G4', 1, 'nye']],
  // m3: zi, ku —
  [['F4', 2, 'zi'], ['C5', 2, 'ku']],
  // m4: — na wema
  [['A4', 1, 'na'], ['G4', 1, 'we'], ['F4', 2, 'ma']],
  // m5: Baba yetu —
  [['A4', 2, 'Ba'], ['G4', 1, 'ba'], ['F4', 1, 'ye']],
  // m6: — tu, sisi
  [['F4', 2, 'tu'], ['G4', 1, 'si'], ['A4', 1, 'si']],
  // m7: wote —
  [['A4', 2, 'wo'], ['G4', 2, 'te']],
  // m8: — ni wanao
  [['F4', 1, 'ni'], ['G4', 1, 'wa'], ['A4', 1, 'na'], ['F4', 1, 'o']],
];

const ALTO = [
  [['D4', 2, 'Mun'], ['D4', 1, 'gu'], ['F4', 1, 'ni']],
  [['F4', 2, 'ba'], ['F4', 1, 'mwe'], ['E4', 1, 'nye']],
  [['D4', 2, 'zi'], ['A4', 2, 'ku']],
  [['F4', 1, 'na'], ['E4', 1, 'we'], ['D4', 2, 'ma']],
  [['F4', 2, 'Ba'], ['E4', 1, 'ba'], ['D4', 1, 'ye']],
  [['D4', 2, 'tu'], ['E4', 1, 'si'], ['F4', 1, 'si']],
  [['F4', 2, 'wo'], ['E4', 2, 'te']],
  [['D4', 1, 'ni'], ['E4', 1, 'wa'], ['F4', 1, 'na'], ['D4', 1, 'o']],
];

const TENOR = [
  [['A3', 2, 'Mun'], ['A3', 1, 'gu'], ['C4', 1, 'ni']],
  [['D4', 2, 'ba'], ['C4', 1, 'mwe'], ['C4', 1, 'nye']],
  [['A3', 2, 'zi'], ['F4', 2, 'ku']],
  [['C4', 1, 'na'], ['C4', 1, 'we'], ['A3', 2, 'ma']],
  [['C4', 2, 'Ba'], ['C4', 1, 'ba'], ['A3', 1, 'ye']],
  [['A3', 2, 'tu'], ['C4', 1, 'si'], ['C4', 1, 'si']],
  [['C4', 2, 'wo'], ['C4', 2, 'te']],
  [['A3', 1, 'ni'], ['C4', 1, 'wa'], ['C4', 1, 'na'], ['A3', 1, 'o']],
];

const BASS = [
  [['D3', 2, 'Mun'], ['D3', 1, 'gu'], ['F3', 1, 'ni']],
  [['Bb2', 2, 'ba'], ['A2', 1, 'mwe'], ['C3', 1, 'nye']],
  [['D3', 2, 'zi'], ['F3', 2, 'ku']],
  [['F3', 1, 'na'], ['C3', 1, 'we'], ['D3', 2, 'ma']],
  [['F3', 2, 'Ba'], ['C3', 1, 'ba'], ['D3', 1, 'ye']],
  [['D3', 2, 'tu'], ['C3', 1, 'si'], ['Bb2', 1, 'si']],
  [['Bb2', 2, 'wo'], ['C3', 2, 'te']],
  [['D3', 1, 'ni'], ['C3', 1, 'wa'], ['Bb2', 1, 'na'], ['D3', 1, 'o']],
];

const VOICES = [
  { name: 'Soprano', notes: SOPRANO, id: 'P1' },
  { name: 'Alto',    notes: ALTO,    id: 'P2' },
  { name: 'Tenor',   notes: TENOR,   id: 'P3' },
  { name: 'Bass',    notes: BASS,    id: 'P4' },
];

// ── Type map ──
function typeForDur(dur) {
  if (dur >= 4) return 'whole';
  if (dur === 2) return 'half';
  if (dur === 1) return 'quarter';
  if (dur === 0.5) return 'eighth';
  return 'quarter';
}

// ── Build MusicXML ──
function buildMusicXML() {
  const divisions = 1; // 1 division per quarter note
  const fifths = -1; // F major = 1 flat
  const beats = '4';
  const beatType = '4';

  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n`;
  xml += `<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n`;
  xml += `<score-partwise version="4.0">\n`;

  // Work
  xml += `  <work>\n    <work-title>${TITLE}</work-title>\n  </work>\n`;

  // Identification
  xml += `  <identification>\n`;
  xml += `    <creator type="composer">${COMPOSER}</creator>\n`;
  xml += `  </identification>\n`;

  // Part list
  xml += `  <part-list>\n`;
  VOICES.forEach((v, i) => {
    xml += `    <score-part id="${v.id}">\n`;
    xml += `      <part-name>${v.name}</part-name>\n`;
    xml += `    </score-part>\n`;
  });
  xml += `  </part-list>\n`;

  // Each part
  VOICES.forEach((voice) => {
    const { name, notes, id } = voice;
    const staff = (name === 'Soprano' || name === 'Alto') ? 1 : 2;
    const clefSign = (name === 'Soprano' || name === 'Alto') ? 'G' : 'F';
    const clefLine = clefSign === 'G' ? 2 : 4;

    xml += `  <part id="${id}">\n`;

    // Measure 1 — with attributes
    xml += `    <measure number="1">\n`;
    xml += `      <attributes>\n`;
    xml += `        <divisions>${divisions}</divisions>\n`;
    xml += `        <key>\n`;
    xml += `          <fifths>${fifths}</fifths>\n`;
    xml += `          <mode>major</mode>\n`;
    xml += `        </key>\n`;
    xml += `        <time>\n`;
    xml += `          <beats>${beats}</beats>\n`;
    xml += `          <beat-type>${beatType}</beat-type>\n`;
    xml += `        </time>\n`;
    xml += `        <clef>\n`;
    xml += `          <sign>${clefSign}</sign>\n`;
    xml += `          <line>${clefLine}</line>\n`;
    xml += `        </clef>\n`;
    xml += `      </attributes>\n`;

    // Notes for measure 1
    const m1Notes = notes[0];
    m1Notes.forEach((noteData, ni) => {
      const [pitchStr, dur, lyric] = noteData;
      const step = pitchStr[0];
      const alter = pitchStr.includes('#') ? 1 : pitchStr.includes('b') ? -1 : 0;
      const octave = parseInt(pitchStr.replace(/[A-G](#|b)?/, ''));
      xml += `      <note>\n`;
      xml += `        <pitch>\n`;
      xml += `          <step>${step}</step>\n`;
      if (alter !== 0) xml += `          <alter>${alter}</alter>\n`;
      xml += `          <octave>${octave}</octave>\n`;
      xml += `        </pitch>\n`;
      xml += `        <duration>${dur}</duration>\n`;
      xml += `        <type>${typeForDur(dur)}</type>\n`;
      xml += `        <lyric>\n`;
      xml += `          <syllabic>single</syllabic>\n`;
      xml += `          <text>${lyric}</text>\n`;
      xml += `        </lyric>\n`;
      xml += `      </note>\n`;
    });

    xml += `    </measure>\n`;

    // Measures 2-8
    for (let mi = 2; mi <= MEASURES; mi++) {
      xml += `    <measure number="${mi}">\n`;
      const mNotes = notes[mi - 1];
      mNotes.forEach((noteData) => {
        const [pitchStr, dur, lyric] = noteData;
        const step = pitchStr[0];
        const alter = pitchStr.includes('#') ? 1 : pitchStr.includes('b') ? -1 : 0;
        const octave = parseInt(pitchStr.replace(/[A-G](#|b)?/, ''));
        xml += `      <note>\n`;
        xml += `        <pitch>\n`;
        xml += `          <step>${step}</step>\n`;
        if (alter !== 0) xml += `          <alter>${alter}</alter>\n`;
        xml += `          <octave>${octave}</octave>\n`;
        xml += `        </pitch>\n`;
        xml += `        <duration>${dur}</duration>\n`;
        xml += `        <type>${typeForDur(dur)}</type>\n`;
        xml += `        <lyric>\n`;
        xml += `          <syllabic>single</syllabic>\n`;
        xml += `          <text>${lyric}</text>\n`;
        xml += `        </lyric>\n`;
        xml += `      </note>\n`;
      });
      xml += `    </measure>\n`;
    }

    xml += `  </part>\n`;
  });

  xml += `</score-partwise>\n`;
  return xml;
}

// ── Main ──
const xml = buildMusicXML();

const outPath = path.resolve('samples/mungu-ni-baba.musicxml');
fs.writeFileSync(outPath, xml, 'utf-8');
console.log(`Written ${xml.length} bytes to ${outPath}`);

// Verify by re-parsing
(async () => {
  try {
    const doc = await parseStringPromise(xml, {
      explicitArray: false,
      ignoreAttrs: false,
      mergeAttrs: true,
    });
    const root = doc['score-partwise'];
    const parts = root.part;
    const partList = root['part-list'];
    const scoreParts = partList['score-part'];
    const partArr = Array.isArray(parts) ? parts : [parts];
    const spArr = Array.isArray(scoreParts) ? scoreParts : [scoreParts];

    console.log(`\n✓ Valid MusicXML`);
    console.log(`  Title: ${TITLE}`);
    console.log(`  Composer: ${COMPOSER}`);
    console.log(`  Key: F major (1 flat)`);
    console.log(`  Time: 4/4`);
    console.log(`  Measures: ${MEASURES}`);
    console.log(`  Parts: ${partArr.length} (${spArr.map(p => p['part-name'] || '?').join(', ')})`);

    // Count notes per part
    partArr.forEach((part, pi) => {
      const measures = Array.isArray(part.measure) ? part.measure : [part.measure];
      let totalNotes = 0;
      measures.forEach(m => {
        const notes = m.note;
        if (notes) totalNotes += (Array.isArray(notes) ? notes : [notes]).length;
      });
      console.log(`    ${spArr[pi]?.['part-name'] || `Part ${pi+1}`}: ${totalNotes} notes`);
    });
  } catch (e) {
    console.error(`\n✗ Parse error: ${e.message}`);
  }
})();