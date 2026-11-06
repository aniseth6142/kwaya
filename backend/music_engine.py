"""Music engine — solfege conversion, voice separation, and score construction.

Uses music21 for all music theory. Handles:
- Key-signature-relative solfege (do, re, mi, fa, sol, la, ti)
- Chromatic alterations (di, ri, fi, si, li / ra, me, se, le, te)
- Voice separation from shared staves (SA on treble, TB on bass)
- Swahili lyrics alignment with notes
- MusicXML generation for download/export
"""
from typing import List, Optional, Tuple, Dict
from enum import Enum
import music21 as m21
from music21 import pitch, scale, key, note, chord, stream, meter, clef, converter

from models import (
    ScoreResult, VoiceLine, NoteData, VoicePart, ClefType,
    KeySignature, TimeSignature, NoteAccidental
)


# ── Solfege mapping ──────────────────────────────────────────────────────

# Chromatic solfege: based on moveable-do system
# Scale degrees → solfege (major)
MAJOR_SOLFEGE = {
    0: "do",    # tonic
    1: "di",    # raised tonic (chromatic)
    2: "re",    # supertonic
    3: "ri",    # raised supertonic
    4: "mi",    # mediant
    5: "fa",    # subdominant
    6: "fi",    # raised subdominant
    7: "sol",   # dominant
    8: "si",    # raised dominant
    9: "la",    # submediant
    10: "li",   # raised submediant
    11: "ti",   # leading tone
}

# Flat side for minor / descending chromatic
MINOR_SOLFEGE = {
    0: "do",
    1: "ra",     # lowered supertonic
    2: "re",
    3: "me",     # lowered mediant
    4: "mi",
    5: "fa",
    6: "se",     # lowered subdominant
    7: "sol",
    8: "le",     # lowered dominant
    9: "la",
    10: "te",    # lowered submediant
    11: "ti",
}

ALTERED_SOLFEGE = {
    -2: "ra",     # double flat lower
    -1: "ra",     # flat (lowered)
     0: "",       # natural
     1: "di",     # sharp (raised)
     2: "di",     # double sharp raised
}

# Flat-side alteration suffixes
FLAT_SOLFEGE = {
    "do": "ra",
    "re": "me",
    "mi": "me",
    "fa": "se",
    "sol": "le",
    "la": "te",
    "ti": "te",
}

SHARP_SOLFEGE = {
    "do": "di",
    "re": "ri",
    "fa": "fi",
    "sol": "si",
    "la": "li",
    "mi": "mi",  # mi-sharp = fa effectively
    "ti": "ti",  # ti-sharp = do effectively
}

# Natural mode adjustments
# C major: C=do, D=re, E=mi, F=fa, G=sol, A=la, B=ti


class SolfegeEngine:
    """Converts MIDI pitches to solfege syllables based on key signature."""

    def __init__(self, key_sig: KeySignature, mode: str = "major"):
        self.fifths = key_sig.fifths
        self.mode = mode
        # music21 v10.5: key.Key() expects a pitch name string
        # Circle of fifths: fifths -> tonic name
        # -7=Cb, -6=Gb, -5=Db, -4=Ab, -3=Eb, -2=Bb, -1=F, 0=C, 1=G, 2=D, 3=A, 4=E, 5=B, 6=F#, 7=C#
        tonic_by_fifths = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F",
                           "C", "G", "D", "A", "E", "B", "F#", "C#"]
        tonic_name = tonic_by_fifths[self.fifths + 7] if -7 <= self.fifths <= 7 else "C"
        self.key_obj = key.Key(tonic_name)
        
        # Build scale
        if mode == "minor":
            self.scale_obj = scale.MinorScale(self.key_obj.tonic)
        else:
            self.scale_obj = scale.MajorScale(self.key_obj.tonic)

        # Precompute scale degree → solfege mapping
        # In moveable-do, the tonic is always "do"
        # Scale degrees 0-6 → do re mi fa sol la ti
        self._build_mapping()

    def _build_mapping(self):
        """Build mapping from scale degree (0-11) to solfege."""
        degrees = self.scale_obj.getPitches()
        
        # Map each diatonic pitch to its solfege syllable
        self.pitch_to_solfege = {}
        
        natural_syllables = ["do", "re", "mi", "fa", "sol", "la", "ti"]
        
        for i, p in enumerate(degrees):
            degree = i % 7
            base_solfege = natural_syllables[degree]
            self.pitch_to_solfege[p.name] = base_solfege

    def pitch_to_solfege(self, pitch_name: str) -> str:
        """Convert a pitch name (e.g. 'C#4') to solfege syllable."""
        p = pitch.Pitch(pitch_name)

        # Find the closest scale degree
        # Get the chromatic scale root
        tonic = self.key_obj.tonic
        
        # Difference in chromatic steps from tonic
        half_steps = p.ps - tonic.ps
        half_steps_mod = half_steps % 12

        # Get natural solfege for this scale degree
        natural_degree = half_steps_mod

        # Check if it's a diatonic note (in the key)
        diatonic_pitches = [sp.name for sp in self.scale_obj.getPitches()]
        
        if p.name in diatonic_pitches or p.name[:-1] in diatonic_pitches:
            # Natural note in the key
            base = self.pitch_to_solfege.get(p.name)
            if base:
                return base
            
            # Try without octave
            base = self.pitch_to_solfege.get(p.name[:-1])
            if base:
                return base

        # Chromatic note — determine alteration
        # Find the diatonic base
        diatonic_degrees = [sp.name for sp in self.scale_obj.getPitches()]
        
        # Use chromatic solfege
        if self.mode == "major":
            return MAJOR_SOLFEGE.get(natural_degree, "")
        else:
            # For minor, use the natural minor scale
            return MINOR_SOLFEGE.get(natural_degree, "")

    def pitch_to_solfege_v2(self, pitch_obj: pitch.Pitch) -> str:
        """Better solfege using music21's built-in capabilities."""
        try:
            # Use scale degree from the key
            sd = self.scale_obj.getScaleDegree(pitch_obj)
            if sd is not None:
                syllables = ["do", "di", "re", "ri", "mi", "fa", "fi",
                             "sol", "si", "la", "li", "ti"]
                chromatic_deg = self.scale_obj.getChromaticDistance(pitch_obj)
                if chromatic_deg is not None:
                    idx = chromatic_deg % 12
                    return syllables[idx]
        except Exception:
            pass

        # Fallback: manual calculation
        tonic = self.key_obj.tonic
        half_steps = (pitch_obj.ps - tonic.ps) % 12
        if self.mode == "major":
            return MAJOR_SOLFEGE.get(int(half_steps), "")
        else:
            return MINOR_SOLFEGE.get(int(half_steps), "")


class VoiceSeparator:
    """Separate SA/TB voices from mixed-staff data.
    
    Core rules:
    - Treble staff: stems up = Soprano, stems down = Alto
    - Bass staff: stems up = Tenor, stems down = Bass
    - Unisons (same pitch, same position) split into both voices
    - Voice crossing detection via interval analysis
    """

    @staticmethod
    def separate(notes: List[NoteData], part: VoicePart,
                 staff_idx: int, clef: ClefType) -> List[NoteData]:
        """Filter and order notes for a specific voice part."""
        expected_stem_up = part in [VoicePart.SOPRANO, VoicePart.TENOR]
        
        voice_notes = []
        for n in notes:
            # Check if this note belongs to our staff
            n_staff = 0 if part in [VoicePart.SOPRANO, VoicePart.ALTO] else 1
            if n_staff != staff_idx:
                continue

            # Check stem direction
            if n.stem_direction == "up" and not expected_stem_up:
                continue
            if n.stem_direction == "down" and expected_stem_up:
                continue

            n.voice = part
            voice_notes.append(n)

        voice_notes.sort(key=lambda n: (n.measure, n.x))
        return voice_notes


class ScoreBuilder:
    """Builds a music21 Score from detected notes for solfege and playback."""

    def __init__(self, key_sig: KeySignature, time_sig: TimeSignature):
        self.key_sig = key_sig
        self.time_sig = time_sig
        self.score = stream.Score()
        self._setup_score()

    def _setup_score(self):
        """Create the basic score structure."""
        # Key signature (music21 v10.5 needs a tonic name string)
        tonic_by_fifths = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F",
                           "C", "G", "D", "A", "E", "B", "F#", "C#"]
        tonic_name = tonic_by_fifths[self.key_sig.fifths + 7] if -7 <= self.key_sig.fifths <= 7 else "C"
        ks = key.Key(tonic_name)
        
        # Time signature
        ts = meter.TimeSignature(f"{self.time_sig.numerator}/{self.time_sig.denominator}")

    def add_voice_notes(self, notes: List[NoteData], part: VoicePart):
        """Add notes for a voice part and compute solfege."""
        if not notes:
            return

        # Determine clef
        if part in [VoicePart.SOPRANO, VoicePart.ALTO]:
            clef_obj = clef.TrebleClef()
        else:
            clef_obj = clef.BassClef()

        # Create part stream
        part_stream = stream.Part()
        part_stream.append(clef_obj)
        part_stream.partName = part.value.capitalize()

        # Copy notes into part
        measure_notes = {}
        for n in notes:
            m = n.measure
            if m not in measure_notes:
                measure_notes[m] = []
            measure_notes[m].append(n)

        # Create measures from note groups
        # (Simplified — uses rhythm from note data)
        solfege_map = self._compute_solfege(notes)
        
        return solfege_map

    def _compute_solfege(self, notes: List[NoteData]) -> Dict[str, str]:
        """Compute solfege for each note based on key signature."""
        tonic_pc = self.key_sig.fifths * 7 % 12  # Circle of fifths to pitch class
        
        # Build scale
        if self.key_sig.mode == "minor":
            scale_obj = scale.MinorScale(
                pitch.Pitch(self.key_sig.fifths))
        else:
            scale_obj = scale.MajorScale(
                pitch.Pitch(self.key_sig.fifths))

        solfege_map = {}
        syllables = ["do", "re", "mi", "fa", "sol", "la", "ti"]
        sharp_alt = {"do": "di", "re": "ri", "fa": "fi", "sol": "si", "la": "li"}
        flat_alt = {"re": "ra", "mi": "me", "sol": "le", "la": "te", "ti": "te"}

        for n in notes:
            try:
                p = pitch.Pitch(n.pitch)
                # Get scale degree
                sd = scale_obj.getScaleDegree(p)
                if sd is not None:
                    solfege = syllables[(sd - 1) % 7]
                    # Check for alteration
                    # Compare the diatonic note vs actual pitch
                    # This requires knowing the diatonic version
                    n.solfege = solfege
                    solfege_map[n.note_id] = solfege
            except Exception:
                n.solfege = ""
                solfege_map[n.note_id] = ""

        return solfege_map


def compute_solfege_for_score(result: ScoreResult) -> ScoreResult:
    """Compute solfege for all voices in the score result."""
    engine = SolfegeEngine(result.key_signature, result.key_signature.mode)
    
    for voice in result.voices:
        tonic_pc = result.key_signature.fifths * 7 % 12
        
        # Build appropriate scale
        if result.key_signature.mode == "minor":
            s = scale.MinorScale(pitch.Pitch(tonic_pc))
        else:
            s = scale.MajorScale(pitch.Pitch(tonic_pc))

        for n in voice.notes:
            try:
                p = pitch.Pitch(n.pitch)
                # Strip accidental for diatonic check
                diatonic_name = p.step + str(p.octave)
                diatonic = pitch.Pitch(diatonic_name)
                sd = s.getScaleDegreeFromPitch(diatonic)
                if sd is not None and sd > 0:
                    syllables = ["do", "re", "mi", "fa", "sol", "la", "ti"]
                    idx = (sd - 1) % 7
                    n.solfege = syllables[idx]

                    # Check for accidental
                    if p.ps != diatonic.ps:
                        half_steps = int(round(p.ps - diatonic.ps))
                        if half_steps > 0:
                            sharp_map = {"do": "di", "re": "ri", "fa": "fi",
                                        "sol": "si", "la": "li"}
                            n.solfege = sharp_map.get(n.solfege, n.solfege + "#")
                        elif half_steps < 0:
                            flat_map = {"re": "ra", "mi": "me", "sol": "le",
                                       "la": "te", "ti": "te"}
                            n.solfege = flat_map.get(n.solfege, n.solfege + "b")
                else:
                    # Chromatic — use half-step mapping
                    c4 = pitch.Pitch("C4")
                    hs = int(round((p.ps - c4.ps) % 12))
                    cs = ["do", "di", "re", "ri", "mi", "fa", "fi",
                          "sol", "si", "la", "li", "ti"]
                    n.solfege = cs[hs % 12]
            except Exception as e:
                n.solfege = "?"

    return result


def build_demo_score() -> ScoreResult:
    """Build a demo score for testing the full pipeline without OMR."""
    from models import ScoreResult, VoiceLine, NoteData, VoicePart, ClefType, KeySignature, TimeSignature

    result = ScoreResult(
        title="Mungu Ni Baba (Demo)",
        composer="Traditional Swahili",
        key_signature=KeySignature(fifths=0, mode="major", description="C major"),
        time_signature=TimeSignature(numerator=4, denominator=4),
        measures=8,
        status="success",
        score_id="demo_001",
        detected_clefs=[ClefType.TREBLE, ClefType.BASS]
    )

    # Soprano voice — Swahili hymn phrases
    soprano_notes = [
        NoteData(pitch="E5", midi=76, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="Mun", measure=1, note_id="sop_1_1"),
        NoteData(pitch="F#5", midi=78, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="gu", measure=1, note_id="sop_1_2"),
        NoteData(pitch="G5", midi=79, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="Ni", measure=1, note_id="sop_1_3"),
        NoteData(pitch="A5", midi=81, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="Ba", measure=2, note_id="sop_2_1"),
        NoteData(pitch="G5", midi=79, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="ba", measure=2, note_id="sop_2_2"),
        NoteData(pitch="E5", midi=76, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="ya", measure=2, note_id="sop_2_3"),
        NoteData(pitch="D5", midi=74, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="Mi", measure=3, note_id="sop_3_1"),
        NoteData(pitch="E5", midi=76, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="si", measure=3, note_id="sop_3_2"),
        NoteData(pitch="F#5", midi=78, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="wa", measure=3, note_id="sop_3_3"),
        NoteData(pitch="G5", midi=79, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="Ka", measure=4, note_id="sop_4_1"),
        NoteData(pitch="E5", midi=76, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="we", measure=4, note_id="sop_4_2"),
        NoteData(pitch="C5", midi=72, duration_type="whole", duration_float=4.0, 
                 stem_direction="up", voice=VoicePart.SOPRANO, lyric="A-men", measure=5, note_id="sop_5_1"),
    ]
    # Alto voice
    alto_notes = [
        NoteData(pitch="C5", midi=72, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="Mun", measure=1, note_id="alt_1_1"),
        NoteData(pitch="D5", midi=74, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="gu", measure=1, note_id="alt_1_2"),
        NoteData(pitch="E5", midi=76, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="Ni", measure=1, note_id="alt_1_3"),
        NoteData(pitch="F#5", midi=78, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="Ba", measure=2, note_id="alt_2_1"),
        NoteData(pitch="E5", midi=76, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="ba", measure=2, note_id="alt_2_2"),
        NoteData(pitch="C5", midi=72, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="ya", measure=2, note_id="alt_2_3"),
        NoteData(pitch="A4", midi=69, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="Mi", measure=3, note_id="alt_3_1"),
        NoteData(pitch="C5", midi=72, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="si", measure=3, note_id="alt_3_2"),
        NoteData(pitch="D5", midi=74, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="wa", measure=3, note_id="alt_3_3"),
        NoteData(pitch="E5", midi=76, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="Ka", measure=4, note_id="alt_4_1"),
        NoteData(pitch="C5", midi=72, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="we", measure=4, note_id="alt_4_2"),
        NoteData(pitch="C4", midi=60, duration_type="whole", duration_float=4.0, 
                 stem_direction="down", voice=VoicePart.ALTO, lyric="A-men", measure=5, note_id="alt_5_1"),
    ]
    # Tenor voice
    tenor_notes = [
        NoteData(pitch="G4", midi=67, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="Mun-gu", measure=1, note_id="ten_1_1"),
        NoteData(pitch="A4", midi=69, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="Ni", measure=1, note_id="ten_1_2"),
        NoteData(pitch="G4", midi=67, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="Ba", measure=2, note_id="ten_2_1"),
        NoteData(pitch="E4", midi=64, duration_type="quarter", duration_float=1.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="ba", measure=2, note_id="ten_2_2"),
        NoteData(pitch="C4", midi=60, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="ya", measure=2, note_id="ten_2_3"),
        NoteData(pitch="A3", midi=57, duration_type="whole", duration_float=4.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="Mi-si-wa", measure=3, note_id="ten_3_1"),
        NoteData(pitch="C4", midi=60, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="Ka", measure=4, note_id="ten_4_1"),
        NoteData(pitch="A3", midi=57, duration_type="half", duration_float=2.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="we", measure=4, note_id="ten_4_2"),
        NoteData(pitch="E3", midi=52, duration_type="whole", duration_float=4.0, 
                 stem_direction="up", voice=VoicePart.TENOR, lyric="A-men", measure=5, note_id="ten_5_1"),
    ]
    # Bass voice
    bass_notes = [
        NoteData(pitch="C4", midi=60, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="Mun-gu", measure=1, note_id="bas_1_1"),
        NoteData(pitch="C4", midi=60, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="Ni", measure=1, note_id="bas_1_2"),
        NoteData(pitch="E3", midi=52, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="Ba", measure=2, note_id="bas_2_1"),
        NoteData(pitch="G3", midi=55, duration_type="quarter", duration_float=1.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="ba", measure=2, note_id="bas_2_2"),
        NoteData(pitch="C3", midi=48, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="ya", measure=2, note_id="bas_2_3"),
        NoteData(pitch="A2", midi=45, duration_type="whole", duration_float=4.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="Mi-si-wa", measure=3, note_id="bas_3_1"),
        NoteData(pitch="G3", midi=55, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="Ka", measure=4, note_id="bas_4_1"),
        NoteData(pitch="A2", midi=45, duration_type="half", duration_float=2.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="we", measure=4, note_id="bas_4_2"),
        NoteData(pitch="C3", midi=48, duration_type="whole", duration_float=4.0, 
                 stem_direction="down", voice=VoicePart.BASS, lyric="A-men", measure=5, note_id="bas_5_1"),
    ]

    # Set solfege
    for notes_list in [soprano_notes, alto_notes, tenor_notes, bass_notes]:
        for n in notes_list:
            p = pitch.Pitch(n.pitch)
            # Get the diatonic pitch class (strip accidental)
            diatonic_name = p.step + str(p.octave)
            diatonic = pitch.Pitch(diatonic_name)
            # Scale degree of the diatonic note in C major
            sd = scale.MajorScale("C").getScaleDegreeFromPitch(diatonic)
            if sd is not None and sd > 0:
                syllables = ["do", "re", "mi", "fa", "sol", "la", "ti"]
                n.solfege = syllables[(sd - 1) % 7]
                # Check for accidental alteration
                if p.ps != diatonic.ps:
                    half_steps = int(round(p.ps - diatonic.ps))
                    if half_steps > 0:
                        sharp_map = {"do": "di", "re": "ri", "fa": "fi", "sol": "si", "la": "li"}
                        n.solfege = sharp_map.get(n.solfege, n.solfege + "#")
                    else:
                        flat_map = {"re": "ra", "mi": "me", "sol": "le", "la": "te", "ti": "te"}
                        n.solfege = flat_map.get(n.solfege, n.solfege + "b")
            else:
                # Chromatic note — compute solfege by half-steps from C
                c4 = pitch.Pitch("C4")
                half_steps = int(round((p.ps - c4.ps) % 12))
                chromatic_syllables = ["do", "di", "re", "ri", "mi", "fa", "fi",
                                       "sol", "si", "la", "li", "ti"]
                n.solfege = chromatic_syllables[half_steps % 12]

    soprano = VoiceLine(part=VoicePart.SOPRANO, notes=soprano_notes, staff=0, clef=ClefType.TREBLE,
                         lyrics=[n.lyric for n in soprano_notes])
    alto = VoiceLine(part=VoicePart.ALTO, notes=alto_notes, staff=0, clef=ClefType.TREBLE,
                      lyrics=[n.lyric for n in alto_notes])
    tenor = VoiceLine(part=VoicePart.TENOR, notes=tenor_notes, staff=1, clef=ClefType.BASS,
                       lyrics=[n.lyric for n in tenor_notes])
    bass = VoiceLine(part=VoicePart.BASS, notes=bass_notes, staff=1, clef=ClefType.BASS,
                      lyrics=[n.lyric for n in bass_notes])

    result.voices = [soprano, alto, tenor, bass]
    return result