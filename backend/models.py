"""Data models for Kwaya — choral score assistant."""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ClefType(str, Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    SOPRANO = "soprano"
    TENOR = "tenor"


class VoicePart(str, Enum):
    SOPRANO = "soprano"
    ALTO = "alto"
    TENOR = "tenor"
    BASS = "bass"


class NoteAccidental(str, Enum):
    FLAT = "flat"
    SHARP = "sharp"
    NATURAL = "natural"
    DOUBLE_FLAT = "double-flat"
    DOUBLE_SHARP = "double-sharp"
    NONE = "none"


class NoteData(BaseModel):
    """A single note with position and duration info."""
    pitch: str = ""           # e.g. "C4", "F#5"
    midi: Optional[int] = None
    step: str = ""            # A-G
    octave: int = 4
    alteration: int = 0       # -1=flat, 0=natural, 1=sharp, 2=double-sharp, -2=double-flat
    duration_type: str = "quarter"  # whole, half, quarter, eighth, 16th, 32nd
    duration_float: float = 1.0    # in quarter beats
    stem_direction: str = ""  # "up" or "down"
    voice: VoicePart = VoicePart.SOPRANO
    solfege: str = ""         # do, re, mi, fa, sol, la, ti (+ chromatic variants)
    lyric: str = ""           # The Swahili syllable for this note
    confidence: float = 1.0
    x: float = 0              # x position in the image (for UI correction)
    y: float = 0              # y position in the image
    measure: int = 1
    note_id: str = ""


class VoiceLine(BaseModel):
    """One complete voice part across the piece."""
    part: VoicePart
    notes: List[NoteData] = []
    staff: int = 0            # 0=treble, 1=bass
    clef: ClefType = ClefType.TREBLE
    lyrics: List[str] = []    # All lyrics syllables in order


class TimeSignature(BaseModel):
    numerator: int = 4
    denominator: int = 4


class KeySignature(BaseModel):
    fifths: int = 0           # negative=flats, positive=sharps
    mode: str = "major"       # or "minor"
    description: str = "C major"


class BarData(BaseModel):
    measure: int
    voices: List[str] = []    # voice part names in this bar


class ScoreResult(BaseModel):
    """The full result of reading and interpreting a score."""
    title: str = "Untitled"
    composer: str = ""
    pages: int = 1
    time_signature: TimeSignature = TimeSignature()
    key_signature: KeySignature = KeySignature()
    voices: List[VoiceLine] = []
    measures: int = 0
    raw_musicxml: str = ""    # The MusicXML for download
    detected_clefs: List[ClefType] = [ClefType.TREBLE, ClefType.BASS]
    status: str = "success"
    message: str = ""
    score_id: str = ""

    def get_voice(self, part: VoicePart) -> Optional[VoiceLine]:
        for v in self.voices:
            if v.part == part:
                return v
        return None

    def to_solfege_text(self, part: VoicePart) -> str:
        """Format solfege + lyrics as readable lines."""
        voice = self.get_voice(part)
        if not voice:
            return f"[{part.value.upper()}] — no notes"

        lines = []
        lines.append(f"── {part.value.upper()} ──")
        lines.append(f"Clef: {voice.clef.value}")
        lines.append(f"Key: {self.key_signature.description}")
        lines.append("")

        # Group by measure
        current_measure = 0
        measure_notes = []
        for n in voice.notes:
            if n.measure != current_measure:
                if measure_notes:
                    lines.append(f"  m.{current_measure}:")
                    for n2 in measure_notes:
                        lyric_part = f" [{n2.lyric}]" if n2.lyric else ""
                        lines.append(f"    {n2.solfege:>8}{lyric_part}  ({n2.pitch}, {n2.duration_type})")
                    lines.append("")
                current_measure = n.measure
                measure_notes = []
            measure_notes.append(n)
        # Last measure
        if measure_notes:
            lines.append(f"  m.{current_measure}:")
            for n2 in measure_notes:
                lyric_part = f" [{n2.lyric}]" if n2.lyric else ""
                lines.append(f"    {n2.solfege:>8}{lyric_part}  ({n2.pitch}, {n2.duration_type})")

        return "\n".join(lines)


class CorrectionRequest(BaseModel):
    score_id: str
    part: VoicePart
    measure: int
    note_index: int
    new_pitch: str = ""
    new_solfege: str = ""
    new_duration: str = ""
    new_lyric: str = ""


class SynthesisRequest(BaseModel):
    score_id: str
    parts: List[VoicePart] = []
    tempo: int = 80
    solo: bool = False       # Play solo or tutti


class UploadResponse(BaseModel):
    score_id: str
    filename: str
    pages: int
    message: str = "Uploaded successfully"


class ProgressUpdate(BaseModel):
    stage: str               # upload, reading, separating, solfege, synthesizing, done
    progress: float          # 0.0 to 1.0
    message: str = ""