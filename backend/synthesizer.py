"""Synthesis engine — sings solfege syllables using FluidSynth choir SoundFont.

Takes note data with solfege syllables and generates WAV audio.
Creates separate audio per voice part, plus a mixed tutti.
Handles syllable timing, pitch bending (portamento), and vowel shaping.
"""
import os
import math
import json
import struct
import wave
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

try:
    import fluidsynth
    HAS_FLUIDSYNTH = True
except ImportError:
    HAS_FLUIDSYNTH = False
    print("Warning: fluidsynth not installed. Using pure-numpy synthesis fallback.")

from models import (
    ScoreResult, VoiceLine, NoteData, VoicePart, SynthesisRequest
)


# ── Solfege frequency mapping ────────────────────────────────────────────
# Formant frequencies for sung vowels (F1, F2 in Hz)
# Based on standard sung vowel formants (Italianate choral style)
VOWEL_FORMANTS = {
    # vowel: (F1, F2, F3)
    'i': (270, 2290, 3010),   # ee (do, di, ri, mi, si, li, ti)
    'e': (400, 1920, 2560),   # eh (re, me, se, te)
    'a': (720, 1240, 2500),   # ah (fa, la, ra, ma, etc.)
    'o': (510, 1000, 2500),   # oh (do, sol)
    'u': (300, 870, 2240),    # oo
    'schwa': (500, 1500, 2500),
}

# Map solfege syllables to primary vowel
SOLFEGE_VOWEL = {
    'do': 'a',       # 'do' → /d/+ /o/ (close o)
    'di': 'i',       # 'di' → /d/+ /i/
    're': 'e',       # 're' → /r/+ /e/
    'ri': 'i',       # 'ri' → /r/+ /i/
    'mi': 'i',       # 'mi' → /m/+ /i/
    'fa': 'a',       # 'fa' → /f/+ /a/
    'fi': 'i',       # 'fi' → /f/+ /i/
    'sol': 'o',      # 'sol' → /s/+ /o/+ /l/
    'si': 'i',       # 'si' → /s/+ /i/
    'la': 'a',       # 'la' → /l/+ /a/
    'li': 'i',       # 'li' → /l/+ /i/
    'ti': 'i',       # 'ti' → /t/+ /i/
    'ra': 'a',       # 'ra' → /r/+ /a/
    'me': 'e',       # 'me' → /m/+ /e/
    'se': 'e',       # 'se' → /s/+ /e/
    'le': 'e',       # 'le' → /l/+ /e/
    'te': 'e',       # 'te' → /t/+ /e/
}

# Consonant onset durations (seconds)
CONSONANT_DUR = {
    'd': 0.05, 'r': 0.04, 'm': 0.06, 'f': 0.05,
    's': 0.07, 'l': 0.05, 't': 0.05, 'n': 0.05,
    'b': 0.06, 'p': 0.05, 'k': 0.06, 'g': 0.05,
    'v': 0.04, 'z': 0.06, '': 0.0,
}


class ChoirSynthesizer:
    """Synthesizes choir audio from solfege + pitch data."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.sf2_path = self._find_soundfont()

    def _find_soundfont(self) -> Optional[str]:
        """Find a suitable SoundFont file."""
        # Check common locations
        search_paths = [
            Path(os.getcwd()) / "soundfonts" / "choir.sf2",
            Path(os.getcwd()) / "soundfonts" / "choir.sf3",
            Path.home() / "soundfonts" / "choir.sf2",
            Path.home() / "AppData" / "Local" / "soundfonts" / "choir.sf2",
            # Check for FluidR3 or similar
            Path(os.getcwd()) / "soundfonts" / "FluidR3_GM.sf2",
            Path(os.getcwd()) / "soundfonts" / "FluidR3.sf2",
        ]

        for p in search_paths:
            if p.exists():
                return str(p)

        # Try to find any .sf2 file
        import glob
        sf2_files = glob.glob(str(Path(os.getcwd()) / "soundfonts" / "*.sf2"))
        if sf2_files:
            return sf2_files[0]

        return None

    def synthesize(self, voice: VoiceLine, tempo: int = 80,
                   key_fifths: int = 0) -> np.ndarray:
        """Synthesize a voice part to audio array."""
        if HAS_FLUIDSYNTH and self.sf2_path:
            return self._synthesize_fluidsynth(voice, tempo)
        else:
            return self._synthesize_pure_numpy(voice, tempo)

    def _synthesize_fluidsynth(self, voice: VoiceLine, tempo: int = 80) -> np.ndarray:
        """Use FluidSynth for synthesis."""
        try:
            fl = fluidsynth.Synth()
            fl.start()

            # Load SoundFont
            sfid = fl.sfload(str(self.sf2_path))
            if sfid == -1:
                print(f"Failed to load SoundFont: {self.sf2_path}")
                return self._synthesize_pure_numpy(voice, tempo)

            # Select choir preset (usually 0 or 52 in GM)
            fl.program_select(0, sfid, 0, 0)  # Channel 0, bank 0, preset 0

            # No built-in note_on/off in basic fluidsynth...
            # Use midi_file approach instead
            return self._synthesize_midi_fluidsynth(voice, tempo)

        except Exception as e:
            print(f"FluidSynth error: {e}")
            return self._synthesize_pure_numpy(voice, tempo)

    def _synthesize_midi_fluidsynth(self, voice: VoiceLine, tempo: int = 80) -> np.ndarray:
        """Generate MIDI file and render with FluidSynth."""
        import tempfile
        import subprocess

        if not voice.notes:
            return np.zeros(self.sample_rate)

        # Calculate beats per second
        bps = tempo / 60.0

        # Total duration
        total_beats = sum(n.duration_float for n in voice.notes)
        total_secs = total_beats / bps + 2.0  # +2 secs padding

        # Create a simple MIDI file
        midi_data = self._build_midi(voice, tempo)

        if midi_data is None:
            return self._synthesize_pure_numpy(voice, tempo)

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            f.write(midi_data)
            midi_path = f.name

        output_wav = midi_path + '.wav'

        try:
            # Render with fluidsynth command line
            if self.sf2_path:
                cmd = ['fluidsynth', '-ni', self.sf2_path, midi_path,
                       '-F', output_wav, '-r', str(self.sample_rate)]
                subprocess.run(cmd, capture_output=True, timeout=30)

                if Path(output_wav).exists():
                    import wave
                    with wave.open(output_wav, 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    Path(output_wav).unlink(missing_ok=True)
                    Path(midi_path).unlink(missing_ok=True)
                    return audio

            Path(midi_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"FluidSynth render error: {e}")
            Path(midi_path).unlink(missing_ok=True)

        return self._synthesize_pure_numpy(voice, tempo)

    def _build_midi(self, voice: VoiceLine, tempo: int = 80) -> Optional[bytes]:
        """Build a standard MIDI file from voice data."""
        try:
            from midiutil import MIDIFile
            track = 0
            midi = MIDIFile(1)
            midi.addTempo(track, 0, tempo)

            current_time = 0.0
            for n in voice.notes:
                if n.midi is None or n.midi <= 0:
                    # Try to compute from pitch
                    from music21 import pitch
                    try:
                        p = pitch.Pitch(n.pitch)
                        n.midi = p.midi
                    except:
                        continue

                dur = n.duration_float
                midi.addNote(track, 0, n.midi, current_time, dur, 100)
                current_time += dur

            import io
            buf = io.BytesIO()
            midi.writeFile(buf)
            return buf.getvalue()
        except ImportError:
            return None
        except Exception as e:
            print(f"MIDI build error: {e}")
            return None

    def _synthesize_pure_numpy(self, voice: VoiceLine, tempo: int = 80) -> np.ndarray:
        """Pure numpy synthesis with formant filtering.

        Generates pitched vocal tones using additive synthesis with
        formant shaping for each solfege vowel.
        """
        sr = self.sample_rate
        bps = tempo / 60.0

        if not voice.notes:
            return np.zeros(int(sr * 0.5))

        # Calculate total duration
        total_beats = sum(n.duration_float for n in voice.notes)
        total_samples = int(total_beats / bps * sr) + sr  # 1s padding

        audio = np.zeros(total_samples)

        current_beat = 0.0
        for n in voice.notes:
            start_sample = int(current_beat / bps * sr)
            dur_beats = n.duration_float
            dur_samples = int(dur_beats / bps * sr)

            if start_sample + dur_samples > len(audio):
                dur_samples = len(audio) - start_sample
            if dur_samples <= 0:
                current_beat += dur_beats
                continue

            # Generate note
            note_audio = self._generate_sung_note(
                midi=n.midi,
                duration=dur_samples,
                solfege=n.solfege,
                pitch_name=n.pitch
            )

            # Mix into audio
            end = min(start_sample + len(note_audio), len(audio))
            audio[start_sample:end] += note_audio[:end - start_sample]

            current_beat += dur_beats

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        return audio.astype(np.float32)

    def _generate_sung_note(self, midi: int, duration: int,
                            solfege: str, pitch_name: str = "") -> np.ndarray:
        """Generate a single sung note with vowel formants."""
        sr = self.sample_rate

        if midi is None or midi <= 0:
            try:
                from music21 import pitch
                p = pitch.Pitch(pitch_name)
                midi = p.midi
            except:
                return np.zeros(duration)

        # Fundamental frequency
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))

        t = np.arange(duration) / sr

        # Get vowel formants
        vowel = SOLFEGE_VOWEL.get(solfege[-2:] if len(solfege) > 1 else solfege, 'a')
        vowel = SOLFEGE_VOWEL.get(solfege, vowel)
        formants = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS['a'])
        f1, f2, f3 = formants

        # Harmonic synthesis with formant filtering
        n_harmonics = min(20, int(sr / (2 * freq)) - 1)
        if n_harmonics < 1:
            n_harmonics = 1

        signal = np.zeros(duration)
        for h in range(1, n_harmonics + 1):
            hfreq = freq * h
            amplitude = 1.0 / h  # Basic spectral slope

            # Formant shaping (simplified — use gain at each formant center)
            # F1 gain
            gain_f1 = 1.0 / (1.0 + ((hfreq - f1) / (f1 * 0.3)) ** 4)
            # F2 gain
            gain_f2 = 0.7 / (1.0 + ((hfreq - f2) / (f2 * 0.2)) ** 4)
            # F3 gain
            gain_f3 = 0.3 / (1.0 + ((hfreq - f3) / (f3 * 0.15)) ** 4)

            gain = amplitude * (0.5 + gain_f1 + gain_f2 + gain_f3)
            signal += gain * np.sin(2 * np.pi * hfreq * t)

        # Apply envelope
        attack = int(0.03 * sr)
        release = int(0.05 * sr)

        if attack + release < duration:
            envelope = np.ones(duration)
            # Attack
            envelope[:attack] = np.linspace(0, 1, attack) ** 0.5
            # Release
            envelope[-release:] = np.linspace(1, 0, release) ** 0.5
            # Slight vibrato
            vibrato = 1.0 + 0.005 * np.sin(2 * np.pi * 5.5 * t)
            signal = signal * envelope * vibrato
        else:
            signal = signal * np.linspace(1, 0, duration) ** 0.5

        # Normalize
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.5

        return signal

    def synthesize_tutti(self, result: ScoreResult, tempo: int = 80) -> Dict[str, np.ndarray]:
        """Synthesize all voice parts."""
        audio_parts = {}
        for voice in result.voices:
            if voice.notes:
                audio = self.synthesize(voice, tempo, result.key_signature.fifths)
                audio_parts[voice.part.value] = audio

        return audio_parts

    @staticmethod
    def save_wav(audio: np.ndarray, path: str, sample_rate: int = 22050):
        """Save audio as WAV file."""
        # Convert to int16
        audio_int16 = (audio * 32767.0).astype(np.int16)

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

    @staticmethod
    def encode_ogg(audio: np.ndarray, path: str, sample_rate: int = 22050):
        """Save as OGG using ffmpeg if available."""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            ChoirSynthesizer.save_wav(audio, wav_path, sample_rate)

        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', wav_path,
                '-c:a', 'libvorbis', '-q:a', '3',
                path
            ], capture_output=True, timeout=30)
        except Exception:
            # Fallback: just rename
            import shutil
            shutil.copy(wav_path, path)
        finally:
            Path(wav_path).unlink(missing_ok=True)


def synthesize_voice(result: ScoreResult, part: VoicePart,
                     tempo: int = 80) -> Optional[bytes]:
    """Synthesize a single voice part to WAV bytes."""
    voice = result.get_voice(part)
    if not voice or not voice.notes:
        return None

    synth = ChoirSynthesizer()
    audio = synth.synthesize(voice, tempo, result.key_signature.fifths)

    # Convert to WAV bytes
    import io
    buf = io.BytesIO()
    audio_int16 = (audio * 32767.0).astype(np.int16)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()