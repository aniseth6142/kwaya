"""Lightweight OMR — detects staff positions, noteheads, and stems.

Designed specifically for SATB choral scores with two voices per staff.
Uses OpenCV heuristics — no heavy ML models.
The correction UI in the frontend is the safety net for imperfect detections.
"""
import cv2
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

from models import (
    ScoreResult, VoiceLine, NoteData, VoicePart, ClefType,
    KeySignature, TimeSignature
)


class StaffDetector:
    """Detects staff lines and groups them into staves."""

    def detect(self, gray: np.ndarray) -> List[dict]:
        """
        Returns list of detected staves, each with:
        {lines: [y0..y4], clef: str, staff_idx: int}
        """
        h, w = gray.shape
        binary = self._binarize(gray)

        # Horizontal projection to find rows with lots of ink
        proj = np.sum(binary, axis=1) / 255
        ink_thresh = w * 0.25

        # Find staff regions (continuous runs of high ink density)
        regions = []
        in_region = False
        start = 0
        for row in range(h):
            if proj[row] > ink_thresh:
                if not in_region:
                    start = row
                    in_region = True
            else:
                if in_region:
                    if row - start > h // 20:
                        regions.append((start, row))
                    in_region = False

        staves = []
        for idx, (r0, r1) in enumerate(regions):
            # Within this region, find the 5 strongest line peaks
            strip = proj[r0:r1]
            # Smooth with small window
            kernel = np.ones(5) / 5
            smoothed = np.convolve(strip, kernel, mode='same')

            # Find local maxima that are strong
            threshold = ink_thresh * 1.5
            peaks = []
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                    peaks.append(r0 + i)

            if len(peaks) < 5:
                continue

            # Take the 5 strongest peaks
            peaks_sorted = sorted(peaks, key=lambda p: proj[p], reverse=True)[:5]
            peaks_sorted.sort()

            # Verify spacing consistency
            gaps = [peaks_sorted[i+1] - peaks_sorted[i] for i in range(4)]
            avg_gap = np.mean(gaps) if gaps else 10
            if avg_gap < 3 or avg_gap > h // 15:
                continue

            # Check no gap is wildly different
            if any(abs(g - avg_gap) > avg_gap * 0.5 for g in gaps):
                # Try to pick better lines
                peaks_sorted = self._refine_lines(peaks, avg_gap)

            clef = ClefType.TREBLE if idx == 0 else ClefType.BASS
            staves.append({
                'y_positions': peaks_sorted,
                'clef': clef,
                'staff_idx': idx,
                'space': np.mean([peaks_sorted[i+1] - peaks_sorted[i] for i in range(4)])
            })

        return staves

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Convert to binary with staff-appropriate thresholding."""
        # Use Otsu for global threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _refine_lines(self, peaks: List[int], avg_gap: float) -> List[int]:
        """Refine line selection to get evenly spaced set of 5."""
        if len(peaks) < 5:
            return peaks[:5] if len(peaks) >= 5 else peaks

        # Try to pick 5 lines with most consistent spacing
        best_combo = None
        best_score = float('inf')
        for i in range(len(peaks) - 4):
            combo = peaks[i:i+5]
            gaps = [combo[j+1] - combo[j] for j in range(4)]
            score = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else float('inf')
            if score < best_score:
                best_score = score
                best_combo = combo

        return best_combo or peaks[:5]


class NoteDetector:
    """Detects noteheads relative to detected staves."""

    def detect(self, gray: np.ndarray, staves: List[dict],
               barlines: List[float]) -> List[dict]:
        """
        Returns list of detected notes, each with:
        {x, y, pitch, midi, filled, stem_dir, measure, staff_idx}
        """
        h, w = gray.shape
        binary = self._binarize(gray)
        notes = []

        for staff in staves:
            staff_notes = self._detect_in_staff(gray, binary, staff, barlines, h, w)
            notes.extend(staff_notes)

        return notes

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _detect_in_staff(self, gray: np.ndarray, binary: np.ndarray,
                          staff: dict, barlines: List[float], h, w) -> List[dict]:
        """Detect noteheads in one staff."""
        y_lines = staff['y_positions']
        space = staff['space']
        y0 = max(0, y_lines[0] - int(space * 3))
        y1 = min(h, y_lines[-1] + int(space * 3))

        region = binary[y0:y1, :]
        region_gray = gray[y0:y1, :]

        # Remove staff lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
        staff_lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, h_kernel)
        region_no_lines = cv2.subtract(region, staff_lines)

        # Find contours
        contours, _ = cv2.findContours(region_no_lines, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # Expected notehead size (based on staff space)
        min_size = (space * 0.3) ** 2
        max_size = (space * 1.5) ** 2

        notes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2 + y0  # Back to image coords

            # Check if in a measure
            measure = 1
            in_measure = False
            for i in range(len(barlines) - 1):
                if barlines[i] <= cx <= barlines[i+1]:
                    measure = i + 1
                    in_measure = True
                    break
            if not in_measure and len(barlines) > 1:
                continue

            # Skip very small contours (noise)
            if bw < 3 or bh < 3:
                continue

            # Filled or open?
            roi = region_gray[y:y+bh, x:x+bw]
            mean_val = np.mean(roi) if roi.size > 0 else 255
            filled = mean_val < 128

            # Pitch from y-position
            pitch, midi = self._y_to_pitch(cy, staff)

            # Detect stem
            stem_dir = self._detect_stem(region_no_lines, x, y, bw, bh, h, w)

            # Duration
            dur_type, dur_float = self._classify_duration(filled, stem_dir, bw, bh, space)

            note = {
                'x': cx,
                'y': cy,
                'pitch': pitch,
                'midi': midi,
                'filled': filled,
                'stem_dir': stem_dir,
                'measure': measure,
                'staff_idx': staff['staff_idx'],
                'duration_type': dur_type,
                'duration_float': dur_float,
                'bw': bw,
                'bh': bh,
                'confidence': 0.8 if filled else 0.6
            }
            notes.append(note)

        return notes

    def _y_to_pitch(self, y: float, staff: dict) -> Tuple[str, int]:
        """Convert y-position to pitch."""
        lines = staff['y_positions']
        space = staff['space']
        clef = staff['clef']

        # Distance from bottom line (in staff-space units)
        bottom_line = lines[-1]
        dy = (bottom_line - y) / space

        if clef == ClefType.TREBLE:
            # Bottom line = E4 = MIDI 64
            midi = 64 + round(dy * 2)
        elif clef == ClefType.BASS:
            # Bottom line = G2 = MIDI 43
            midi = 43 + round(dy * 2)
        else:
            midi = 64 + round(dy * 2)

        midi = max(0, min(127, midi))
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        note_name = note_names[midi % 12]

        return f"{note_name}{octave}", midi

    def _detect_stem(self, binary: np.ndarray, x: int, y: int,
                     bw: int, bh: int, h_img: int, w_img: int) -> str:
        """Detect stem direction above/below the notehead."""
        if bw < 2:
            return ""

        cx = x + bw // 2
        cy = y + bh // 2

        search_w = max(8, bw * 2)
        search_h = 40

        x_start = max(0, cx - search_w // 2)
        x_end = min(binary.shape[1], cx + search_w // 2)

        # Above
        y_above_start = max(0, cy - search_h)
        y_above_end = max(0, cy - 1)
        above = binary[y_above_start:y_above_end, x_start:x_end] if y_above_end > y_above_start else np.zeros((1, 1))

        # Below
        y_below_start = min(binary.shape[0] - 1, cy + bh)
        y_below_end = min(binary.shape[0], cy + search_h)
        below = binary[y_below_start:y_below_end, x_start:x_end] if y_below_end > y_below_start else np.zeros((1, 1))

        above_ink = np.sum(above) / 255 if above.size > 1 else 0
        below_ink = np.sum(below) / 255 if below.size > 1 else 0

        # A stem is a narrow column of ink
        # Check if the ink is concentrated in a narrow band
        above_col_ink = np.sum(above, axis=0) / 255 if above.ndim > 1 and above.shape[0] > 0 else np.zeros(x_end - x_start)
        below_col_ink = np.sum(below, axis=0) / 255 if below.ndim > 1 and below.shape[0] > 0 else np.zeros(x_end - x_start)

        # A stem should be about 2-4 pixels wide
        if np.max(above_col_ink) > max(3, search_h * 0.3):
            return "up"
        elif np.max(below_col_ink) > max(3, search_h * 0.5):
            return "down"

        return ""

    def _classify_duration(self, filled: bool, stem_dir: str,
                           bw: int, bh: int, space: float) -> Tuple[str, float]:
        if not filled and not stem_dir:
            return ("whole", 4.0)
        elif not filled and stem_dir:
            return ("half", 2.0)
        elif filled and stem_dir:
            # Quarter by default (beams/flags detected later)
            return ("quarter", 1.0)
        elif filled and not stem_dir:
            return ("quarter", 1.0)
        return ("quarter", 1.0)


class BarlinesDetector:
    """Detects measure barlines in the image."""

    def detect(self, gray: np.ndarray, staves: List[dict]) -> List[float]:
        h, w = gray.shape
        if not staves:
            return [0, w - 1]

        # Focus on staff region
        y_min = min(s['y_positions'][0] for s in staves) - 20
        y_max = max(s['y_positions'][-1] for s in staves) + 20
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        region = gray[y_min:y_max, :]
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Look for vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, region.shape[0] // 2))
        vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        proj = np.sum(vert, axis=0) / 255
        threshold = (y_max - y_min) * 0.3

        barlines = [0.0]
        in_bar = False
        for x in range(w):
            if proj[x] > threshold and not in_bar:
                barlines.append(float(x))
                in_bar = True
            elif proj[x] < threshold * 0.5:
                in_bar = False

        barlines.append(float(w - 1))

        # Deduplicate close barlines
        filtered = [barlines[0]]
        min_gap = w // 25
        for bl in barlines[1:]:
            if bl - filtered[-1] > min_gap:
                filtered.append(bl)

        return filtered


class VoiceAssigner:
    """Assign notes to voice parts based on stem direction and staff position."""

    def assign(self, notes: List[dict], staves: List[dict]) -> dict:
        voices = defaultdict(list)

        for note in notes:
            staff_idx = note.get('staff_idx', 0)
            stem_dir = note.get('stem_dir', '')

            if staff_idx == 0:
                part = VoicePart.SOPRANO if stem_dir in ('up', '') else VoicePart.ALTO
            else:
                part = VoicePart.TENOR if stem_dir in ('up', '') else VoicePart.BASS

            voices[part].append(note)

        return dict(voices)


def run_pipeline(image_path: str, debug: bool = False) -> ScoreResult:
    """Main entry point: image → ScoreResult."""
    img = cv2.imread(str(image_path))
    if img is None:
        return ScoreResult(status="error", message=f"Cannot read: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Step 1: Detect staff lines
    staff_det = StaffDetector()
    staves = staff_det.detect(gray)

    if not staves:
        return ScoreResult(status="error",
                           message="No staves detected. Ensure the image shows sheet music clearly.")

    # Step 2: Detect barlines
    bar_det = BarlinesDetector()
    barlines = bar_det.detect(gray, staves)

    # Step 3: Detect noteheads
    note_det = NoteDetector()
    notes = note_det.detect(gray, staves, barlines)

    # Step 4: Assign voices
    voice_assigner = VoiceAssigner()
    voice_notes = voice_assigner.assign(notes, staves)

    # Step 5: Build ScoreResult
    result = ScoreResult()
    result.status = "success"
    result.measures = max(len(barlines) - 1, 1)

    voice_order = [VoicePart.SOPRANO, VoicePart.ALTO, VoicePart.TENOR, VoicePart.BASS]
    for part in voice_order:
        part_notes = voice_notes.get(part, [])
        if not part_notes and part in [VoicePart.SOPRANO, VoicePart.TENOR]:
            # For voices with no up-stem notes, try creating from unassigned
            continue

        part_notes.sort(key=lambda n: (n['measure'], n['x']))

        staff_idx = 0 if part in [VoicePart.SOPRANO, VoicePart.ALTO] else 1
        clef = ClefType.TREBLE if staff_idx == 0 else ClefType.BASS

        note_data_list = []
        for i, n in enumerate(part_notes):
            nd = NoteData(
                pitch=n['pitch'],
                midi=n['midi'],
                duration_type=n['duration_type'],
                duration_float=n['duration_float'],
                stem_direction=n.get('stem_dir', ''),
                voice=part,
                lyric="",
                confidence=n.get('confidence', 0.7),
                x=n['x'],
                y=n['y'],
                measure=n.get('measure', 1),
                note_id=f"{part.value}_{n.get('measure', 1)}_{i}"
            )
            note_data_list.append(nd)

        vl = VoiceLine(
            part=part,
            notes=note_data_list,
            staff=staff_idx,
            clef=clef
        )
        result.voices.append(vl)

    return result