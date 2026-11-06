"""Lightweight OMR — detects staff positions, noteheads, and stems.

Designed specifically for SATB choral scores with two voices per staff.
Uses OpenCV heuristics — no heavy ML models.
The correction UI in the frontend is the safety net for imperfect detections.

v2: morphology-based staff detection (robust on real scans),
    cleaned-up note detection with better stem/flag handling.
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
    """Detects staff lines using morphology + gap-filling dilation.

    Robust on real scans: noteheads/stems break staff lines into segments,
    so we first dilate horizontally to bridge the gaps, then extract long
    horizontal runs. Staff lines cluster into 5-line groups by spacing.
    """

    def detect(self, gray: np.ndarray) -> List[dict]:
        h, w = gray.shape
        binary = self._binarize(gray)

        # Step 1: fill gaps in staff lines (bridges noteheads, stems, lyrics)
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        filled = cv2.dilate(binary, fill_kernel, iterations=1)

        # Step 2: keep only long horizontal strips (staff lines) and
        #         discard vertical note elements
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        lines_img = cv2.morphologyEx(filled, cv2.MORPH_OPEN, open_kernel)

        # Step 3: row projection of extracted lines
        proj = np.sum(lines_img, axis=1) / 255
        strong_thresh = w * 0.35

        # Find candidate line rows (clusters of consecutive strong rows)
        line_rows = []
        y = 0
        while y < h:
            if proj[y] > strong_thresh:
                start = y
                while y < h and proj[y] > strong_thresh * 0.5:
                    y += 1
                # Use weighted center of the run
                run = proj[start:y]
                idxs = np.arange(start, y)
                center = int(np.average(idxs, weights=run))
                line_rows.append(center)
            else:
                y += 1

        if len(line_rows) < 5:
            return []

        # Step 4: group into staves (5 evenly-spaced lines with consistent gap)
        staves = self._group_into_staves(line_rows, h, w)

        # Step 5: assign clefs — staves alternate treble, bass (top to bottom)
        for idx, staff in enumerate(staves):
            staff['staff_idx'] = idx
            staff['clef'] = ClefType.TREBLE if idx % 2 == 0 else ClefType.BASS

        return staves

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        # Otsu binarization; invert so ink = white.
        # NOTE: no morphological cleanup here — at scan DPI, staff lines are
        # only 1-2px thick and any erosion (even a 2x2 open) destroys them.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _group_into_staves(self, line_rows: List[int], h: int, w: int) -> List[dict]:
        """Group detected line rows into logical staves of 5 evenly-spaced lines.

        A staff's 5 lines are close together — the gap between consecutive
        lines is small (roughly h/200 to h/40 at typical scan DPI; we observed
        ~10px at 150 DPI on a 1650px page). We only accept candidate staves
        whose predicted line spacing is in that plausible range, which
        prevents accidentally grouping the FIRST line of many staves.
        """
        rows = sorted(line_rows)

        # Plausible staff-line gap (in pixels): depends on page height
        gap_min = max(4, int(h / 300))
        gap_max = int(h / 40)          # generous upper bound

        candidates = []   # (score, ys, gap)
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                gap = rows[j] - rows[i]
                if gap < gap_min or gap > gap_max:
                    continue
                # Predict the 5 positions from this pair
                predicted = [rows[i] + k * gap for k in range(5)]
                matched = []
                used_for_match = set()
                ok = True
                for k, p in enumerate(predicted):
                    if k == 0:
                        # The first predicted position IS the anchor row
                        matched.append(i)
                        used_for_match.add(i)
                        continue
                    best_row = None
                    best_d = float('inf')
                    for idx, r in enumerate(rows):
                        if idx in used_for_match:
                            continue
                        d = abs(r - p)
                        if d < best_d and d <= gap * 0.4:
                            best_d = d
                            best_row = idx
                    if best_row is None:
                        ok = False
                        break
                    matched.append(best_row)
                    used_for_match.add(best_row)
                if not ok:
                    continue
                gaps = [rows[matched[k+1]] - rows[matched[k]] for k in range(4)]
                avg = np.mean(gaps)
                score = np.std(gaps) / avg if avg > 0 else float('inf')
                if score < 0.12:
                    ys = [rows[idx] for idx in matched]
                    candidates.append((score, sorted(ys), avg))

        # Deduplicate overlapping candidates — keep the most consistent one
        candidates.sort(key=lambda c: (c[0], c[1][0]))
        chosen = []
        for score, ys, avg_gap in candidates:
            overlap = False
            for _, chosen_ys, _ in chosen:
                shared = set(ys) & set(chosen_ys)
                if shared:
                    overlap = True
                    break
            if not overlap:
                chosen.append((score, ys, avg_gap))

        # Sort top-to-bottom, cap at sensible number (e.g. 12 systems)
        chosen.sort(key=lambda c: c[1][0])
        staves = []
        for score, ys, avg_gap in chosen[:12]:
            staves.append({'y_positions': ys, 'space': avg_gap})
        return staves


class NoteDetector:
    """Detects noteheads relative to detected staves, with stem detection."""

    def detect(self, gray: np.ndarray, staves: List[dict],
               barlines: List[float]) -> List[dict]:
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
        y_lines = staff['y_positions']
        space = staff['space']
        y0 = max(0, int(y_lines[0] - space * 3))
        y1 = min(h, int(y_lines[-1] + space * 3))

        region = binary[y0:y1, :]

        # Remove staff lines (long horizontal runs) before contour detection
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                 (max(w // 20, 30), 1))
        staff_lines_img = cv2.morphologyEx(region, cv2.MORPH_OPEN, line_kernel)
        region_no_lines = cv2.subtract(region, staff_lines_img)

        contours, _ = cv2.findContours(region_no_lines, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        min_size = (space * 0.3) ** 2
        max_size = (space * 1.5) ** 2

        notes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Require roughly circular/elliptical notehead (aspect ratio)
            if bh > 0 and (bw / bh > 2.5 or bh / bw > 2.5):
                continue

            cx = x + bw // 2
            cy = y + bh // 2 + y0

            measure = 1
            in_measure = False
            for i in range(len(barlines) - 1):
                if barlines[i] <= cx <= barlines[i + 1]:
                    measure = i + 1
                    in_measure = True
                    break
            if not in_measure and len(barlines) > 1:
                continue

            if bw < 3 or bh < 3:
                continue

            roi = gray[y0 + y:y0 + y + bh, x:x + bw]
            mean_val = np.mean(roi) if roi.size > 0 else 255
            filled = mean_val < 128

            pitch, midi = self._y_to_pitch(cy, staff)

            stem_dir = self._detect_stem(region_no_lines, x, y, bw, bh)

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
        lines = staff['y_positions']
        space = staff['space']
        clef = staff.get('clef', ClefType.TREBLE)

        bottom_line = lines[-1]
        dy = (bottom_line - y) / space

        if clef == ClefType.TREBLE:
            # Bottom line = E4 = MIDI 64; each line/space step = 2 semitones
            midi = 64 + round(dy * 2)
        elif clef == ClefType.BASS:
            # Bottom line = G2 = MIDI 43
            midi = 43 + round(dy * 2)
        else:
            midi = 64 + round(dy * 2)

        midi = max(0, min(127, midi))
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        name = note_names[midi % 12]
        return f"{name}{octave}", midi

    def _detect_stem(self, binary: np.ndarray, x: int, y: int,
                     bw: int, bh: int) -> str:
        """Detect stem direction by looking above/below the notehead."""
        if bw < 2:
            return ""

        cx = x + bw // 2
        cy = y + bh // 2
        search_w = max(8, bw * 2)
        search_h = 40

        x_start = max(0, cx - search_w // 2)
        x_end = min(binary.shape[1], cx + search_w // 2)

        y_above_start = max(0, cy - search_h)
        y_above_end = max(0, cy - 1)
        y_below_start = min(binary.shape[0], cy + bh)
        y_below_end = min(binary.shape[0], cy + search_h)

        def col_ink(y0, y1):
            if y1 <= y0:
                return 0.0
            strip = binary[y0:y1, x_start:x_end]
            col_counts = np.sum(strip, axis=0) / 255
            return float(np.max(col_counts)) if col_counts.size else 0.0

        above = col_ink(y_above_start, y_above_end)
        below = col_ink(y_below_start, y_below_end)

        # A stem is a narrow (2-4px) tall column of ink
        if above > max(4, search_h * 0.3):
            return "up"
        elif below > max(4, search_h * 0.5):
            return "down"

        return ""

    def _classify_duration(self, filled: bool, stem_dir: str,
                           bw: int, bh: int, space: float) -> Tuple[str, float]:
        if not filled and not stem_dir:
            return ("whole", 4.0)
        elif not filled and stem_dir:
            return ("half", 2.0)
        elif filled and stem_dir:
            return ("quarter", 1.0)
        return ("quarter", 1.0)


class BarlinesDetector:
    """Detects measure barlines using vertical morphology."""

    def detect(self, gray: np.ndarray, staves: List[dict]) -> List[float]:
        h, w = gray.shape
        if not staves:
            return [0, w - 1]

        y_min = max(0, int(min(s['y_positions'][0] for s in staves) - 20))
        y_max = min(h, int(max(s['y_positions'][-1] for s in staves) + 20))

        region = gray[y_min:y_max, :]
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Extract vertical lines taller than half the region
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (1, max((y_max - y_min) // 2, 8)))
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

        # Deduplicate
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

    # Step 1: Detect staff lines (morphology-based)
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
    result.status = "success" if notes else "partial"
    if not notes:
        result.message = f"Detected {len(staves)} staves but no noteheads. Try a clearer scan."
    result.measures = max(len(barlines) - 1, 1)

    voice_order = [VoicePart.SOPRANO, VoicePart.ALTO, VoicePart.TENOR, VoicePart.BASS]
    for part in voice_order:
        part_notes = voice_notes.get(part, [])
        if not part_notes:
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