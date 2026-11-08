"""Kwaya — FastAPI backend for the Swahili choral score assistant.

Endpoints:
  POST /api/upload        — Upload PDF/PNG score
  POST /api/process       — Run OMR + solfege on uploaded score
  GET  /api/score/{id}    — Get score result JSON
  POST /api/correct       — Apply a correction to a misread note
  POST /api/synthesize    — Generate audio for one or more parts
  GET  /api/audio/{id}/{part}.wav  — Stream synthesized audio
  POST /api/lyrics         — Set/update lyrics for a voice
  GET  /api/export/{id}/{format}   — Export as MusicXML/PDF
  GET  /api/demo           — Load the demo score
"""
import os
import sys
import json
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    ScoreResult, VoiceLine, NoteData, VoicePart, ClefType,
    KeySignature, TimeSignature, CorrectionRequest, SynthesisRequest,
    UploadResponse
)
from music_engine import compute_solfege_for_score, build_demo_score
from synthesizer import ChoirSynthesizer, synthesize_voice

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "output"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store (simple, no DB needed for single user)
scores: dict = {}

# ── App Initialization ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load demo score on startup."""
    demo = build_demo_score()
    demo.score_id = "demo_001"
    scores["demo_001"] = demo
    yield
    # Cleanup
    scores.clear()

app = FastAPI(
    title="Kwaya — Choral Score Assistant",
    description="Upload sheet music, get solfege + synthetic choir playback",
    version="0.1.0",
    lifespan=lifespan
)

# CORS — allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper Functions ────────────────────────────────────────────────────

def _get_score(score_id: str) -> ScoreResult:
    score = scores.get(score_id)
    if not score:
        raise HTTPException(status_code=404, detail=f"Score {score_id} not found")
    return score

def _pydantic_to_dict(obj):
    """Convert Pydantic model to dict for JSON response."""
    return json.loads(obj.model_dump_json())


# ── API Endpoints ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "kwaya", "scores_loaded": len(scores)}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_score(file: UploadFile = File(...)):
    """Upload a PDF or PNG score image."""
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif',
                   '.musicxml', '.xml', '.mxl'):
        raise HTTPException(400, f"Unsupported format: {ext}. Use PDF, PNG, JPEG, or MusicXML.")

    score_id = hashlib.md5(file.filename.encode()).hexdigest()[:12]

    # Save file
    dest = UPLOAD_DIR / f"{score_id}{ext}"
    with open(dest, 'wb') as f:
        content = await file.read()
        f.write(content)

    return UploadResponse(
        score_id=score_id,
        filename=file.filename,
        pages=1,
        message=f"Uploaded {file.filename}" + (" — ready for processing" if ext == '.png' else "")
    )


@app.post("/api/process")
async def process_score(score_id: str = Form(...)):
    """Run OMR + solfege processing.

    Supports:
    • Already-processed scores (return cached)
    • MusicXML uploads (parsed via music21 — most reliable)
    • PNG scores (OpenCV OMR — experimental, works on clean scans)
    """
    # Check for existing score
    if score_id in scores and scores[score_id].voices:
        return _pydantic_to_dict(scores[score_id])

    # Try to find uploaded file
    path = None
    ext_found = None
    for ext in ('.musicxml', '.xml', '.mxl', '.pdf', '.png', '.jpg', '.jpeg', '.tiff'):
        p = UPLOAD_DIR / f"{score_id}{ext}"
        if p.exists():
            path = p
            ext_found = ext
            break

    if not path:
        placeholder = ScoreResult(
            status="partial", message="No uploaded file found for this score ID.",
            score_id=score_id
        )
        scores[score_id] = placeholder
        return _pydantic_to_dict(placeholder)

    # ── MusicXML path (most reliable) ──────────────────────────────
    if ext_found in ('.musicxml', '.xml', '.mxl'):
        try:
            from music21 import converter, stream, note, chord, clef, key, meter, pitch as m21_pitch
            m21_score = converter.parse(str(path))

            result = ScoreResult(
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                status="success", score_id=score_id
            )

            # Detect key and time from first part
            first_part = list(m21_score.parts)[0] if m21_score.parts else None
            if first_part:
                ks = first_part.recurse().getElementsByClass(key.Key).first()
                if ks:
                    result.key_signature = KeySignature(
                        fifths=ks.sharps,
                        mode=ks.mode,
                        description=str(ks)
                    )
                ts = first_part.recurse().getElementsByClass(meter.TimeSignature).first()
                if ts:
                    result.time_signature = TimeSignature(
                        numerator=ts.numerator, denominator=ts.denominator
                    )

            # Extract each part as a voice
            voice_order = [VoicePart.SOPRANO, VoicePart.ALTO,
                           VoicePart.TENOR, VoicePart.BASS]
            max_measure = 0

            for idx, m21_part in enumerate(m21_score.parts):
                if idx >= 4:
                    break
                part = voice_order[idx]

                # Determine clef
                clef_found = ClefType.TREBLE
                c = m21_part.recurse().getElementsByClass(clef.Clef).first()
                if c:
                    if c.name.lower() == 'bass':
                        clef_found = ClefType.BASS

                staff_idx = 0 if part in (VoicePart.SOPRANO, VoicePart.ALTO) else 1

                note_list = []
                for el in m21_part.recurse().notesAndRests:
                    if el.isRest:
                        continue
                    if el.isChord:
                        # Chord: take top note for soprano/tenor, bottom for alto/bass
                        notes_in_chord = el.notes
                        if part in (VoicePart.SOPRANO, VoicePart.TENOR):
                            chosen = notes_in_chord[-1]  # highest
                        else:
                            chosen = notes_in_chord[0]   # lowest
                    else:
                        chosen = el

                    dur_float = el.duration.quarterLength
                    if dur_float >= 3.0:
                        dur_type = "whole"
                    elif dur_float >= 1.5:
                        dur_type = "half"
                    elif dur_float >= 0.75:
                        dur_type = "quarter"
                    elif dur_float >= 0.375:
                        dur_type = "eighth"
                    else:
                        dur_type = "16th"

                    # Get measure number
                    meas = 1
                    try:
                        meas = max(1, int(el.measureNumber or 1))
                    except:
                        pass
                    max_measure = max(max_measure, meas)

                    # Lyric
                    lyric = ""
                    try:
                        from music21 import lyrics
                        for ly in el.lyrics:
                            if ly and ly.text:
                                lyric = ly.text
                                break
                    except:
                        pass

                    nd = NoteData(
                        pitch=chosen.nameWithOctave,
                        midi=chosen.pitch.midi,
                        duration_type=dur_type,
                        duration_float=dur_float,
                        stem_direction="up" if part in (VoicePart.SOPRANO, VoicePart.TENOR) else "down",
                        voice=part,
                        lyric=lyric,
                        measure=meas,
                        note_id=f"{part.value}_{meas}_{len(note_list)}"
                    )
                    note_list.append(nd)

                vl = VoiceLine(
                    part=part,
                    notes=note_list,
                    staff=staff_idx,
                    clef=clef_found,
                    lyrics=[n.lyric for n in note_list]
                )
                result.voices.append(vl)

            result.measures = max_measure
            result = compute_solfege_for_score(result)
            scores[score_id] = result
            return _pydantic_to_dict(result)

        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=422,
                content={"status": "error",
                         "message": f"MusicXML parsing failed: {e}",
                         "detail": traceback.format_exc(),
                         "suggestion": "Check the file format or try the Demo score."}
            )

    # ── PDF path (render first page as image, then run OMR) ────────
    if ext_found == '.pdf':
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            first_page = doc[0]
            pix = first_page.get_pixmap(dpi=200)
            png_path = path.with_suffix('.png')
            pix.save(str(png_path))
            # Fall through to PNG OMR below
            path = png_path
            ext_found = '.png'
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"status": "error",
                         "message": f"PDF rendering failed: {str(e)}",
                         "suggestion": "Try uploading as MusicXML or use the Demo score."}
            )

    # ── PNG path (OpenCV OMR — experimental) ──────────────────────
    if ext_found == '.png':
        try:
            from omr.engine import run_pipeline
            result = run_pipeline(str(path))
            result.score_id = score_id
            if result.status == "success" and result.voices and sum(len(v.notes) for v in result.voices) > 5:
                result = compute_solfege_for_score(result)
                scores[score_id] = result
                return _pydantic_to_dict(result)
            else:
                # OMR failed or too few notes — return helpful message
                return JSONResponse(
                    status_code=422,
                    content={"status": "error",
                             "message": "OMR could not reliably detect notes from this image.",
                             "suggestion": "Export the score as MusicXML from notation software and upload that instead. Or try the Demo score."}
                )
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"status": "error",
                         "message": f"OMR failed: {str(e)}",
                         "suggestion": "Export the score as MusicXML from notation software and upload that instead."}
            )

    # ── Fallback ──────────────────────────────────────────────────
    placeholder = ScoreResult(
        status="partial",
        message=f"Uploaded {ext_found} — direct OMR is only available for PNG scores. "
                "MusicXML files parse instantly. Try the Demo score to test the full pipeline.",
        score_id=score_id
    )
    scores[score_id] = placeholder
    return _pydantic_to_dict(placeholder)


@app.get("/api/score/{score_id}")
async def get_score(score_id: str):
    """Get the full score result."""
    return _pydantic_to_dict(_get_score(score_id))


@app.post("/api/correct")
async def correct_note(req: CorrectionRequest):
    """Apply a correction to a note."""
    score = _get_score(req.score_id)
    voice = score.get_voice(req.part)
    if not voice:
        raise HTTPException(404, f"Voice part {req.part} not found")

    # Find notes in the given measure
    measure_notes = [n for n in voice.notes if n.measure == req.measure]
    if req.note_index >= len(measure_notes):
        raise HTTPException(404, f"Note index {req.note_index} out of range in measure {req.measure}")

    target = measure_notes[req.note_index]

    if req.new_pitch:
        target.pitch = req.new_pitch
        from music21 import pitch
        try:
            p = pitch.Pitch(req.new_pitch)
            target.midi = p.midi
        except:
            pass
    if req.new_solfege:
        target.solfege = req.new_solfege
    if req.new_duration:
        target.duration_type = req.new_duration
    if req.new_lyric:
        target.lyric = req.new_lyric

    return {"status": "corrected", "note_id": target.note_id,
            "new": {"pitch": target.pitch, "solfege": target.solfege,
                    "lyric": target.lyric}}


@app.post("/api/lyrics")
async def update_lyrics(score_id: str = Form(...),
                         part: VoicePart = Form(...),
                         lyrics_json: str = Form(...)):
    """Set lyrics for a voice part (JSON array of strings)."""
    score = _get_score(score_id)
    voice = score.get_voice(part)
    if not voice:
        raise HTTPException(404, f"Voice part {part} not found")

    try:
        lyrics = json.loads(lyrics_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in lyrics_json")

    for i, lyric in enumerate(lyrics):
        if i < len(voice.notes):
            voice.notes[i].lyric = lyric

    voice.lyrics = lyrics
    return {"status": "updated", "part": part.value, "count": len(lyrics)}


@app.post("/api/synthesize")
async def synthesize(req: SynthesisRequest):
    """Generate audio for voice parts."""
    score = _get_score(req.score_id)

    part_list = req.parts
    if not part_list:
        part_list = [VoicePart.SOPRANO, VoicePart.ALTO,
                     VoicePart.TENOR, VoicePart.BASS]

    results = {}
    for part in part_list:
        voice = score.get_voice(part)
        if voice and voice.notes:
            synth = ChoirSynthesizer()
            audio = synth.synthesize(voice, req.tempo, score.key_signature.fifths)

            # Save as WAV
            wav_path = OUTPUT_DIR / f"{req.score_id}_{part.value}.wav"
            ChoirSynthesizer.save_wav(audio, str(wav_path))
            results[part.value] = f"/api/audio/{req.score_id}/{part.value}.wav"

    return {"status": "synthesized", "files": results, "tempo": req.tempo}


@app.get("/api/audio/{score_id}/{part}.wav")
async def get_audio(score_id: str, part: str):
    """Stream synthesized audio."""
    wav_path = OUTPUT_DIR / f"{score_id}_{part}.wav"
    if not wav_path.exists():
        raise HTTPException(404, f"Audio not found for {part}. Synthesize first.")

    # Read the file and return raw bytes (FileResponse has issues on Windows)
    try:
        data = wav_path.read_bytes()
        return Response(content=data, media_type="audio/wav",
                        headers={
                            "Content-Disposition": f'inline; filename="{score_id}_{part}.wav"',
                            "Content-Length": str(len(data))
                        })
    except Exception as e:
        raise HTTPException(500, f"Failed to read audio: {e}")


@app.get("/api/demo")
async def load_demo():
    """Load the built-in demo score (Swahili hymn)."""
    demo = build_demo_score()
    demo.score_id = "demo_001"
    scores["demo_001"] = demo

    return _pydantic_to_dict(demo)


@app.get("/api/export/{score_id}/musicxml")
async def export_musicxml(score_id: str):
    """Export score as MusicXML."""
    score = _get_score(score_id)
    if not score.voices:
        raise HTTPException(404, "No voices to export")

    # Build MusicXML using music21
    from music21 import stream, note, chord, meter, key, clef, tempo as m21_tempo

    s = stream.Score()

    for voice in score.voices:
        part = stream.Part()
        part.partName = voice.part.value.capitalize()

        # Add clef
        if voice.clef == ClefType.BASS:
            part.append(clef.BassClef())
        else:
            part.append(clef.TrebleClef())

        # Add key signature
        ks = key.Key()
        ks.sharps = score.key_signature.fifths  # negative = flats
        part.append(ks)

        # Add time signature
        ts = meter.TimeSignature(
            f"{score.time_signature.numerator}/{score.time_signature.denominator}")
        part.append(ts)

        # Group notes by measure
        measures = {}
        for n in voice.notes:
            m = n.measure
            if m not in measures:
                measures[m] = []
            measures[m].append(n)

        for m_num in sorted(measures.keys()):
            m_stream = stream.Measure(number=m_num)
            for n_data in measures[m_num]:
                try:
                    n = note.Note(n_data.pitch)
                    if n_data.duration_type == "whole":
                        n.duration.quarterLength = 4.0
                    elif n_data.duration_type == "half":
                        n.duration.quarterLength = 2.0
                    elif n_data.duration_type == "quarter":
                        n.duration.quarterLength = 1.0
                    elif n_data.duration_type == "eighth":
                        n.duration.quarterLength = 0.5
                    elif n_data.duration_type == "16th":
                        n.duration.quarterLength = 0.25
                    else:
                        n.duration.quarterLength = n_data.duration_float

                    # Add lyric
                    if n_data.lyric:
                        from music21 import lyrics as m21_lyrics
                        n.addLyric(n_data.lyric)

                    m_stream.append(n)
                except Exception:
                    pass
            part.append(m_stream)

        s.append(part)

    xml_str = s.write('musicxml')
    with open(xml_str, 'r', encoding='utf-8') as f:
        content = f.read()
    Path(xml_str).unlink(missing_ok=True)

    return Response(content=content, media_type="application/xml",
                    headers={"Content-Disposition": f"attachment; filename={score_id}.musicxml"})


# ── Serve Static Frontend ───────────────────────────────────────────────
# Mount static files at /static/ (not at / — that would intercept API routes)
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the chat frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        return Response(content=content, media_type="text/html")
    return Response("<h1>Kwaya — Choral Score Assistant</h1><p>Frontend not found. Run the server and open /</p>", media_type="text/html")


# ── Chat Endpoint — Agentic Interface ───────────────────────────────────

@app.post("/api/chat")
async def chat(req: Request):
    """Agentic chat endpoint — understands natural language commands.

    Routes messages to the right action and returns structured responses
    that the chat frontend renders as rich content.
    """
    body = await req.json()
    msg = body.get("message", "").strip()
    score_id = body.get("score_id", "demo_001")
    msg_lower = msg.lower()

    # ── Help / Greeting ──────────────────────────────────────────────
    if not msg or msg_lower in ("help", "hi", "hello", "start"):
        return {
            "role": "assistant",
            "type": "text",
            "content": (
                "👋 **Karibu Kwaya!** I'm your choral score assistant.\n\n"
                "Here's what I can do:\n\n"
                "• **Upload a score** — drag a PNG/PDF into the chat or click 📎\n"
                "• **`load demo`** — try the built-in Swahili hymn \"Mungu Ni Baba\"\n"
                "• **`synthesize`** / **`play`** — hear all voices\n"
                "• **`correct [part] m.[N] note [i] to [pitch]`** — fix a misread note\n"
                "• **`export`** — download as MusicXML\n"
                "• **`tempo [bpm]`** — change playback speed\n"
                "• Ask me anything about the score!\n\n"
                "Tunaanza? 🎵"
            )
        }

    # ── Load Demo Score ──────────────────────────────────────────────
    if msg_lower in ("load demo", "demo", "demo score"):
        try:
            demo = build_demo_score()
            demo.score_id = "demo_001"
            scores["demo_001"] = demo

            # Build voice summary
            parts_summary = []
            for v in demo.voices:
                total_beats = sum(n.duration_float for n in v.notes)
                parts_summary.append({
                    "part": v.part.value,
                    "note_count": len(v.notes),
                    "duration_beats": round(total_beats, 1)
                })

            return {
                "role": "assistant",
                "type": "score_loaded",
                "content": f"✅ Loaded **{demo.title}** by {demo.composer}",
                "data": {
                    "score_id": "demo_001",
                    "score": _pydantic_to_dict(demo),
                    "summary": parts_summary
                }
            }
        except Exception as e:
            return {"role": "assistant", "type": "error",
                    "content": f"❌ Failed to load demo: {str(e)}"}

    # ── Synthesize / Play ────────────────────────────────────────────
    if any(w in msg_lower for w in ("synthesize", "play", "sing", "hear", "listen")):
        score = scores.get(score_id)
        if not score or not score.voices:
            return {"role": "assistant", "type": "error",
                    "content": "No score loaded. Try `load demo` first or upload one."}

        # Parse tempo from message
        tempo = 80
        import re
        tempo_match = re.search(r'tempo\s*(\d+)', msg_lower)
        if tempo_match:
            tempo = max(40, min(200, int(tempo_match.group(1))))

        try:
            results = {}
            for voice in score.voices:
                if voice.notes:
                    synth = ChoirSynthesizer()
                    audio = synth.synthesize(voice, tempo, score.key_signature.fifths)
                    wav_path = OUTPUT_DIR / f"{score_id}_{voice.part.value}.wav"
                    ChoirSynthesizer.save_wav(audio, str(wav_path))
                    results[voice.part.value] = f"/api/audio/{score_id}/{voice.part.value}.wav"

            return {
                "role": "assistant",
                "type": "audio_ready",
                "content": f"🔊 Synthesized {len(results)} voice(s) at ♩={tempo} bpm",
                "data": {"files": results, "tempo": tempo}
            }
        except Exception as e:
            return {"role": "assistant", "type": "error",
                    "content": f"❌ Synthesis failed: {str(e)}"}

    # ── Export ───────────────────────────────────────────────────────
    if msg_lower in ("export", "export musicxml", "download", "xml"):
        score = scores.get(score_id)
        if not score or not score.voices:
            return {"role": "assistant", "type": "error",
                    "content": "Nothing to export. Load a score first."}
        try:
            # Build MusicXML using music21 (same logic as /api/export endpoint)
            from music21 import stream, note, meter, key, clef as m21_clef
            s = stream.Score()
            for voice in score.voices:
                p = stream.Part()
                p.partName = voice.part.value.capitalize()
                p.append(m21_clef.TrebleClef() if voice.clef == ClefType.TREBLE else m21_clef.BassClef())
                p.append(key.Key(score.key_signature.description.split()[0]))
                p.append(meter.TimeSignature(f"{score.time_signature.numerator}/{score.time_signature.denominator}"))
                meas = {}
                for nd in voice.notes:
                    mnum = nd.measure or 1
                    meas.setdefault(mnum, []).append(nd)
                for mnum in sorted(meas):
                    ms = stream.Measure(number=mnum)
                    for nd in meas[mnum]:
                        try:
                            no = note.Note(nd.pitch)
                            no.duration.quarterLength = nd.duration_float
                            if nd.lyric:
                                from music21 import lyrics
                                no.addLyric(nd.lyric)
                            ms.append(no)
                        except Exception:
                            pass
                    p.append(ms)
                s.append(p)
            xml_path = s.write('musicxml')
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            Path(xml_path).unlink(missing_ok=True)
            return {
                "role": "assistant",
                "type": "export",
                "content": "📄 MusicXML exported — click to download",
                "data": {"xml": content, "filename": f"{score_id}.musicxml"}
            }
        except Exception as e:
            return {"role": "assistant", "type": "error",
                    "content": f"❌ Export failed: {str(e)}"}

    # ── Correct a note ──────────────────────────────────────────────
    # Pattern: "correct [part] m.[N] note [i] to [pitch/solfege]"
    if "correct" in msg_lower:
        import re
        # Try to parse: correct soprano m.2 note 0 to F#5
        cm = re.search(r'correct\s+(\w+)\s+m\.?\s*(\d+)\s+note\s+(\d+)\s+to\s+(\S+)', msg_lower)
        if cm:
            part_str = cm.group(1)
            measure = int(cm.group(2))
            note_idx = int(cm.group(3))
            target = cm.group(4)

            # Map part string to VoicePart
            part_map = {"soprano": VoicePart.SOPRANO, "alto": VoicePart.ALTO,
                        "tenor": VoicePart.TENOR, "bass": VoicePart.BASS}
            part = part_map.get(part_str)
            if not part:
                return {"role": "assistant", "type": "error",
                        "content": f"Unknown part '{part_str}'. Use soprano, alto, tenor, or bass."}

            score = scores.get(score_id)
            if not score:
                return {"role": "assistant", "type": "error",
                        "content": "No score loaded."}

            voice = score.get_voice(part)
            if not voice:
                return {"role": "assistant", "type": "error",
                        "content": f"No {part_str} voice found."}

            measure_notes = [n for n in voice.notes if n.measure == measure]
            if note_idx >= len(measure_notes):
                return {"role": "assistant", "type": "error",
                        "content": f"Note index {note_idx} out of range in m.{measure} (only {len(measure_notes)} notes)."}

            target_note = measure_notes[note_idx]
            old_pitch = target_note.pitch
            old_solfege = target_note.solfege

            # Determine if target is a pitch or solfege
            if any(c.isdigit() for c in target):
                target_note.pitch = target.upper()
                # Recompute solfege
                from music21 import pitch
                try:
                    p = pitch.Pitch(target.upper())
                    target_note.midi = p.midi
                except:
                    pass
            else:
                target_note.solfege = target.lower()

            return {
                "role": "assistant",
                "type": "corrected",
                "content": (f"✅ Corrected **{part_str}** m.{measure} note {note_idx}: "
                           f"`{old_pitch}/{old_solfege}` → `{target_note.pitch}/{target_note.solfege}`"),
                "data": {
                    "part": part_str,
                    "measure": measure,
                    "note_index": note_idx,
                    "old": {"pitch": old_pitch, "solfege": old_solfege},
                    "new": {"pitch": target_note.pitch, "solfege": target_note.solfege}
                }
            }

        return {"role": "assistant", "type": "text",
                "content": "To correct a note, say something like:\n"
                           "`correct soprano m.2 note 0 to F#5`\n"
                           "or `correct alto m.1 note 2 to fi`"}

    # ── Tempo ────────────────────────────────────────────────────────
    if "tempo" in msg_lower:
        import re
        tm = re.search(r'tempo\s*(\d+)', msg_lower)
        if tm:
            bpm = max(40, min(200, int(tm.group(1))))
            return {"role": "assistant", "type": "text",
                    "content": f"⏱ Set tempo to **♩={bpm}** bpm. Use `synthesize` to re-render with the new tempo."}
        return {"role": "assistant", "type": "text",
                "content": "Current tempo is 80 bpm. Say `tempo 120` to change it."}

    # ── Lyrics ───────────────────────────────────────────────────────
    if "lyrics" in msg_lower or "words" in msg_lower:
        return {"role": "assistant", "type": "text",
                "content": "You can update lyrics in the solfege display above — "
                           "click the ✏️ on any note, or type comma-separated lyrics in the input box."}

    # ── Score info ───────────────────────────────────────────────────
    if any(w in msg_lower for w in ("info", "score", "what", "status", "voices", "notes")):
        score = scores.get(score_id)
        if not score or not score.voices:
            return {"role": "assistant", "type": "text",
                    "content": "No score loaded. Try `load demo` or upload one."}
        total_notes = sum(len(v.notes) for v in score.voices)
        lines = [f"📊 **{score.title}**"]
        lines.append(f"   Key: {score.key_signature.description}  |  Time: {score.time_signature.numerator}/{score.time_signature.denominator}")
        lines.append(f"   Measures: {score.measures}  |  Total notes: {total_notes}")
        for v in score.voices:
            lines.append(f"   • **{v.part.value.capitalize()}**: {len(v.notes)} notes")
        return {"role": "assistant", "type": "text", "content": "\n".join(lines)}

    # ── Quality / sound questions ────────────────────────────────────
    if any(w in msg_lower for w in ("robotic", "sound", "quality", "robotic", "synthesizer")):
        return {"role": "assistant", "type": "text",
                "content": (
                    "🎛️ **About the audio quality:**\n\n"
                    "Currently I use **pure-numpy formant synthesis** — each note is generated "
                    "as a harmonic series shaped by vowel formants (F1, F2, F3). This is why it "
                    "sounds a bit robotic compared to a real choir.\n\n"
                    "**To improve the sound, I can:**\n"
                    "1. Set up **FluidSynth** with a choir SoundFont (we have `GM.sf2` already)\n"
                    "2. Add **vibrato and portamento** between notes\n"
                    "3. Layer **multiple formants** per voice for richness\n\n"
                    "Want me to work on the audio quality?"
                )}

    # ── Fallback: general response ───────────────────────────────────
    return {
        "role": "assistant",
        "type": "text",
        "content": (
            f"I understand you said: *\"{msg[:100]}\"*\n\n"
            "Here's what I can help with:\n"
            "• `load demo` — try the demo score\n"
            "• `synthesize` — hear the choir\n"
            "• `info` — see score details\n"
            "• `correct soprano m.2 note 0 to F#5` — fix a note\n"
            "• `export` — download MusicXML\n"
            "• `help` — see all commands\n\n"
            "Or just upload a score PNG/PDF and I'll read it! 🎵"
        )
    }


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("KWAYA_PORT", 8742))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)