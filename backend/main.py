"""
Infinite Loop – AI Backend v3
Real-Time Voice Cloning Detection
==================================
Stack:
  Voice:        RawNet2 + AASIST (MPS/CPU)
  Transcription: OpenAI Whisper (local, base model)
  Conversation:  Groq LLM (Llama-3.1-70B)
  AURA Score:   0.60 * voice + 0.40 * convo
  PDF Report:   fpdf2
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("infinite-loop")

# ─── Load .env ───────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

# ─── Groq LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
try:
    from groq import Groq
    _groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    HAS_GROQ = bool(GROQ_API_KEY)
    if HAS_GROQ:
        log.info("Groq LLM ready ✓ (Llama-3.1-70B)")
    else:
        log.warning("GROQ_API_KEY not set — using keyword NLP fallback")
except ImportError:
    _groq = None
    HAS_GROQ = False
    log.warning("groq package not installed")

# ─── Whisper STT ─────────────────────────────────────────────────────────────
try:
    import whisper as _whisper_lib
    _whisper_model = _whisper_lib.load_model("base")
    HAS_WHISPER = True
    log.info("Whisper 'base' loaded ✓")
except Exception as e:
    _whisper_model = None
    if HAS_GROQ:
        HAS_WHISPER = True
        log.info("Whisper ready ✓ (Cloud Whisper via Groq whisper-large-v3)")
    else:
        HAS_WHISPER = False
        log.warning(f"Whisper unavailable: {e}")

# ─── RawNet2 + AASIST ────────────────────────────────────────────────────────
try:
    from models.detector import run_models, _init_models
    _init_models()
    HAS_MODELS = True
    log.info("RawNet2 + AASIST ready ✓")
except Exception as e:
    HAS_MODELS = False
    log.warning(f"Model init failed: {e}")
    def run_models(audio, sr=16000):
        return {"score": 0, "rawnet2": 0, "aasist": 0, "device": "unavailable"}

# ─── PDF ─────────────────────────────────────────────────────────────────────
try:
    from fpdf import FPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    log.warning("fpdf2 not installed — PDF disabled")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Infinite Loop v3", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# SCAM KEYWORD LEXICON (fallback when Groq unavailable)
# ══════════════════════════════════════════════════════════════════════════════
SCAM_KEYWORDS: dict[str, int] = {
    "digital arrest": 30, "arrest warrant": 28, "cbi": 22, "customs officer": 20,
    "police": 15, "fir": 18, "ed officer": 20, "enforcement directorate": 22,
    "narcotics": 18, "money laundering": 20, "seized": 15, "drugs": 12,
    "wire transfer": 18, "transfer money": 20, "send money": 20, "pay now": 15,
    "urgent payment": 18, "bitcoin": 15, "crypto": 12, "gift card": 18,
    "bank account": 10, "otp": 12, "kyc": 8, "kidnapped": 25, "accident": 12,
    "hospital": 10, "ransom": 25, "dead": 14, "dying": 14, "emergency": 10,
    "help me": 12, "act now": 10, "immediately": 8, "right now": 10,
    "within minutes": 15, "or else": 12, "last chance": 12,
    "confidential": 8, "do not tell": 15, "keep secret": 12,
    "aadhar": 10, "pan card": 10, "court order": 18, "arrest": 15,
    "ied": 15, "bomb": 20, "terrorist": 20, "interpol": 18,
}


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_whisper(raw_bytes: bytes, sr: int = 16000) -> str:
    """Transcribe raw PCM int16 bytes → text using local Whisper or Groq whisper-large-v3."""
    if not HAS_WHISPER or not raw_bytes:
        return ""
    try:
        if _whisper_model is not None:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result = _whisper_model.transcribe(audio, language="en", fp16=False,
                                               condition_on_previous_text=False)
            text = result.get("text", "").strip()
        elif _groq is not None:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(raw_bytes)
            buf.seek(0)
            res = _groq.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("chunk.wav", buf.read(), "audio/wav")
            )
            text = (res.text or "").strip()
        else:
            return ""

        if len(text) < 2:
            return ""
        log.info(f"Whisper: '{text[:80]}'")
        return text
    except Exception as e:
        log.warning(f"Whisper error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# GROQ LLM CONVERSATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a cyber security AI expert specialising in voice scam detection.
Analyse the conversation transcript and return ONLY a valid JSON object with these fields:
{
  "score": <integer 0-100, how suspicious/scam-like this conversation is>,
  "risk_level": <"safe" | "suspicious" | "likely_scam" | "definite_scam">,
  "flags": [<list of specific suspicious phrases or tactics found, plain English, max 5>],
  "summary": "<one sentence plain English explanation of why this score was given>",
  "tactics": [<list of scam tactics identified, e.g. "impersonating authority", "creating urgency", max 3>]
}

Scoring guide:
  0-20:  Normal conversation, no red flags
  21-50: Some suspicious elements, proceed with caution
  51-80: Strong indicators of scam / social engineering
  81-100: Near-certain scam, dangerous content detected

Be concise. Respond ONLY with the JSON, no extra text."""


async def analyse_with_groq(transcript: str) -> dict:
    """Send transcript to Groq Llama for LLM-based scam analysis."""
    if not HAS_GROQ or not transcript or len(transcript.strip()) < 5:
        return _keyword_fallback(transcript)

    try:
        loop = asyncio.get_event_loop()

        def _call():
            return _groq.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Transcript:\n{transcript}"},
                ],
                max_tokens=300,
                temperature=0.1,
            )

        response = await loop.run_in_executor(None, _call)
        raw = response.choices[0].message.content.strip()

        # Parse JSON from LLM
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        data  = json.loads(raw[start:end])

        return {
            "score":      min(100, max(0, int(data.get("score", 0)))),
            "risk_level": data.get("risk_level", "safe"),
            "flags":      data.get("flags", []),
            "summary":    data.get("summary", ""),
            "tactics":    data.get("tactics", []),
            "source":     "groq-llama3",
        }

    except Exception as e:
        log.warning(f"Groq error: {e} — using keyword fallback")
        return _keyword_fallback(transcript)


def _keyword_fallback(transcript: str) -> dict:
    """Rule-based scam detection when Groq is unavailable."""
    text    = (transcript or "").lower()
    weight  = 0
    flags   = []
    for kw, w in SCAM_KEYWORDS.items():
        if kw in text:
            weight += w
            flags.append(kw)

    score = round(min(100, weight / 60 * 100), 0)
    if score <= 20:   risk = "safe"
    elif score <= 50: risk = "suspicious"
    elif score <= 80: risk = "likely_scam"
    else:             risk = "definite_scam"

    return {
        "score":      int(score),
        "risk_level": risk,
        "flags":      flags[:5],
        "summary":    f"Found {len(flags)} suspicious keywords" if flags else "No suspicious patterns detected",
        "tactics":    [],
        "source":     "keyword-fallback",
    }


# ══════════════════════════════════════════════════════════════════════════════
# AURA SCORE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_aura(voice_score: float, convo_score: float) -> dict:
    aura = round(0.60 * voice_score + 0.40 * convo_score, 1)
    aura = min(100.0, max(0.0, aura))

    if aura <= 30:
        tier, verdict, action, color = (
            "LOW", "Voice sounds authentic. Safe to continue.", "PERMIT", "#10b981")
    elif aura <= 60:
        tier, verdict, action, color = (
            "MEDIUM", "Some suspicious signals. Ask for verification.", "VERIFY", "#facc15")
    elif aura <= 80:
        tier, verdict, action, color = (
            "HIGH", "Strong signs of fake or cloned voice. Escalate immediately.", "ESCALATE", "#f97316")
    else:
        tier, verdict, action, color = (
            "CRITICAL", "VOICE CLONING DETECTED. End call immediately and report.", "BLOCK", "#ef4444")

    return {
        "aura_score": aura, "tier": tier, "verdict": verdict,
        "action": action, "color": color,
        "voice_score": voice_score, "convo_score": convo_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _clean_pdf_text(text: str) -> str:
    """Sanitise text for FPDF latin-1 core fonts."""
    if not text:
        return ""
    replacements = {
        "\u2022": "-", "\u2013": "-", "\u2014": "--",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u2192": "->", "\u2713": "[OK]",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Filter out anything not representable in latin-1
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(session: dict) -> Optional[bytes]:
    """Generate a full PDF forensic report from session data."""
    if not HAS_PDF:
        return None

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Header ────────────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(10, 10, 10)
        pdf.cell(0, 12, "INFINITE LOOP", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, "AI Voice Cloning Detection Report", ln=True, align="C")
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}", ln=True, align="C")
        pdf.ln(6)

        # ── AURA Score Box ────────────────────────────────────────────────────
        final_aura   = session.get("final_aura_score", 0)
        tier         = session.get("final_tier", "UNKNOWN")
        verdict      = _clean_pdf_text(session.get("final_verdict", ""))

        if final_aura <= 30:   fill = (220, 252, 231)
        elif final_aura <= 60: fill = (254, 249, 195)
        elif final_aura <= 80: fill = (255, 237, 213)
        else:                  fill = (254, 226, 226)

        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"FINAL AURA SCORE: {final_aura} / 100  |  {tier}", ln=True, fill=True, align="C")
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 8, verdict, ln=True, align="C")
        pdf.ln(4)

        # ── Score Breakdown ───────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(10, 10, 10)
        pdf.cell(0, 8, "Score Breakdown", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(245, 245, 245)

        rows = [
            ("Voice Authenticity Score (60% weight)",
             f"{session.get('avg_voice_score', 0):.1f} / 100"),
            ("Conversation Suspicion Score (40% weight)",
             f"{session.get('avg_convo_score', 0):.1f} / 100"),
            ("RawNet2 Deepfake Score (avg)",
             f"{session.get('avg_rawnet2', 0):.1f} / 100"),
            ("AASIST Deepfake Score (avg)",
             f"{session.get('avg_aasist', 0):.1f} / 100"),
            ("Total Chunks Analysed",
             str(session.get("chunk_count", 0))),
            ("Session Duration",
             f"{session.get('duration_s', 0):.1f} seconds"),
        ]
        for i, (label, val) in enumerate(rows):
            pdf.set_fill_color(245, 245, 245) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(140, 7, label, border=1, fill=True)
            pdf.cell(50, 7, val, border=1, fill=True, align="C", ln=True)
        pdf.ln(4)

        # ── LLM Analysis ─────────────────────────────────────────────────────
        llm = session.get("llm_analysis", {})
        if llm:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "AI Conversation Analysis", ln=True)
            pdf.set_font("Helvetica", "", 10)

            summary = _clean_pdf_text(llm.get("summary", "No analysis available"))
            pdf.multi_cell(0, 6, f"Summary: {summary}")
            pdf.ln(2)

            flags = llm.get("flags", [])
            if flags:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, "Suspicious phrases detected:", ln=True)
                pdf.set_font("Helvetica", "", 10)
                for f in flags:
                    pdf.cell(0, 5, _clean_pdf_text(f"  - {f}"), ln=True)

            tactics = llm.get("tactics", [])
            if tactics:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, "Scam tactics identified:", ln=True)
                pdf.set_font("Helvetica", "", 10)
                for t in tactics:
                    pdf.cell(0, 5, _clean_pdf_text(f"  - {t}"), ln=True)
            pdf.ln(4)

        # ── Score Timeline ────────────────────────────────────────────────────
        timeline = session.get("score_timeline", [])
        if timeline:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Score Timeline (per 3-second chunk)", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(25, 6, "Chunk", border=1, fill=True, align="C")
            pdf.cell(40, 6, "Voice Score", border=1, fill=True, align="C")
            pdf.cell(40, 6, "Convo Score", border=1, fill=True, align="C")
            pdf.cell(45, 6, "AURA Score", border=1, fill=True, align="C")
            pdf.cell(40, 6, "Tier", border=1, fill=True, align="C", ln=True)

            for i, entry in enumerate(timeline):
                fill = i % 2 == 0
                pdf.set_fill_color(248, 248, 248) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(25, 5, str(entry.get("chunk", i+1)), border=1, fill=True, align="C")
                pdf.cell(40, 5, str(entry.get("voice_score", 0)), border=1, fill=True, align="C")
                pdf.cell(40, 5, str(entry.get("convo_score", 0)), border=1, fill=True, align="C")
                pdf.cell(45, 5, str(entry.get("aura_score", 0)), border=1, fill=True, align="C")
                pdf.cell(40, 5, entry.get("tier", ""), border=1, fill=True, align="C", ln=True)
            pdf.ln(4)

        # ── Full Transcript ───────────────────────────────────────────────────
        transcript = _clean_pdf_text(session.get("full_transcript", "").strip())
        if transcript:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Full Conversation Transcript", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_fill_color(248, 248, 248)
            pdf.multi_cell(0, 5, transcript, border=1, fill=True)
            pdf.ln(4)

        # ── Recommendation ────────────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Recommendation", ln=True)
        pdf.set_font("Helvetica", "", 10)

        if final_aura > 80:
            pdf.set_text_color(180, 0, 0)
            pdf.multi_cell(0, 6, (
                "CRITICAL ALERT: This call shows strong evidence of AI voice cloning or scam. "
                "Do NOT share any personal information, OTP, bank details, or money. "
                "Hang up immediately and report to:\n"
                "  National Cyber Crime Helpline: 1930\n"
                "  Online: https://cybercrime.gov.in"
            ))
        elif final_aura > 60:
            pdf.set_text_color(180, 80, 0)
            pdf.multi_cell(0, 6,
                "HIGH SUSPICION: Verify caller identity through a separate channel. "
                "Do not act on requests made in this call without verification.")
        elif final_aura > 30:
            pdf.set_text_color(100, 80, 0)
            pdf.multi_cell(0, 6, "MODERATE: Some suspicious patterns. Exercise caution.")
        else:
            pdf.set_text_color(0, 120, 60)
            pdf.multi_cell(0, 6, "LOW RISK: Conversation appears authentic.")

        pdf.set_text_color(10, 10, 10)
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, "Generated by Infinite Loop -- AI-Powered Voice Cloning Detection System", ln=True, align="C")

        return bytes(pdf.output())

    except Exception as e:
        log.error(f"PDF generation error: {e}", exc_info=True)
        return None

    except Exception as e:
        log.error(f"PDF generation error: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/analyse")
async def analyse_stream(ws: WebSocket):
    await ws.accept()
    client = ws.client
    log.info(f"WS connected: {client}")

    session = {
        "chunk_count":      0,
        "voice_scores":     [],
        "convo_scores":     [],
        "rawnet2_scores":   [],
        "aasist_scores":    [],
        "full_transcript":  "",
        "llm_analysis":     {},
        "score_timeline":   [],
        "start_time":       time.time(),
        "final_aura_score": 0,
        "final_tier":       "LOW",
        "final_verdict":    "",
    }

    try:
        while True:
            msg = await ws.receive()
            msg_type = msg.get("type", "")

            # ── Disconnect ───────────────────────────────────────────────────
            if msg_type == "websocket.disconnect":
                break

            # ── Text / JSON ──────────────────────────────────────────────────
            if msg_type == "websocket.receive" and msg.get("text"):
                text = msg["text"]

                if text == "ping":
                    await ws.send_json({"type": "pong"})
                    continue

                # Transcript from browser Web Speech API
                try:
                    jmsg = json.loads(text)
                    mtype = jmsg.get("type")

                    if mtype == "transcript":
                        browser_text = jmsg.get("text", "").strip()
                        if browser_text:
                            session["full_transcript"] = browser_text
                            # Quick LLM analysis on fresh transcript
                            llm = await analyse_with_groq(browser_text)
                            session["llm_analysis"] = llm
                            c_score = llm["score"]
                            session["convo_scores"].append(c_score)
                            v = session["voice_scores"][-1] if session["voice_scores"] else 0
                            aura = compute_aura(v, c_score)
                            session["final_aura_score"] = aura["aura_score"]
                            session["final_tier"]       = aura["tier"]
                            session["final_verdict"]    = aura["verdict"]
                            await ws.send_json({
                                "type": "analysis_result",
                                "chunk": session["chunk_count"],
                                "latency_ms": 0,
                                "aura": aura,
                                "voice": {"score": v, "rawnet2": 0, "aasist": 0},
                                "conversation": {
                                    "score": c_score,
                                    "risk_level": llm.get("risk_level", "safe"),
                                    "flags": llm.get("flags", []),
                                    "summary": llm.get("summary", ""),
                                    "tactics": llm.get("tactics", []),
                                    "transcript": browser_text,
                                    "source": llm.get("source", ""),
                                },
                                "session": {
                                    "chunks": session["chunk_count"],
                                    "elapsed_s": round(time.time() - session["start_time"], 1),
                                },
                                "mode": "groq-llm",
                            })

                    elif mtype == "generate_pdf":
                        # Client requests PDF at end of call
                        session["duration_s"]        = round(time.time() - session["start_time"], 1)
                        session["avg_voice_score"]   = round(sum(session["voice_scores"]) / max(len(session["voice_scores"]), 1), 1)
                        session["avg_convo_score"]   = round(sum(session["convo_scores"]) / max(len(session["convo_scores"]), 1), 1)
                        session["avg_rawnet2"]        = round(sum(session["rawnet2_scores"]) / max(len(session["rawnet2_scores"]), 1), 1)
                        session["avg_aasist"]         = round(sum(session["aasist_scores"]) / max(len(session["aasist_scores"]), 1), 1)

                        # Final LLM pass on full transcript
                        if session["full_transcript"]:
                            final_llm = await analyse_with_groq(session["full_transcript"])
                            session["llm_analysis"] = final_llm

                        pdf_bytes = await asyncio.get_event_loop().run_in_executor(
                            None, generate_pdf_report, session)
                        if pdf_bytes:
                            b64 = base64.b64encode(pdf_bytes).decode()
                            await ws.send_json({"type": "pdf_ready", "pdf_b64": b64})
                        else:
                            await ws.send_json({"type": "pdf_error", "message": "PDF generation failed"})

                except (json.JSONDecodeError, KeyError):
                    pass
                continue

            # ── Binary PCM audio ─────────────────────────────────────────────
            if msg_type != "websocket.receive":
                continue
            raw_bytes = msg.get("bytes") or b""
            if not raw_bytes:
                continue

            session["chunk_count"] += 1
            chunk_num = session["chunk_count"]
            t0 = time.perf_counter()

            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            loop = asyncio.get_event_loop()

            # Voice model inference
            model_result = await loop.run_in_executor(None, run_models, audio_np, 16000)
            v_score = model_result["score"]

            # Whisper transcription
            if HAS_WHISPER:
                whisper_text = await loop.run_in_executor(None, transcribe_whisper, raw_bytes, 16000)
                if whisper_text:
                    session["full_transcript"] += " " + whisper_text
                    session["full_transcript"] = session["full_transcript"].strip()[-1000:]

            # Conversation analysis (LLM or keyword)
            transcript_so_far = session["full_transcript"]
            if transcript_so_far:
                llm_result = await analyse_with_groq(transcript_so_far)
            else:
                llm_result = {"score": 0, "risk_level": "safe", "flags": [], "summary": "", "tactics": [], "source": "no-transcript"}
            c_score = llm_result["score"]

            # AURA
            aura = compute_aura(v_score, c_score)

            # Track history
            session["voice_scores"].append(v_score)
            session["convo_scores"].append(c_score)
            session["rawnet2_scores"].append(model_result.get("rawnet2", v_score))
            session["aasist_scores"].append(model_result.get("aasist", v_score))
            session["llm_analysis"]     = llm_result
            session["final_aura_score"] = aura["aura_score"]
            session["final_tier"]       = aura["tier"]
            session["final_verdict"]    = aura["verdict"]
            session["score_timeline"].append({
                "chunk":       chunk_num,
                "voice_score": v_score,
                "convo_score": c_score,
                "aura_score":  aura["aura_score"],
                "tier":        aura["tier"],
            })

            elapsed = round((time.perf_counter() - t0) * 1000, 1)

            await ws.send_json({
                "type":       "analysis_result",
                "chunk":      chunk_num,
                "latency_ms": elapsed,
                "aura":       aura,
                "voice": {
                    "score":   v_score,
                    "rawnet2": model_result.get("rawnet2", v_score),
                    "aasist":  model_result.get("aasist", v_score),
                    "device":  model_result.get("device", "cpu"),
                },
                "conversation": {
                    "score":      c_score,
                    "risk_level": llm_result.get("risk_level", "safe"),
                    "flags":      llm_result.get("flags", []),
                    "summary":    llm_result.get("summary", ""),
                    "tactics":    llm_result.get("tactics", []),
                    "transcript": session["full_transcript"][-300:],
                    "source":     llm_result.get("source", ""),
                },
                "session": {
                    "chunks":    chunk_num,
                    "elapsed_s": round(time.time() - session["start_time"], 1),
                },
                "mode": f"rawnet2+aasist+{'groq' if HAS_GROQ else 'keywords'}",
            })

            log.info(f"[chunk {chunk_num}] voice={v_score} convo={c_score} AURA={aura['aura_score']} {elapsed}ms")

    except WebSocketDisconnect:
        log.info(f"WS disconnected: {client}")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            log.info(f"WS closed: {client}")
        else:
            log.error(f"WS RuntimeError: {e}", exc_info=True)
    except Exception as e:
        log.error(f"WS error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    return {
        "status":  "ok",
        "whisper": HAS_WHISPER,
        "groq":    HAS_GROQ,
        "models":  HAS_MODELS,
        "pdf":     HAS_PDF,
        "mode":    (
            "full-ai" if (HAS_GROQ and HAS_WHISPER and HAS_MODELS)
            else "partial" if (HAS_GROQ or HAS_WHISPER)
            else "heuristic-only"
        ),
        "aura_weights": {"voice": "60%", "conversation": "40%"},
    }


@app.get("/")
async def root():
    return {
        "service": "Infinite Loop v3 — Voice Cloning Detection",
        "ws":      "/ws/analyse",
        "health":  "/health",
        "docs":    "/docs",
    }
