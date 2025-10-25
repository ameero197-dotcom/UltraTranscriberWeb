# UltraTranscriberWeb.py — FastAPI (Polished & Fixed)
# -*- coding: utf-8 -*-
"""
Ultra Transcriber – Professional Web Edition (FastAPI)
- تصميم حديث مع دعم ثنائي اللغة (EN/AR)
- رسائل ETA بشرية بدل "0s"
- تكامل AdSense اختياري
- واجهة احترافية ومتجاوبة على جميع الأجهزة
- نقاط نهاية: /dashboard, /api/upload, /api/job/{id}, /healthz
"""

from __future__ import annotations

import os
import re
import json
import shutil
import tempfile
import uuid
import subprocess
import logging
import traceback
import zipfile
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from datetime import datetime

os.environ["HF_HOME"] = "/tmp/hf_cache"


# ✨ مسارات آمنة للكتابة على Hugging Face (داخل /tmp)
BASE_DIR = Path(os.getenv("APP_DATA_DIR", "/tmp/ultra"))
UPLOAD_DIR = BASE_DIR / "uploads"
TRANS_DIR = BASE_DIR / "transcripts"
DB_PATH   = BASE_DIR / "utweb_pro.sqlite3"

# أنشئ المجلدات قبل أي استخدام
BASE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)

# 🔐 اللوج إلى /tmp أيضاً
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "ultra_transcriber_pro.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)


from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

import uvicorn
import stripe as stripe_sdk

import torch
from faster_whisper import WhisperModel

# ---------------- Configuration ----------------
APP_TITLE = "Ultra Transcriber Pro"
BILLING_MODE = os.getenv("BILLING_MODE", "ads").lower()
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
ADSENSE_CLIENT = os.getenv("ADSENSE_CLIENT", "")  # فارغ = تعطيل الإعلانات
APP_SECRET = os.getenv("APP_SECRET", "ultra-secret-key-2025")
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")  # اختياري

if STRIPE_SECRET_KEY:
    stripe_sdk.api_key = STRIPE_SECRET_KEY


logging.basicConfig(
    filename="ultra_transcriber_pro.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)

# ---------------- Utilities ----------------
VALID_EXTS = (
    ".mp3", ".wav", ".mp4", ".m4a", ".mov", ".aac",
    ".flac", ".ogg", ".wma", ".mkv", ".webm",
)

def ensure_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def ffprobe_duration(path: Path) -> float:
    try:
        out = run_cmd(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path),
            ]
        )
        return float(out.strip())
    except Exception:
        return 0.0

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/*?:\"<>|]+", "_", name)

def srt_timestamp(t: float) -> str:
    if t is None:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = (seg.get("text") or "").strip()
            f.write(f"{i}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}\n\n")

# Cache model per-process
_MODEL_CACHE: dict = {}

def load_model(model_size: str) -> WhisperModel:
    key = f"{model_size}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    logging.info(f"Loading faster-whisper: size={model_size} device={device} compute={compute}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute)
    except Exception as e:
        logging.warning(f"GPU load failed ({e}), retrying on CPU/int8")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _MODEL_CACHE[key] = model
    return model

def split_media(src: Path, chunk_minutes: int, overlap_seconds: int, workdir: Path) -> List[Path]:
    dur = ffprobe_duration(src)
    def to_wav(dst_path: Path, start=None, dur_s=None):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if start is not None:
            cmd += ["-ss", str(start)]
        if dur_s is not None:
            cmd += ["-t", str(dur_s)]
        cmd += ["-i", str(src), "-ac", "1", "-ar", "16000", "-y", str(dst_path)]
        run_cmd(cmd)

    if dur == 0.0 or dur <= chunk_minutes * 60 + 5:
        out = workdir / f"000000_{sanitize_filename(src.stem)}.wav"
        to_wav(out)
        return [out]

    seg_len = chunk_minutes * 60
    chunk_paths = []
    start = 0.0
    idx = 0
    while start < dur:
        end = min(start + seg_len, dur)
        out_name = f"{idx:06d}_{sanitize_filename(src.stem)}.wav"
        out_path = workdir / out_name
        to_wav(out_path, start=start, dur_s=(end - start))
        chunk_paths.append(out_path)
        idx += 1
        start = end - overlap_seconds
        if end >= dur:
            break
    return chunk_paths

def transcribe_file(model: WhisperModel, path: Path, language: Optional[str]):
    segments, info = model.transcribe(
        str(path),
        language=(None if not language or language == "auto" else language),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    segs = []
    lines = []
    for s in segments:
        txt = s.text.strip()
        segs.append({"start": float(s.start), "end": float(s.end), "text": txt})
        if txt:
            lines.append(txt)
    text_full = "\n".join(lines)
    detected = getattr(info, "language", None) or (language if language and language != "auto" else "unknown")
    return detected, segs, text_full

def stitch_segments(chunk_segs_list, chunk_offsets):
    stitched = []
    for segs, offset in zip(chunk_segs_list, chunk_offsets):
        for s in segs:
            stitched.append({"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"]})
    return stitched

def process_many(
    files: List[Path],
    out_dir: Path,
    model_size: str = "small",
    lang_choice: str = "auto",
    chunk_minutes: int = 30,
    overlap_seconds: int = 2,
    merge_all: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[Path, Optional[Path], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="utweb_pro_"))
    model = load_model(model_size)
    merged_texts = []

    total = len(files)
    for i, f in enumerate(files, start=1):
        if progress_callback:
            progress_callback(f"Processing file {i}/{total}: {f.name}", i / total * 100)
        if f.suffix.lower() not in VALID_EXTS:
            if progress_callback:
                progress_callback(f"⚠️ Unsupported format: {f.name}", i / total * 100)
            continue
        workdir = tmp_root / ("chunks_" + uuid.uuid4().hex)
        workdir.mkdir(parents=True, exist_ok=True)
        language = None if lang_choice == "auto" else lang_choice

        chunks = split_media(f, chunk_minutes, overlap_seconds, workdir)
        chunk_segs_list, chunk_offsets = [], []
        cursor_offset = 0.0
        detected_langs = []
        for ch in chunks:
            try:
                det, segs, _ = transcribe_file(model, ch, language)
                detected_langs.append(det)
                ch_dur = ffprobe_duration(ch)
                chunk_segs_list.append(segs)
                chunk_offsets.append(cursor_offset)
                cursor_offset += max(0.0, ch_dur - overlap_seconds)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ Chunk failed {ch.name}: {e}", i / total * 100)
                logging.error(traceback.format_exc())

        stitched = stitch_segments(chunk_segs_list, chunk_offsets)
        full_text = "\n".join([s["text"] for s in stitched if s.get("text")])
        detected_lang = language or (detected_langs[0] if detected_langs else "unknown")
        safe_stem = sanitize_filename(f.stem)
        out_txt = out_dir / f"{safe_stem}_({detected_lang}).txt"
        out_srt = out_dir / f"{safe_stem}_({detected_lang}).srt"
        out_txt.write_text(full_text, encoding="utf-8")
        write_srt(stitched, out_srt)
        if merge_all:
            header = f"\n\n===== {f.name} ({detected_lang}) =====\n"
            merged_texts.append(header + full_text)

        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    merged_txt_path = None
    merged_text = ""
    if merge_all and merged_texts:
        merged_txt_path = out_dir / ("ALL_MERGED_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".txt")
        merged_text = "".join(merged_texts)
        merged_txt_path.write_text(merged_text, encoding="utf-8")

    zip_path = out_dir / ("Transcripts_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.iterdir():
            if p.is_file() and p.suffix.lower() in [".txt", ".srt"]:
                z.write(p, arcname=p.name)

    try:
        shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception:
        pass

    if progress_callback:
        progress_callback("✅ Processing complete!", 100)
    return zip_path, merged_txt_path, merged_text


# ---------------- Database ----------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            is_subscriber INTEGER DEFAULT 0,
            preferred_lang TEXT DEFAULT 'en',
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            message TEXT,
            model_size TEXT,
            lang_choice TEXT,
            chunk_minutes INTEGER,
            overlap_seconds INTEGER,
            merge_all INTEGER,
            progress INTEGER DEFAULT 0,
            out_zip TEXT,
            merged_txt TEXT,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT
        )
        """
    )
    con.commit()
    con.close()

init_db()


# ---------------- FastAPI app ----------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET)
# CORS: نسمح للجميع بدون Credentials لتفادي تعارض (* + credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/transcripts", StaticFiles(directory=str(TRANS_DIR), html=False), name="transcripts")


# ---------------- i18n & Styles ----------------
TRANSLATIONS = {
    "en": {
        "app_title": "Ultra Transcriber Pro",
        "subtitle": "AI-Powered Audio & Video Transcription",
        "dashboard": "Dashboard",
        "drag_drop": "Drag & drop files here or click to browse",
        "supported_formats": "Supported: MP3, WAV, MP4, M4A, MOV, AAC, FLAC, OGG, WMA, MKV, WEBM",
        "model_size": "AI Model Size",
        "model_tooltip": "Larger models are more accurate but slower. 'small' recommended for most uses.",
        "language": "Language",
        "language_tooltip": "Select 'auto' for automatic detection or choose a specific language",
        "chunk_size": "Chunk Size (minutes)",
        "chunk_tooltip": "Split long files into chunks. Larger chunks = better context, smaller = faster processing",
        "overlap": "Overlap (seconds)",
        "overlap_tooltip": "Audio overlap between chunks to avoid missing words at boundaries",
        "merge_transcripts": "Merge All Transcripts",
        "merge_tooltip": "Combine all transcriptions into one master file",
        "start_transcription": "Start Transcription",
        "processing": "Processing",
        "eta": "Estimated Time",
        "complete": "Complete",
        "download_zip": "Download All Files (ZIP)",
        "preview_merged": "Preview Merged Transcript",
        "error": "Error",
        "help": "Help",
        "help_text": "1) Choose model and language. 2) Drop audio/video files. 3) Click Start. We’ll process, show progress, and human-readable ETA messages.",
        "ffmpeg_missing": "FFmpeg/FFprobe not available on server. Please install FFmpeg.",
        "eta_wait": "Please wait…",
        "eta_minutes": "This may take a few minutes…",
        "eta_working": "Working on it…",
    },
    "ar": {
        "app_title": "ترانسكرايبر الترا برو",
        "subtitle": "تفريغ صوتي ومرئي بالذكاء الاصطناعي",
        "dashboard": "لوحة التحكم",
        "drag_drop": "اسحب الملفات هنا أو انقر للتصفح",
        "supported_formats": "الصيغ المدعومة: MP3, WAV, MP4, M4A, MOV, AAC, FLAC, OGG, WMA, MKV, WEBM",
        "model_size": "حجم نموذج الذكاء الاصطناعي",
        "model_tooltip": "النماذج الأكبر أدق لكنها أبطأ. ننصح بـ 'small' للاستخدام العام.",
        "language": "اللغة",
        "language_tooltip": "اختر 'auto' للكشف التلقائي أو حدد لغة بعينها",
        "chunk_size": "حجم المقطع (دقائق)",
        "chunk_tooltip": "قسّم الملفات الطويلة. الأكبر = سياق أفضل، الأصغر = أسرع.",
        "overlap": "التراكب (ثوانٍ)",
        "overlap_tooltip": "تراكب بسيط لتفادي فقدان كلمات على الحدود",
        "merge_transcripts": "دمج جميع النصوص",
        "merge_tooltip": "اجمع كل التفريغات في ملف شامل واحد",
        "start_transcription": "بدء التفريغ",
        "processing": "جارٍ المعالجة",
        "eta": "الوقت المتوقع",
        "complete": "اكتمل",
        "download_zip": "تحميل كل الملفات (ZIP)",
        "preview_merged": "معاينة النص المدمج",
        "error": "خطأ",
        "help": "مساعدة",
        "help_text": "١) اختر النموذج واللغة. ٢) أسقط ملفات الصوت/الفيديو. ٣) اضغط بدء. سنعرض التقدم ورسائل وقت تقديري بشرية.",
        "ffmpeg_missing": "برنامج FFmpeg/FFprobe غير متوفر على الخادم. يرجى تثبيت FFmpeg.",
        "eta_wait": "انتظر قليلاً…",
        "eta_minutes": "قد تستغرق العملية بضع دقائق…",
        "eta_working": "نقوم بالمعالجة…",
    },
}

MODERN_STYLE = """
<style>
:root{
  --bg:#0a0e27;
  --card:rgba(20,25,50,.95);
  --border:rgba(99,102,241,.22);
  --soft:rgba(99,102,241,.10);
  --accent1:#667eea;
  --accent2:#764ba2;
  --muted:#a1a8c3;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;background:var(--bg);color:#fff}
.container{max-width:1100px;margin:0 auto;padding:1.25rem}
@media (min-width:768px){ .container{padding:2rem} }
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:1.25rem}
.header{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1rem;flex-wrap:wrap}
.logo-section{display:flex;align-items:center;gap:1rem}
.logo-icon{width:44px;height:44px;background:linear-gradient(135deg,var(--accent1),var(--accent2));border-radius:12px;display:flex;align-items:center;justify-content:center}
.title{font-weight:800}
.subtitle{color:var(--muted);font-size:.9rem}
.lang-switch{display:flex;gap:.5rem}
.lang-btn{padding:.5rem 1rem;border-radius:8px;border:1px solid var(--border);background:var(--soft);color:#cfd3ff;cursor:pointer}
.lang-btn.active{background:#6366f1;color:#fff}
.input, select{
  width:100%;padding:.9rem 1rem;border-radius:12px;
  border:1px solid var(--border);
  background:#0f1535;      /* خلفية داكنة لحقل الإدخال والقائمة */
  color:#fff;outline:none;
}
select option{background:#0f1535;color:#fff}
.btn{padding:.9rem 1.1rem;border:0;border-radius:12px;cursor:pointer}
.btn-primary{background:linear-gradient(135deg,var(--accent1),var(--accent2));color:#fff}
.btn-secondary{background:var(--soft);color:#cfd3ff;border:1px solid var(--border)}
.btn-icon{min-width:44px}
.upload-zone{border:2px dashed rgba(99,102,241,.35);border-radius:16px;padding:2rem;text-align:center;cursor:pointer;margin-top:1rem}
.file-list{display:flex;flex-direction:column;gap:.75rem;margin-top:1rem}
.file-item{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1rem;border:1px solid var(--border);border-radius:12px;background:var(--soft)}
.progress-container{height:12px;border-radius:12px;background:rgba(99,102,241,.12);overflow:hidden}
.progress-bar{height:100%;background:linear-gradient(135deg,var(--accent1),var(--accent2));width:0%}
.status-panel{margin-top:1rem;padding:1rem;border:1px solid var(--border);border-radius:12px;background:var(--soft)}
.preview-section{margin-top:1rem;padding:1rem;border:1px solid var(--border);border-radius:12px;background:var(--soft)}
.badge{display:inline-block;padding:.25rem .5rem;border:1px solid rgba(99,102,241,.3);border-radius:8px;color:#6366f1;background:rgba(99,102,241,.15)}
.bg-animated{position:fixed;inset:0;z-index:-1;background:radial-gradient(circle at 20% 20%,rgba(99,102,241,.18),transparent 40%),radial-gradient(circle at 80% 80%,rgba(139,92,246,.18),transparent 40%)}
.error-message{color:#ef4444;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);padding:.75rem;border-radius:8px;margin-top:1rem}
.help{font-size:.9rem;color:var(--muted);margin-top:.5rem}
.help-box{margin-top:1rem;padding:1rem;border:1px dashed rgba(99,102,241,.3);border-radius:12px;color:#cfd3ff;background:rgba(99,102,241,.06)}
.tooltip{display:inline-block;margin-left:.4rem;opacity:.8}
/* شبكة الإعدادات: تباعد احترافي وتجربة متجاوبة */
.settings-grid{
  display:grid;
  grid-template-columns:repeat(12,1fr);
  gap:1rem;            /* تباعد يمنع التضارب بين Chunk و Overlap */
  margin-top:1rem;
}
.setting-item{grid-column:span 12}
@media (min-width:640px){
  .setting-item{grid-column:span 6}
}
@media (min-width:1024px){
  .setting-item{grid-column:span 3}
}
.label{display:flex;align-items:center;gap:.4rem;margin-bottom:.45rem;font-weight:600}
.help-inline{font-size:.85rem;color:var(--muted);margin-top:.35rem}
.results-header{display:flex;gap:.75rem;align-items:center}
.spinner{width:16px;height:16px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin 1s linear infinite;display:inline-block;margin-right:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
"""

# ---------------- صفحات HTML ----------------
def get_translation(lang: str, key: str) -> str:
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

def render_dashboard(user: dict, subscribed: bool, lang: str = "en") -> HTMLResponse:
    t = lambda k: get_translation(lang, k)
    dir_attr = "rtl" if lang == "ar" else "ltr"
    status_badge = "Premium" if subscribed else ("Ad-Supported" if BILLING_MODE == "ads" else "Free")

    ads_section = ""
    if ADSENSE_CLIENT:
        ads_section = (
            '<div class="card" style="margin-top: 2rem;">'
            '<div style="text-align:center;padding:1rem;color:#a1a8c3;font-size:0.875rem;margin-bottom:1rem;">📢 Advertisement</div>'
            '<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=' + ADSENSE_CLIENT + '" crossorigin="anonymous"></script>'
            '<ins class="adsbygoogle" style="display:block" data-ad-client="' + ADSENSE_CLIENT + '" data-ad-slot="1234567890" data-ad-format="auto" data-full-width-responsive="true"></ins>'
            '<script>(adsbygoogle = window.adsbygoogle || []).push({});</script>'
            "</div>"
        )

    head = (
        "<!DOCTYPE html>"
        + f'<html lang="{lang}" dir="{dir_attr}">'
        + "<head>"
        + '<meta charset="UTF-8">'
        + '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        + "<title>" + t("dashboard") + " - " + t("app_title") + "</title>"
        + MODERN_STYLE
        + "</head>"
        + "<body><div class='bg-animated'></div><div class='container'>"
        + "<div class='card'>"
        + "<div class='header'>"
        + "<div class='logo-section'>"
        + "<div class='logo-icon'>🎙️</div>"
        + "<div>"
        + "<div class='title'>" + t("app_title") + "</div>"
        + "<div class='subtitle'>Public Mode • <span class='badge'>" + status_badge + "</span></div>"
        + "</div></div>"
        + "<div class='lang-switch'>"
        + "<button class='lang-btn " + ("active" if lang=='en' else "") + "' onclick=\"window.location.search='?lang=en'\">EN</button>"
        + "<button class='lang-btn " + ("active" if lang=='ar' else "") + "' onclick=\"window.location.search='?lang=ar'\">AR</button>"
        + "</div></div>"
        + "<div class='help-box'><strong>❓ " + t("help") + ":</strong> " + t("help_text") + "</div>"
        + "<div style='border-top:1px solid rgba(99,102,241,.2);margin:1.25rem 0;'></div>"
        + "<div class='upload-zone' id='uploadZone' onclick=\"document.getElementById('fileInput').click()\">"
        + "<div class='upload-icon'>📁</div>"
        + "<div style='font-weight:600;font-size:1.125rem;margin-bottom:0.5rem;'>" + t("drag_drop") + "</div>"
        + "<div style='color:#a1a8c3;font-size:0.875rem;'>" + t("supported_formats") + "</div>"
        + "<input type='file' id='fileInput' multiple accept='audio/*,video/*' style='display:none;'>"
        + "</div>"
        + "<div id='fileList' class='file-list' style='display:none;'></div>"
        + "<div class='settings-grid'>"
        +   "<div class='setting-item'>"
        +     "<label class='label'>" + t("model_size") + "<span class='tooltip' title='" + t("model_tooltip") + "'>ⓘ</span></label>"
        +     "<select id='modelSize' class='input'>"
        +       "<option value='tiny'>Tiny (Fast)</option>"
        +       "<option value='base'>Base</option>"
        +       "<option value='small' selected>Small (Recommended)</option>"
        +       "<option value='medium'>Medium</option>"
        +       "<option value='large-v2'>Large V2</option>"
        +       "<option value='large-v3'>Large V3 (Best)</option>"
        +     "</select>"
        +   "</div>"
        +   "<div class='setting-item'>"
        +     "<label class='label'>" + t("language") + "<span class='tooltip' title='" + t("language_tooltip") + "'>ⓘ</span></label>"
        +     "<select id='language' class='input'>"
        +       "<option value='auto' selected>Auto Detect</option>"
        +       "<option value='en'>English</option>"
        +       "<option value='ar'>العربية</option>"
        +       "<option value='es'>Español</option>"
        +       "<option value='fr'>Français</option>"
        +       "<option value='de'>Deutsch</option>"
        +       "<option value='sv'>Svenska</option>"
        +       "<option value='tr'>Türkçe</option>"
        +     "</select>"
        +   "</div>"
        +   "<div class='setting-item'>"
        +     "<label class='label'>" + t("chunk_size") + "<span class='tooltip' title='" + t("chunk_tooltip") + "'>ⓘ</span></label>"
        +     "<input type='number' id='chunkSize' class='input' value='30' min='1' max='120'>"
        +     "<div class='help-inline'>"+ t("chunk_tooltip") +"</div>"
        +   "</div>"
        +   "<div class='setting-item'>"
        +     "<label class='label'>" + t("overlap") + "<span class='tooltip' title='" + t("overlap_tooltip") + "'>ⓘ</span></label>"
        +     "<input type='number' id='overlap' class='input' value='2' min='0' max='10'>"
        +     "<div class='help-inline'>"+ t("overlap_tooltip") +"</div>"
        +   "</div>"
        + "</div>"
        + "<div style='margin-top:1rem;display:flex;align-items:center;gap:.75rem;'>"
        +   "<label style='display:flex;align-items:center;gap:.5rem;cursor:pointer;'>"
        +   "<input type='checkbox' id='mergeAll' checked style='width:18px;height:18px;cursor:pointer;'>"
        +   "<span style='font-weight:600;'>" + t("merge_transcripts") + "</span>"
        +   "<span class='tooltip' title='" + t("merge_tooltip") + "'>ⓘ</span>"
        +   "</label>"
        + "</div>"
        + "<button class='btn btn-primary' id='startBtn' onclick='startTranscription()' style='margin-top:1.25rem;width:100%;justify-content:center;font-size:1.0625rem;'>"
        + "<span>🚀</span> " + t("start_transcription") + "</button>"
        + "<div id='statusPanel' style='display:none;'></div>"
        + "</div>"  # .card
        + ads_section
        + "<div style='text-align:center;margin-top:2rem;color:#a1a8c3;font-size:0.875rem;'>"
        + "© " + str(datetime.utcnow().year) + " Ultra Transcriber • "
        + "<a href='/about' style='color:#a1a8c3;'>About</a> • "
        + "<a href='/privacy' style='color:#a1a8c3;'>Privacy Policy</a> • "
        + "<a href='/contact' style='color:#a1a8c3;'>Contact</a>"
        + "</div>"
        + "</div>"  # .container
    )

    # JS script as a single triple-quoted string with a placeholder to avoid Python concatenation issues
    t_dict_json = json.dumps(TRANSLATIONS.get(lang, TRANSLATIONS["en"]))
    script = """
<script>
var selectedFiles=[];var currentJobId=null;var startTime=null;
var t=__TJSON__;
const uploadZone=document.getElementById('uploadZone');
const fileInput=document.getElementById('fileInput');
const fileList=document.getElementById('fileList');
uploadZone.addEventListener('dragover',function(e){e.preventDefault();});
uploadZone.addEventListener('drop',function(e){e.preventDefault();handleFiles(e.dataTransfer.files);});
fileInput.addEventListener('change',function(e){handleFiles(e.target.files);});
function handleFiles(files){selectedFiles=Array.from(files);displayFiles();}
function displayFiles(){
  if(selectedFiles.length===0){fileList.style.display='none';return;}
  fileList.style.display='flex';
  fileList.innerHTML=selectedFiles.map(function(f,i){
    return (
      '<div class="file-item">'+
        '<div class="file-info"><div style="font-weight:600">'+ escapeHtml(f.name) +'</div>'+
        '<div style="color:#a1a8c3;font-size:.875rem">'+ formatFileSize(f.size) +'</div></div>'+
        '<button class="btn btn-secondary btn-icon" onclick="removeFile('+i+')" title="Remove">🗑️</button>'+
      '</div>'
    );
  }).join('');
}
function removeFile(i){selectedFiles.splice(i,1);displayFiles();}
function formatFileSize(b){if(b<1024)return b+' B';if(b<1048576)return (b/1024).toFixed(1)+' KB';if(b<1073741824)return (b/1048576).toFixed(1)+' MB';return (b/1073741824).toFixed(1)+' GB';}
async function startTranscription(){
  if(selectedFiles.length===0){alert('Please select files first!');return;}
  var fd=new FormData();
  selectedFiles.forEach(function(f){fd.append('files',f);});
  fd.append('model',document.getElementById('modelSize').value);
  fd.append('lang',document.getElementById('language').value);
  fd.append('chunk',document.getElementById('chunkSize').value);
  fd.append('overlap',document.getElementById('overlap').value);
  fd.append('merge',document.getElementById('mergeAll').checked?'1':'0');
  var btn=document.getElementById('startBtn');btn.disabled=true;btn.innerHTML='<div class="spinner"></div> Uploading...';
  try{
    var r=await fetch('/api/upload',{method:'POST',body:fd});
    var j=await r.json();
    if(!r.ok){
      alert((j && (j.detail||j.error))||'Error starting transcription');
      btn.disabled=false;btn.innerHTML='<span>🚀</span> '+t.start_transcription;return;
    }
    currentJobId=j.job_id;startTime=Date.now();showStatusPanel();pollJobStatus();
  }catch(e){
    alert('Error: '+e.message);
    btn.disabled=false;btn.innerHTML='<span>🚀</span> '+t.start_transcription;
  }
}
function showStatusPanel(){
  var p=document.getElementById('statusPanel');p.style.display='block';
  p.innerHTML=
    '<div class="status-panel">'+
      '<div style="display:flex;align-items:center;justify-content:space-between;gap:.75rem;">'+
        '<div style="font-weight:700">⚙️ '+t.processing+'</div>'+
        '<div style="display:flex;align-items:center;gap:.5rem;"><div class="spinner"></div><span id="etaText">'+t.eta_wait+'</span></div>'+
      '</div>'+
      '<div class="progress-container"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>'+
      '<div id="logsContainer" style="margin-top:1rem;max-height:260px;overflow:auto;"></div>'+
    '</div>';
}
async function pollJobStatus(){
  try{
    var r=await fetch('/api/job/'+currentJobId);var d=await r.json();
    updateProgress(d);
    if(d.status==='done'){showResults(d);}
    else if(d.status==='error'){showError(d);}
    else{setTimeout(pollJobStatus,1500);}
  }catch(e){
    console.error('Polling error:',e);setTimeout(pollJobStatus,2500);
  }
}
function updateProgress(d){
  var pr=d.progress||0;
  var bar=document.getElementById('progressBar');if(bar){bar.style.width=pr+'%';}
  var eta=document.getElementById('etaText');
  if(eta){
    if(pr<10){eta.textContent=t.eta_wait;}
    else if(pr<60){eta.textContent=t.eta_minutes;}
    else{eta.textContent=t.eta_working;}
  }
  if(d.logs){
    var lc=document.getElementById('logsContainer');if(lc){
      var lines=String(d.logs||'').split('\\n');
      lc.innerHTML=lines.map(function(line){
        var c='log-line';
        if(line.indexOf('✅')>-1)c+=' success';
        else if(line.indexOf('❌')>-1)c+=' error';
        else if(line.indexOf('⚠️')>-1)c+=' warning';
        return '<div class="'+c+'" style="padding:.25rem 0;">'+escapeHtml(line)+'</div>';
      }).join('');
      lc.scrollTop=lc.scrollHeight;
    }
  }
}
function showResults(d){
  var p=document.getElementById('statusPanel');
  var zipPart=d.zip_url?('<a href="'+d.zip_url+'" download class="btn btn-primary" style="flex:1;"><span>📦</span> '+t.download_zip+'</a>'):'';
  var mergedPart=d.merged_txt?(
    '<div class="preview-section">'+
      '<div style="font-weight:600;display:flex;align-items:center;gap:.5rem;cursor:pointer;" onclick="togglePreview()"><span id="previewIcon">▶️</span> '+t.preview_merged+'</div>'+
      '<div id="previewContent" style="display:none;white-space:pre-wrap;">'+escapeHtml(d.merged_txt.substring(0,10000))+(d.merged_txt.length>10000?'\\n\\n... (truncated)':'')+'</div>'+
    '</div>'
  ):'';
  p.innerHTML=
    '<div class="results-section">'+
      '<div class="results-header"><div style="width:44px;height:44px;border-radius:12px;background:rgba(34,197,94,.2);display:flex;align-items:center;justify-content:center;">✅</div><div><div class="title">'+t.complete+'!</div><div class="subtitle">Your transcription is ready</div></div></div>'+
      '<div style="display:flex;gap:.75rem;margin-top:1rem;">'+zipPart+'</div>'+
      mergedPart+
    '</div>';
  selectedFiles=[];displayFiles();
}
function showError(d){
  var p=document.getElementById('statusPanel');
  var msg=(d && (d.message||d.logs))||'Unknown error';
  p.innerHTML=
    '<div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);padding:1rem;border-radius:12px;">'+
      '<div style="display:flex;gap:.75rem;align-items:center;"><div style="width:44px;height:44px;border-radius:12px;background:rgba(239,68,68,0.2);display:flex;align-items:center;justify-content:center;">❌</div><div><div class="title">'+t.error+'</div><div class="subtitle">Something went wrong</div></div></div>'+
      '<div style="margin-top:1rem;white-space:pre-wrap;">'+escapeHtml(String(msg)).slice(-4000)+'</div>'+
      '<button class="btn btn-primary" onclick="location.reload()" style="margin-top:1rem;">🔄 Try Again</button>'+
    '</div>';
}
function togglePreview(){
  var c=document.getElementById('previewContent');var i=document.getElementById('previewIcon');
  if(c&&i){if(c.style.display==='none'){c.style.display='block';i.textContent='▼';}else{c.style.display='none';i.textContent='▶️';}}
}
function escapeHtml(t){var d=document.createElement('div');d.textContent=t;return d.innerHTML;}
</script>
"""
    script = script.replace("__TJSON__", t_dict_json)

    tail = "</body></html>"
    html = head + script + tail
    return HTMLResponse(html)

# ---------------- Reusable Page Renderer (same look as dashboard) ----------------
def render_simple_page(page_title: str, body_html: str, lang: str = "en") -> HTMLResponse:
    dir_attr = "rtl" if lang == "ar" else "ltr"

    # قسم الإعلانات نفسه المستخدم في الـ dashboard (يظهر فقط إذا ADSENSE_CLIENT موجود)
    ads_section = ""
    if ADSENSE_CLIENT:
        ads_section = (
            '<div class="card" style="margin-top: 2rem;">'
            '<div style="text-align:center;padding:1rem;color:#a1a8c3;font-size:0.875rem;margin-bottom:1rem;">📢 Advertisement</div>'
            '<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=' + ADSENSE_CLIENT + '" crossorigin="anonymous"></script>'
            '<ins class="adsbygoogle" style="display:block" data-ad-client="' + ADSENSE_CLIENT + '" data-ad-slot="1234567890" data-ad-format="auto" data-full-width-responsive="true"></ins>'
            '<script>(adsbygoogle = window.adsbygoogle || []).push({});</script>'
            "</div>"
        )

    # رأس الصفحة + نفس MODERN_STYLE + نفس الحاوية/الكارد
    head = (
        "<!DOCTYPE html>"
        + f'<html lang="{lang}" dir="{dir_attr}">'
        + "<head>"
        + '<meta charset="UTF-8">'
        + '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        + f"<title>{page_title} - {get_translation(lang,'app_title')}</title>"
        + MODERN_STYLE
        + "</head>"
        + "<body><div class='bg-animated'></div><div class='container'>"
        + "<div class='card'>"
        + "<div class='header'>"
        + "<div class='logo-section'>"
        + "<div class='logo-icon'>🎙️</div>"
        + "<div>"
        + f"<div class='title'>{get_translation(lang,'app_title')}</div>"
        + "<div class='subtitle'>"
        + f"{get_translation(lang,'dashboard')} • "
        + "<a href='/dashboard' style='color:#cfd3ff;text-decoration:underline;'>Home</a>"
        + "</div>"
        + "</div></div>"
        + "<div class='lang-switch'>"
        + f"<button class='lang-btn {'active' if lang=='en' else ''}' onclick=\"window.location.search='?lang=en'\">EN</button>"
        + f"<button class='lang-btn {'active' if lang=='ar' else ''}' onclick=\"window.location.search='?lang=ar'\">AR</button>"
        + "</div></div>"
        + f"<h1 style='margin:0 0 .5rem 0;'>{page_title}</h1>"
        + "<div style='border-top:1px solid rgba(99,102,241,.2);margin:1.25rem 0;'></div>"
        + "<div style='line-height:1.75;color:#e8eaff'>"  # محتوى الصفحة
        + body_html
        + "</div>"
        + "</div>"  # .card
        + ads_section
        + "<div style='text-align:center;margin-top:2rem;color:#a1a8c3;font-size:0.875rem;'>"
        + "© " + str(datetime.utcnow().year) + " Ultra Transcriber • "
        + "<a href='/about' style='color:#a1a8c3;'>About</a> • "
        + "<a href='/privacy' style='color:#a1a8c3;'>Privacy Policy</a> • "
        + "<a href='/contact' style='color:#a1a8c3;'>Contact</a>"
        + "</div>"
        + "</div>"  # .container
        + "</body></html>"
    )
    return HTMLResponse(head)


# ---------------- Info Pages (use the same design) ----------------
@app.get("/about", response_class=HTMLResponse)
def about(lang: Optional[str] = "en"):
    body = (
        "<p><strong>Ultra Transcriber Pro</strong> is a transcription tool built for high accuracy and speed. "
        "It supports multiple languages and is designed to handle long audio/video files efficiently.</p>"
        "<p>Created by TechSolver • 2025</p>"
    )
    return render_simple_page("About Ultra Transcriber", body, lang or "en")


@app.get("/privacy", response_class=HTMLResponse)
def privacy(lang: Optional[str] = "en"):
    body = (
        "<h2>Data Handling</h2>"
        "<ul>"
        "<li>Uploaded files are processed temporarily under <code>/tmp</code> and removed after transcription.</li>"
        "<li>No permanent storage or third-party sharing.</li>"
        "<li>HTTPS is required; do not upload sensitive content.</li>"
        "</ul>"
        "<h2>Contact</h2>"
        "<p>Email: <a href='mailto:contact@ultratranscriber.com'>contact@ultratranscriber.com</a></p>"
    )
    return render_simple_page("Privacy Policy", body, lang or "en")


@app.get("/contact", response_class=HTMLResponse)
def contact(lang: Optional[str] = "en"):
    body = (
        "<p>We’d love to hear from you. For support, feedback, or business inquiries:</p>"
        "<p><a href='mailto:contact@ultratranscriber.com'>contact@ultratranscriber.com</a></p>"
        "<p>Response time: within 24 hours.</p>"
    )
    return render_simple_page("Contact", body, lang or "en")



# ---------------- Jobs helpers ----------------
def job_update(
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    out_zip: Optional[str] = None,
    merged_txt_path: Optional[Path] = None
):
    con = db()
    cur = con.cursor()
    row = cur.execute("SELECT message FROM jobs WHERE id=?", (job_id,)).fetchone()
    prev_msg = row["message"] if row and row["message"] else ""
    new_msg = prev_msg
    if message:
        new_msg = (prev_msg + ("\n" if prev_msg else "") + message)[:200000]
    sets = []
    params = []
    if status is not None:
        sets.append("status=?")
        params.append(status)
    if progress is not None:
        sets.append("progress=?")
        params.append(int(progress))
    if message is not None:
        sets.append("message=?")
        params.append(new_msg)
    if out_zip is not None:
        sets.append("out_zip=?")
        params.append(out_zip)
    if merged_txt_path is not None:
        merged_txt_content = ""
        try:
            merged_txt_content = merged_txt_path.read_text(encoding="utf-8")
        except Exception:
            merged_txt_content = ""
        sets.append("merged_txt=?")
        params.append(merged_txt_content)
    if not sets:
        con.close()
        return
    params.append(job_id)
    cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", params)
    con.commit()
    con.close()

def background_transcribe(
    job_id: str,
    file_paths: List[Path],
    model_size: str,
    lang_choice: str,
    chunk_minutes: int,
    overlap_seconds: int,
    merge_all: bool,
):
    out_dir = TRANS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def cb(msg: str, prog: float):
        job_update(job_id, progress=int(prog), message=msg)

    try:
        job_update(job_id, status="running", message="Starting processing...")
        zip_path, merged_txt_path, _ = process_many(
            file_paths,
            out_dir,
            model_size=model_size,
            lang_choice=lang_choice,
            chunk_minutes=chunk_minutes,
            overlap_seconds=overlap_seconds,
            merge_all=merge_all,
            progress_callback=cb,
        )
        # رابط نسبي يعمل خلف أي بروكسي/دومين
        rel_zip = "/" + str(zip_path.relative_to(TRANS_DIR)).replace("\\", "/")
        zip_url = f"/transcripts{rel_zip}"
        job_update(
            job_id,
            status="done",
            progress=100,
            message="✅ All done",
            out_zip=zip_url,
            merged_txt_path=merged_txt_path,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        job_update(job_id, status="error", message=f"❌ Error: {e}\n{tb}")


# ---------------- Routes ----------------
@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/dashboard")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/dashboard")
async def get_dashboard(request: Request, lang: Optional[str] = "en"):
    user = {"email": "guest@ultratranscriber.com"}
    subscribed = False
    return render_dashboard(user, subscribed, lang or "en")

@app.post("/api/upload")
async def api_upload(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = Form("small"),
    lang: str = Form("auto"),
    chunk: int = Form(30),
    overlap: int = Form(2),
    merge: str = Form("1"),
):
    try:
        if not ensure_ffmpeg():
            return JSONResponse(status_code=400, content={"detail": TRANSLATIONS["en"]["ffmpeg_missing"]})

        job_id = uuid.uuid4().hex
        merge_all = merge == "1"

        con = db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO jobs(id,status,message,model_size,lang_choice,chunk_minutes,overlap_seconds,merge_all,progress,created_at,started_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id, "queued", "", model, lang, int(chunk), int(overlap),
                1 if merge_all else 0, 0, datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
            ),
        )
        con.commit()
        con.close()

        job_upload_dir = UPLOAD_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        for uf in files:
            filename = sanitize_filename(uf.filename or f"upload_{uuid.uuid4().hex}")
            suffix = Path(filename).suffix.lower()
            if suffix not in VALID_EXTS:
                logging.warning(f"Unsupported format skipped: {filename}")
                continue
            dest = job_upload_dir / filename
            with dest.open("wb") as out:
                while True:
                    chunk_bytes = await uf.read(1024 * 1024)
                    if not chunk_bytes:
                        break
                    out.write(chunk_bytes)
            saved_paths.append(dest)

        if not saved_paths:
            return JSONResponse(status_code=400, content={"detail": "No valid files uploaded"})

        background.add_task(
            background_transcribe,
            job_id,
            saved_paths,
            model,
            lang,
            int(chunk),
            int(overlap),
            merge_all,
        )
        return {"job_id": job_id}
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return JSONResponse(status_code=500, content={"detail": f"{e}", "trace": tb})

@app.get("/api/job/{job_id}")
async def api_job(job_id: str):
    con = db()
    cur = con.cursor()
    row = cur.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    con.close()
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Job not found"})

    msg = row["message"] or ""
    if not msg:
        try:
            log_txt = Path("ultra_transcriber_pro.log").read_text(encoding="utf-8")
            msg = log_txt[-4000:]
        except Exception:
            msg = ""

    return {
        "id": row["id"],
        "status": row["status"],
        "progress": row["progress"],
        "logs": msg,
        "message": msg,
        "zip_url": row["out_zip"],
        "merged_txt": row["merged_txt"] or "",
    }


# ---------------- Static Info Pages ----------------
@app.get("/about", response_class=HTMLResponse)
def about():
    html = """
    <h1>About Ultra Transcriber</h1>
    <p>Ultra Transcriber Pro is an AI-powered transcription tool built for high accuracy and speed.
    It supports multiple languages and is designed to handle long audio or video files efficiently.</p>
    <p>Created by TechSolver • 2025</p>
    """
    return HTMLResponse(html)

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    html = """
    <h1>Privacy Policy</h1>
    <p>We respect your privacy. Uploaded files are processed temporarily and deleted automatically
    after transcription. No data is shared with third parties or stored permanently.</p>
    <p>By using this service, you agree to our terms of use.</p>
    """
    return HTMLResponse(html)

@app.get("/contact", response_class=HTMLResponse)
def contact():
    html = """
    <h1>Contact Us</h1>
    <p>For inquiries, feedback, or support, please contact us at:
    <a href='mailto:contact@ultratranscriber.com'>contact@ultratranscriber.com</a></p>
    """
    return HTMLResponse(html)



# ---------------- Local run ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
