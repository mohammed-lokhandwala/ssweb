import os
import cv2
import numpy as np
import base64
import json
import shutil
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, abort

app = Flask(__name__)

# ── ML endpoint (HuggingFace Space) ──────────────────────────────────────────
ML_API = "https://mohammedlokhandwala-ssweb.hf.space/extract"

# ── Storage ───────────────────────────────────────────────────────────────────
DB_PATH   = Path("./qdrant_db")
LOCK_FILE = DB_PATH / ".lock"
BACKUP    = Path("./faces_backup.json")


_db = None


# ═══════════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _bust_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except Exception:
            shutil.rmtree(DB_PATH, ignore_errors=True)


def get_db():
    global _db
    if _db is None:
        _bust_lock()
        try:
            from qdrant_client import QdrantClient
            _db = QdrantClient(path=str(DB_PATH))
        except Exception:
            from qdrant_client import QdrantClient
            _db = QdrantClient(":memory:")
            _restore_backup(_db)
    return _db


def _restore_backup(client):
    if not BACKUP.exists():
        return
    try:
        from qdrant_client.models import PointStruct, VectorParams, Distance
        data = json.loads(BACKUP.read_text())
        if not data:
            return
        if not client.collection_exists("faces"):
            client.create_collection(
                "faces",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
        pts = [
            PointStruct(id=r["id"], vector=r["vector"], payload=r["payload"])
            for r in data
        ]
        client.upsert("faces", points=pts)
    except Exception:
        pass


def _save_backup(client):
    try:
        pts, _ = client.scroll(
            "faces", limit=10000, with_payload=True, with_vectors=True
        )
        data = [
            {"id": p.id, "vector": p.vector, "payload": p.payload}
            for p in pts
        ]
        BACKUP.write_text(json.dumps(data))
    except Exception:
        pass


def ensure_collection():
    from qdrant_client.models import VectorParams, Distance
    client = get_db()
    if not client.collection_exists("faces"):
        client.create_collection(
            "faces",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )


ensure_collection()


# ═══════════════════════════════════════════════════════════════════════════════
#  ML HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def get_embeddings_from_ml(img):
    _, buffer = cv2.imencode(".jpg", img)
    files = {"file": ("image.jpg", buffer.tobytes(), "image/jpeg")}
    try:
        res = requests.post(ML_API, files=files, timeout=30)
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def require_admin():
    key = request.args.get("key") or request.json.get("key", "") if request.is_json else request.args.get("key", "")
    if key != ADMIN_KEY:
        abort(403)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML — USER FACING (Search + Gallery)
# ═══════════════════════════════════════════════════════════════════════════════

USER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SS WebCreation — Face Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@200;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:#080a0f; --surface:#0d1117; --card:#111820; --border:#1e2a38;
    --accent:#00ffe5; --accent2:#ff3c6e; --accent3:#7b61ff;
    --text:#e8edf3; --muted:#4a5a6a; --success:#00e87a; --warn:#ffba00;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,229,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,229,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.15) 2px,rgba(0,0,0,.15) 4px);pointer-events:none;z-index:0}
  .container{max-width:1100px;margin:0 auto;padding:0 24px;position:relative;z-index:1}
  header{border-bottom:1px solid var(--border);padding:20px 0;position:sticky;top:0;background:rgba(8,10,15,.88);backdrop-filter:blur(14px);z-index:100}
  .header-inner{display:flex;align-items:center;justify-content:space-between}
  .brand{display:flex;flex-direction:column}
  .brand-name{font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.14em;color:var(--accent);text-shadow:0 0 20px rgba(0,255,229,.5),0 0 60px rgba(0,255,229,.2);line-height:1}
  .brand-name span{color:var(--accent2)}
  .brand-sub{font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.22em;color:var(--muted);text-transform:uppercase;margin-top:3px}
  .status-pill{font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.1em;padding:4px 12px;border-radius:20px;border:1px solid var(--accent);color:var(--accent);display:flex;align-items:center;gap:6px}
  .status-pill .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 6px var(--accent);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  .tabs{display:flex;margin:40px 0 0;border-bottom:1px solid var(--border)}
  .tab{font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;padding:14px 28px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .2s;user-select:none}
  .tab:hover{color:var(--text)}
  .tab.active{color:var(--accent);border-bottom-color:var(--accent)}
  .tab .badge{background:var(--accent2);color:#fff;border-radius:4px;font-size:.55rem;padding:1px 5px;margin-left:6px;vertical-align:middle}
  .panel{display:none;padding:40px 0}
  .panel.active{display:block;animation:fadeUp .35s ease}
  @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}

  /* ── Drop zone ── */
  .drop-zone{border:1px dashed var(--border);border-radius:4px;padding:60px 40px;text-align:center;cursor:pointer;transition:all .25s;position:relative;overflow:hidden;background:var(--card)}
  .drop-zone::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 50% 0%,rgba(0,255,229,.05) 0%,transparent 70%);opacity:0;transition:opacity .3s}
  .drop-zone:hover,.drop-zone.dragover{border-color:var(--accent);background:rgba(0,255,229,.03)}
  .drop-zone:hover::before,.drop-zone.dragover::before{opacity:1}
  .drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .drop-icon{font-size:2.2rem;margin-bottom:14px;opacity:.4;display:block}
  .drop-label{font-family:'DM Mono',monospace;font-size:.75rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase}
  .drop-label strong{color:var(--accent);font-weight:500}

  /* ── Preview ── */
  .preview-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-top:20px}
  .preview-item{position:relative;border-radius:3px;overflow:hidden;border:1px solid var(--border);aspect-ratio:1;background:var(--card)}
  .preview-item img{width:100%;height:100%;object-fit:cover;display:block}
  .preview-item .remove{position:absolute;top:4px;right:4px;width:20px;height:20px;background:rgba(255,60,110,.9);border:none;border-radius:2px;color:#fff;font-size:.7rem;cursor:pointer;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .2s}
  .preview-item:hover .remove{opacity:1}

  /* ── Buttons ── */
  .btn{display:inline-flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;padding:12px 28px;border-radius:3px;border:1px solid;cursor:pointer;transition:all .2s;position:relative;overflow:hidden;background:transparent}
  .btn::after{content:'';position:absolute;inset:0;background:currentColor;opacity:0;transition:opacity .2s}
  .btn:hover::after{opacity:.08}
  .btn-primary{border-color:var(--accent);color:var(--accent)}
  .btn-primary:hover{box-shadow:0 0 20px rgba(0,255,229,.25)}
  .btn-ghost{border-color:var(--border);color:var(--muted)}
  .btn:disabled{opacity:.35;cursor:not-allowed}

  /* ── Results ── */
  .result-card{border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-bottom:16px;background:var(--card);transition:border-color .2s}
  .result-card:hover{border-color:var(--accent3)}
  .result-header{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--border);background:rgba(0,0,0,.2)}
  .result-query-img{display:flex;align-items:center;gap:12px}
  .query-thumb{width:44px;height:44px;border-radius:3px;object-fit:cover;border:1px solid var(--border)}
  .result-file{font-family:'DM Mono',monospace;font-size:.75rem;color:var(--text);letter-spacing:.04em}
  .result-meta{font-size:.65rem;color:var(--muted);margin-top:2px}
  .no-match{display:inline-flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:20px;padding:4px 12px}
  .match-list{padding:16px 20px;display:flex;flex-direction:column;gap:10px}
  .match-row{display:flex;align-items:center;gap:16px}
  .match-img-wrap{position:relative;flex-shrink:0}
  .match-thumb{width:52px;height:52px;border-radius:3px;object-fit:cover;border:1px solid var(--border)}
  .match-score-badge{position:absolute;bottom:-6px;left:50%;transform:translateX(-50%);font-family:'DM Mono',monospace;font-size:.55rem;white-space:nowrap;padding:2px 6px;border-radius:20px;border:1px solid;background:var(--bg)}
  .match-info{flex:1}
  .match-file{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text)}
  .match-bar-wrap{margin-top:6px;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
  .match-bar{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.16,1,.3,1)}
  .score-high{color:var(--success);border-color:var(--success)} .score-high .match-bar{background:var(--success)}
  .score-med{color:var(--warn);border-color:var(--warn)}   .score-med .match-bar{background:var(--warn)}
  .score-low{color:var(--accent2);border-color:var(--accent2)} .score-low .match-bar{background:var(--accent2)}

  /* ── Slider ── */
  .slider-row{display:flex;align-items:center;gap:14px;margin-bottom:24px}
  .slider-label{font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;white-space:nowrap}
  input[type=range]{flex:1;-webkit-appearance:none;height:2px;background:var(--border);border-radius:2px;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);cursor:pointer}
  .slider-val{font-family:'Bebas Neue',sans-serif;font-size:1.1rem;color:var(--accent);min-width:40px;text-align:right}

  /* ── Gallery ── */
  .gallery-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-top:24px}
  .gallery-item{position:relative;border-radius:4px;overflow:hidden;border:2px solid var(--border);aspect-ratio:1;background:var(--card);cursor:pointer;transition:all .2s}
  .gallery-item:hover{border-color:var(--accent);transform:scale(1.02);box-shadow:0 0 16px rgba(0,255,229,.2)}
  .gallery-item.selected{border-color:var(--accent);box-shadow:0 0 20px rgba(0,255,229,.35)}
  .gallery-item img{width:100%;height:100%;object-fit:cover;display:block}
  .gallery-item .g-label{position:absolute;bottom:0;left:0;right:0;background:rgba(8,10,15,.85);font-family:'DM Mono',monospace;font-size:.55rem;color:var(--muted);padding:5px 8px;letter-spacing:.06em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .gallery-item .g-check{position:absolute;top:6px;right:6px;width:18px;height:18px;border-radius:50%;background:var(--accent);display:none;align-items:center;justify-content:center;font-size:.6rem;color:#000;font-weight:700}
  .gallery-item.selected .g-check{display:flex}
  .gallery-empty{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted);text-align:center;padding:60px 0}
  .gallery-tip{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);margin-bottom:20px;letter-spacing:.06em}
  .gallery-tip strong{color:var(--accent)}

  /* ── Stats bar ── */
  .stats-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:4px;margin-bottom:32px;overflow:hidden}
  .stat{background:var(--card);padding:20px 24px}
  .stat-val{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;line-height:1;color:var(--accent);text-shadow:0 0 20px rgba(0,255,229,.3)}
  .stat-key{font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-top:4px}

  /* ── Misc ── */
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .mt-md{margin-top:28px}
  .section-title{font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:10px}
  .section-title::after{content:'';flex:1;height:1px;background:var(--border)}
  .progress-wrap{height:2px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:16px;display:none}
  .progress-wrap.show{display:block}
  .progress-bar{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent3));width:0%;transition:width .3s;border-radius:2px}
  .progress-bar.indeterminate{animation:indeterminate 1.2s infinite ease-in-out;width:40%!important}
  @keyframes indeterminate{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}
  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
  .search-mode-toggle{display:flex;gap:0;margin-bottom:28px;border:1px solid var(--border);border-radius:4px;overflow:hidden}
  .mode-btn{flex:1;font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;padding:12px;cursor:pointer;color:var(--muted);background:var(--card);border:none;transition:all .2s}
  .mode-btn.active{background:rgba(0,255,229,.08);color:var(--accent)}
  .mode-section{display:none}
  .mode-section.active{display:block}

  /* ── Project info banner ── */
  .project-banner{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:28px 32px;margin:32px 0 0;position:relative;overflow:hidden}
  .project-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent3),var(--accent2))}
  .pb-top{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;flex-wrap:wrap}
  .pb-title{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;letter-spacing:.1em;color:var(--text);line-height:1;margin-bottom:6px}
  .pb-title span{color:var(--accent)}
  .pb-meta{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
  .pb-chip{font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.1em;padding:3px 10px;border-radius:20px;border:1px solid var(--border);color:var(--muted)}
  .pb-chip a{color:var(--accent);text-decoration:none}
  .pb-chip a:hover{text-decoration:underline}
  .pb-desc{font-size:.82rem;color:var(--muted);line-height:1.7;margin-top:16px;max-width:780px}
  .pb-desc strong{color:var(--text)}
  .pb-reqs{margin-top:20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
  .pb-req{background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:4px;padding:12px 16px;display:flex;gap:10px;align-items:flex-start}
  .pb-req-icon{color:var(--accent);font-size:.85rem;margin-top:1px;flex-shrink:0}
  .pb-req-text{font-family:'DM Mono',monospace;font-size:.62rem;color:var(--muted);line-height:1.6}
  .pb-req-text strong{color:var(--text);display:block;margin-bottom:2px}

  /* ── Random demo ── */
  .demo-box{background:linear-gradient(135deg,rgba(0,255,229,.04),rgba(123,97,255,.04));border:1px solid var(--border);border-radius:6px;padding:20px 24px;margin-top:20px;display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
  .demo-left{flex:1}
  .demo-title{font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:4px}
  .demo-desc{font-size:.8rem;color:var(--muted);line-height:1.5}
  .btn-demo{display:inline-flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;padding:12px 24px;border-radius:3px;border:1px solid var(--accent3);color:var(--accent3);background:rgba(123,97,255,.08);cursor:pointer;transition:all .2s;white-space:nowrap}
  .btn-demo:hover{box-shadow:0 0 20px rgba(123,97,255,.25);background:rgba(123,97,255,.14)}
  .btn-demo:disabled{opacity:.4;cursor:not-allowed}
  .demo-result-preview{display:none;margin-top:14px;padding:14px 16px;background:rgba(0,0,0,.3);border-radius:4px;border:1px solid var(--border);font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);display:flex;align-items:center;gap:12px}
  .demo-result-preview img{width:48px;height:48px;border-radius:3px;object-fit:cover;border:1px solid var(--border)}
</style>
</head>
<body>
<header>
  <div class="container header-inner">
    <div class="brand">
      <div class="brand-name">SS <span>WebCreation</span></div>
      <div class="brand-sub">Face Recognition System</div>
    </div>
    <div class="status-pill"><span class="dot"></span>SYSTEM ONLINE</div>
  </div>
</header>
<div class="container">

  <!-- ══ PROJECT INFO BANNER ══ -->
  <div class="project-banner">
    <div class="pb-top">
      <div>
        <div class="pb-title">Wedding Face <span>Recognition</span> System</div>
        <div class="pb-meta">
          <span class="pb-chip">Built by Mohammed Lokhandwala</span>
          <span class="pb-chip"><a href="mailto:mohammedlokhand2021@gmail.com">mohammedlokhand2021@gmail.com</a></span>
          <span class="pb-chip">InsightFace · Qdrant · Flask</span>
          <span class="pb-chip">512-dim Cosine Similarity</span>
        </div>
      </div>
    </div>
    <p class="pb-desc">
      We are given a pool of <strong>1,000 wedding photos</strong>. The goal is to build a system that, when provided with any single photo, can identify and retrieve all other photos in which the same person appears.
      Each face in the dataset is processed using <strong>InsightFace</strong> to generate a unique numerical representation (embedding). These embeddings are stored in a <strong>Qdrant vector database</strong>.
      For any query image, the system generates its embedding and performs a similarity search against the stored vectors to find the closest matches.
      The solution works without manual tagging or labeling — relying entirely on facial features and vector similarity.
    </p>
  </div>

  <!-- ══ RANDOM DEMO ══ -->
  <div class="demo-box">
    <div class="demo-left">
      <div class="demo-title">Try a Quick Demo</div>
      <div class="demo-desc">Not sure where to start? Hit the button — the system picks a random photo from the database and searches for all matching faces instantly.</div>
      <div id="demo-picked" style="display:none;align-items:center;gap:12px;padding:12px 14px;background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:4px;margin-top:14px">
        <img id="demo-picked-img" src="" style="width:52px;height:52px;border-radius:4px;object-fit:cover;border:1px solid var(--border)"/>
        <div>
          <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:var(--muted);margin-bottom:2px">RANDOMLY PICKED</div>
          <div id="demo-picked-name" style="font-family:'DM Mono',monospace;font-size:.68rem;color:var(--accent)"></div>
        </div>
      </div>
    </div>
    <button class="btn-demo" onclick="runRandomDemo()" id="btn-demo">Pick Random and Search</button>
  </div>

  <div class="tabs" style="margin-top:28px">
    <div class="tab active" onclick="switchTab('search')">SEARCH</div>
    <div class="tab" onclick="switchTab('gallery')">GALLERY <span class="badge" id="gallery-badge">—</span></div>
  </div>

  <!-- ══ SEARCH PANEL ══ -->
  <div class="panel active" id="panel-search">
    <p class="section-title">Find yourself in the wedding photos</p>

    <div class="search-mode-toggle">
      <button class="mode-btn active" onclick="setMode('gallery')">Search from Gallery</button>
      <button class="mode-btn" onclick="setMode('upload')">Upload a Photo</button>
    </div>

    <!-- Gallery pick mode — now default/active -->
    <div class="mode-section active" id="mode-gallery">
      <p class="gallery-tip">Click any photo below to use it as your search query. <strong>Selected photos will be highlighted.</strong></p>
      <div class="gallery-grid" id="pick-grid"><div class="gallery-empty">Loading gallery…</div></div>
    </div>

    <!-- Upload mode -->
    <div class="mode-section" id="mode-upload">
      <div class="drop-zone" id="search-drop">
        <input type="file" id="search-input" accept="image/*" multiple onchange="previewFiles(this,'search-preview')" onclick="event.stopPropagation()"/>
        <span class="drop-icon">◎</span>
        <div class="drop-label">Drop your photo here or <strong>click to browse</strong></div>
        <div class="drop-label" style="margin-top:6px;font-size:.62rem;">PNG · JPG · WEBP</div>
      </div>
      <div class="preview-grid" id="search-preview"></div>
    </div>

    <div class="mt-md row">
      <div class="slider-row" style="flex:1;margin:0">
        <span class="slider-label">Min Similarity</span>
        <input type="range" min="0" max="100" value="50" id="threshold-slider" oninput="document.getElementById('threshold-val').textContent=this.value+'%'"/>
        <span class="slider-val" id="threshold-val">50%</span>
      </div>
    </div>
    <div class="mt-md row">
      <button class="btn btn-primary" onclick="runSearch()" id="btn-search"><span>◎</span> SEARCH FACES</button>
    </div>
    <div class="progress-wrap" id="search-progress"><div class="progress-bar indeterminate"></div></div>
    <div id="search-results" class="mt-md"></div>
  </div>

  <!-- ══ GALLERY PANEL ══ -->
  <div class="panel" id="panel-gallery">
    <p class="section-title">All indexed wedding photos</p>
    <div class="stats-bar">
      <div class="stat"><div class="stat-val" id="stat-total">—</div><div class="stat-key">Faces Indexed</div></div>
      <div class="stat"><div class="stat-val" id="stat-photos">—</div><div class="stat-key">Photos in DB</div></div>
      <div class="stat"><div class="stat-val">COS</div><div class="stat-key">Distance Metric</div></div>
    </div>
    <div class="gallery-grid" id="gallery-grid"><div class="gallery-empty">Loading…</div></div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let searchFiles   = [];
let pickedFile    = null;
let currentMode   = 'gallery';
let galleryLoaded = false;

// ── Random Demo ───────────────────────────────────────────────────────────────
async function runRandomDemo() {
  const btn = document.getElementById('btn-demo');
  btn.disabled = true;
  btn.textContent = '⏳ Loading…';
  try {
    const res  = await fetch('/api/gallery');
    const data = await res.json();
    if(!data.photos || !data.photos.length) {
      alert('No photos indexed yet. Ask the admin to index some photos first.');
      btn.disabled = false; btn.innerHTML = '⚡ Pick Random &amp; Search';
      return;
    }
    // Pick a random photo
    const photo = data.photos[Math.floor(Math.random() * data.photos.length)];
    btn.textContent = `⏳ Searching…`;

    // Show mini preview of picked photo
    const pickedEl = document.getElementById('demo-picked');
    document.getElementById('demo-picked-img').src = `data:image/jpeg;base64,${photo.thumbnail}`;
    document.getElementById('demo-picked-name').textContent = photo.file;
    pickedEl.style.display = 'flex';

    // Convert thumbnail to File and search
    const byteStr = atob(photo.thumbnail);
    const ab = new ArrayBuffer(byteStr.length);
    const ia = new Uint8Array(ab);
    for(let i=0;i<byteStr.length;i++) ia[i]=byteStr.charCodeAt(i);
    const file = new File([ab], photo.file, {type:'image/jpeg'});
    const b64  = photo.thumbnail;

    // Switch to search tab and show results there
    switchTab('search');
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('search-progress').classList.add('show');
    document.getElementById('btn-search').disabled = true;

    const sres  = await fetch('/api/search', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        filename: photo.file,
        image: b64,
        threshold: parseInt(document.getElementById('threshold-slider').value) / 100
      })
    });
    const sdata = await sres.json();
    renderResult(sdata, file, `data:image/jpeg;base64,${b64}`);

    document.getElementById('search-progress').classList.remove('show');
    document.getElementById('btn-search').disabled = false;
  } catch(e) {
    alert('Demo failed: ' + e.message);
  }
  btn.disabled = false;
  btn.innerHTML = 'Pick Random and Search';
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i)=>{
    t.classList.toggle('active', ['search','gallery'][i]===tab);
  });
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
  if(tab==='gallery') loadGallery();
}

// ── Search mode toggle ────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach((b,i)=>{
    b.classList.toggle('active', ['gallery','upload'][i]===mode);
  });
  document.querySelectorAll('.mode-section').forEach((s,i)=>{
    s.classList.toggle('active', ['mode-gallery','mode-upload'][i]==='mode-'+mode);
  });
  if(mode==='gallery' && !galleryLoaded) loadPickGrid();
}

// ── File preview (upload mode) ────────────────────────────────────────────────
function previewFiles(input, gridId) {
  const files = Array.from(input.files);
  const grid  = document.getElementById(gridId);
  grid.innerHTML = '';
  searchFiles = [];
  files.forEach(f => {
    searchFiles.push(f);
    const reader = new FileReader();
    reader.onload = e => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.innerHTML = `<img src="${e.target.result}"/><button class="remove" onclick="removePreview(this,'${f.name}')">✕</button>`;
      grid.appendChild(item);
    };
    reader.readAsDataURL(f);
  });
}

function removePreview(btn, name) {
  btn.closest('.preview-item').remove();
  searchFiles = searchFiles.filter(f=>f.name!==name);
}

// ── Gallery picker (search from gallery) ─────────────────────────────────────
async function loadPickGrid() {
  galleryLoaded = true;
  const grid = document.getElementById('pick-grid');
  grid.innerHTML = '<div class="gallery-empty">Loading…</div>';
  try {
    const res  = await fetch('/api/gallery');
    const data = await res.json();
    if(!data.photos || !data.photos.length) {
      grid.innerHTML = '<div class="gallery-empty">No photos indexed yet.</div>';
      return;
    }
    grid.innerHTML = '';
    data.photos.forEach(p => {
      const item = document.createElement('div');
      item.className = 'gallery-item';
      item.dataset.filename = p.file;
      item.innerHTML = `<img src="data:image/jpeg;base64,${p.thumbnail}" loading="lazy"/><div class="g-label">${p.file}</div><div class="g-check">✓</div>`;
      item.onclick = () => togglePick(item, p);
      grid.appendChild(item);
    });
  } catch(e) {
    grid.innerHTML = '<div class="gallery-empty">Failed to load gallery.</div>';
  }
}

function togglePick(item, photo) {
  // Single-select: one photo at a time for search
  document.querySelectorAll('#pick-grid .gallery-item').forEach(el=>el.classList.remove('selected'));
  item.classList.add('selected');
  // Convert base64 thumbnail to a File object for search
  const byteStr = atob(photo.thumbnail);
  const ab = new ArrayBuffer(byteStr.length);
  const ia = new Uint8Array(ab);
  for(let i=0;i<byteStr.length;i++) ia[i]=byteStr.charCodeAt(i);
  pickedFile = new File([ab], photo.file, {type:'image/jpeg'});
}

// ── Gallery panel ─────────────────────────────────────────────────────────────
async function loadGallery() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = '<div class="gallery-empty">Loading…</div>';
  try {
    const res  = await fetch('/api/gallery');
    const data = await res.json();
    const total = data.total_faces ?? 0;
    document.getElementById('stat-total').textContent  = total;
    document.getElementById('stat-photos').textContent = data.photos ? data.photos.length : 0;
    document.getElementById('gallery-badge').textContent = data.photos ? data.photos.length : 0;
    if(!data.photos || !data.photos.length) {
      grid.innerHTML = '<div class="gallery-empty">No photos indexed yet.</div>';
      return;
    }
    grid.innerHTML = '';
    data.photos.forEach(p => {
      const item = document.createElement('div');
      item.className = 'gallery-item';
      item.innerHTML = `<img src="data:image/jpeg;base64,${p.thumbnail}" loading="lazy"/><div class="g-label">${p.file}</div>`;
      grid.appendChild(item);
    });
  } catch(e) {
    grid.innerHTML = '<div class="gallery-empty">Failed to load gallery.</div>';
  }
}

// ── Search ────────────────────────────────────────────────────────────────────
async function toBase64(file) {
  return new Promise(r=>{
    const reader = new FileReader();
    reader.onload = e => r(e.target.result.split(',')[1]);
    reader.readAsDataURL(file);
  });
}

async function runSearch() {
  const threshold = parseInt(document.getElementById('threshold-slider').value) / 100;
  let filesToSearch = [];

  if(currentMode==='upload') {
    filesToSearch = searchFiles;
  } else {
    if(!pickedFile) { alert('Please select a photo from the gallery first.'); return; }
    filesToSearch = [pickedFile];
  }

  if(!filesToSearch.length) { alert('Please select or upload a photo to search.'); return; }

  document.getElementById('search-progress').classList.add('show');
  document.getElementById('btn-search').disabled = true;
  document.getElementById('search-results').innerHTML = '';

  for(const file of filesToSearch) {
    const b64 = await toBase64(file);
    const res  = await fetch('/api/search', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({filename: file.name, image: b64, threshold})
    });
    const data = await res.json();
    renderResult(data, file, `data:image/jpeg;base64,${b64}`);
  }

  document.getElementById('search-progress').classList.remove('show');
  document.getElementById('btn-search').disabled = false;
}

function scoreClass(s) {
  if(s>=.8) return 'score-high';
  if(s>=.6) return 'score-med';
  return 'score-low';
}

function renderResult(data, file, previewSrc) {
  const wrap = document.getElementById('search-results');
  const card = document.createElement('div');
  card.className = 'result-card';
  let matchesHtml = '';
  if(data.error) {
    matchesHtml = `<div class="no-match">✕ ${data.error}</div>`;
  } else if(!data.results || !data.results.length) {
    matchesHtml = `<div class="no-match">No faces detected</div>`;
  } else {
    data.results.forEach(faceResult => {
      if(!faceResult.matches.length) {
        matchesHtml += `<div class="no-match">Face #${faceResult.face_index} → no matches above threshold</div>`;
      } else {
        faceResult.matches.forEach(m => {
          const sc  = scoreClass(m.score);
          const pct = Math.round(m.score*100);
          const thumb = m.thumbnail
            ? `<img class="match-thumb" src="data:image/jpeg;base64,${m.thumbnail}"/>`
            : `<div class="match-thumb" style="background:var(--border)"></div>`;
          matchesHtml += `<div class="match-row">
            <div class="match-img-wrap">${thumb}<span class="match-score-badge ${sc}">${pct}%</span></div>
            <div class="match-info">
              <div class="match-file">${m.file} <span style="color:var(--muted);font-size:.62rem;">· face #${m.face_index}</span></div>
              <div class="match-bar-wrap ${sc}"><div class="match-bar" style="width:${pct}%"></div></div>
            </div></div>`;
        });
      }
    });
    matchesHtml = `<div class="match-list">${matchesHtml}</div>`;
  }
  card.innerHTML = `
    <div class="result-header">
      <div class="result-query-img">
        <img class="query-thumb" src="${previewSrc}"/>
        <div><div class="result-file">${file.name}</div>
        <div class="result-meta">${data.results?data.results.length:0} face(s) detected</div></div>
      </div>
    </div>${matchesHtml}`;
  wrap.appendChild(card);
}

// ── Drag & drop ───────────────────────────────────────────────────────────────
const dropEl = document.getElementById('search-drop');
dropEl.addEventListener('click', ()=> document.getElementById('search-input').click());
dropEl.addEventListener('dragover', e=>{e.preventDefault();dropEl.classList.add('dragover')});
dropEl.addEventListener('dragleave', ()=>dropEl.classList.remove('dragover'));
dropEl.addEventListener('drop', e=>{
  e.preventDefault(); dropEl.classList.remove('dragover');
  const dt = new DataTransfer();
  Array.from(e.dataTransfer.files).forEach(f=>dt.items.add(f));
  document.getElementById('search-input').files = dt.files;
  previewFiles(document.getElementById('search-input'),'search-preview');
});

// Load gallery on init since it's the default mode
loadPickGrid();

// Init badge
fetch('/api/gallery').then(r=>r.json()).then(d=>{
  document.getElementById('gallery-badge').textContent = d.photos?d.photos.length:0;
}).catch(()=>{});
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML — ADMIN PAGE  (/admin?key=...)
# ═══════════════════════════════════════════════════════════════════════════════

ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Admin — Face Indexer</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root{--bg:#080a0f;--card:#111820;--border:#1e2a38;--accent:#00ffe5;--accent2:#ff3c6e;--accent3:#7b61ff;--text:#e8edf3;--muted:#4a5a6a;--success:#00e87a}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;padding:0}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,229,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,229,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}

  /* ── Header ── */
  header{border-bottom:1px solid var(--border);padding:18px 32px;background:rgba(8,10,15,.95);backdrop-filter:blur(14px);display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
  .brand{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:.14em;color:var(--accent2)}
  .brand span{color:var(--accent)}
  .admin-badge{font-family:'DM Mono',monospace;font-size:.6rem;letter-spacing:.14em;padding:4px 10px;border-radius:20px;border:1px solid var(--accent2);color:var(--accent2)}

  /* ── Layout ── */
  .page{max-width:1200px;margin:0 auto;padding:32px;position:relative;z-index:1}
  .section-title{font-family:'DM Mono',monospace;font-size:.62rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:16px;display:flex;align-items:center;gap:10px}
  .section-title::after{content:'';flex:1;height:1px;background:var(--border)}
  .card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:24px;margin-bottom:28px}

  /* ── Drop zone ── */
  .drop-zone{border:1px dashed var(--border);border-radius:4px;padding:40px 30px;text-align:center;cursor:pointer;position:relative;background:rgba(0,0,0,.2);transition:all .2s}
  .drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .drop-zone:hover{border-color:var(--accent);background:rgba(0,255,229,.03)}
  .drop-label{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted)}
  .drop-label strong{color:var(--accent)}

  /* ── Preview (upload queue) ── */
  .preview-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(90px,1fr));gap:8px;margin:16px 0}
  .preview-item{position:relative;aspect-ratio:1;border-radius:3px;overflow:hidden;border:1px solid var(--border)}
  .preview-item img{width:100%;height:100%;object-fit:cover}

  /* ── Buttons ── */
  .btn{font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;padding:10px 22px;border-radius:3px;border:1px solid;cursor:pointer;background:transparent;transition:all .2s}
  .btn-primary{border-color:var(--accent);color:var(--accent)}
  .btn-primary:hover{box-shadow:0 0 16px rgba(0,255,229,.2)}
  .btn-danger{border-color:var(--accent2);color:var(--accent2)}
  .btn-ghost{border-color:var(--border);color:var(--muted)}
  .btn:disabled{opacity:.35;cursor:not-allowed}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:16px}

  /* ── Terminal log ── */
  .terminal{background:#06090d;border:1px solid var(--border);border-radius:4px;padding:16px;font-family:'DM Mono',monospace;font-size:.7rem;line-height:1.8;max-height:220px;overflow-y:auto;margin-top:16px}
  .terminal:empty::before{content:'> awaiting input…';color:var(--muted)}
  .log-line{display:block}
  .ok{color:var(--success)} .err{color:var(--accent2)} .info{color:var(--accent)} .dim{color:var(--muted)}
  .ts{color:var(--muted);margin-right:8px}

  /* ── Progress ── */
  .progress-wrap{height:2px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:12px;display:none}
  .progress-wrap.show{display:block}
  .progress-bar{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent3));animation:indeterminate 1.2s infinite ease-in-out;width:40%}
  @keyframes indeterminate{0%{transform:translateX(-100%)}100%{transform:translateX(350%)}}

  /* ── Gallery manager ── */
  .gallery-stats{display:flex;gap:24px;margin-bottom:20px}
  .gstat{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted)}
  .gstat strong{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:var(--accent);display:block;line-height:1}

  .mgmt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
  .mgmt-item{position:relative;border-radius:4px;overflow:hidden;border:2px solid var(--border);aspect-ratio:1;background:var(--card);transition:border-color .2s;group}
  .mgmt-item:hover{border-color:var(--accent2)}
  .mgmt-item img{width:100%;height:100%;object-fit:cover;display:block}
  .mgmt-item .m-label{position:absolute;bottom:0;left:0;right:0;background:rgba(8,10,15,.9);font-family:'DM Mono',monospace;font-size:.5rem;color:var(--muted);padding:5px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .mgmt-item .m-faces{position:absolute;top:6px;left:6px;font-family:'DM Mono',monospace;font-size:.55rem;background:rgba(8,10,15,.85);color:var(--accent);border:1px solid var(--accent);border-radius:10px;padding:2px 7px}
  .mgmt-item .m-del{position:absolute;top:6px;right:6px;width:26px;height:26px;background:rgba(255,60,110,.9);border:none;border-radius:3px;color:#fff;font-size:.75rem;cursor:pointer;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .15s;font-weight:700}
  .mgmt-item:hover .m-del{opacity:1}
  .mgmt-item.deleting{opacity:.4;pointer-events:none}
  .mgmt-empty{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted);text-align:center;padding:50px 0;grid-column:1/-1}

  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>
<header>
  <div class="brand"><span>SS</span> WebCreation — Admin</div>
  <div class="admin-badge">⬡ ADMIN PANEL</div>
</header>
<div class="page">

  <!-- ══ INDEX NEW PHOTOS ══ -->
  <div class="card">
    <p class="section-title">Index new photos</p>
    <div class="drop-zone" onclick="document.getElementById('admin-input').click()">
      <input type="file" id="admin-input" accept="image/*" multiple onchange="previewFiles(this)"/>
      <div class="drop-label">Drop images here or <strong>click to browse</strong></div>
      <div class="drop-label" style="margin-top:5px;font-size:.6rem;">PNG · JPG · WEBP · BMP</div>
    </div>
    <div class="preview-grid" id="admin-preview"></div>
    <div class="row">
      <button class="btn btn-primary" onclick="runIndex()" id="btn-index">⬡ INDEX FACES</button>
      <button class="btn btn-danger"  onclick="clearDB()">✕ CLEAR ENTIRE DB</button>
    </div>
    <div class="progress-wrap" id="admin-progress"><div class="progress-bar"></div></div>
    <div class="terminal" id="admin-log"></div>
  </div>

  <!-- ══ GALLERY MANAGER ══ -->
  <div class="card">
    <p class="section-title">Manage indexed photos</p>
    <div class="gallery-stats">
      <div class="gstat"><strong id="g-total">—</strong>Total Faces</div>
      <div class="gstat"><strong id="g-photos">—</strong>Unique Photos</div>
    </div>
    <div class="row" style="margin-top:0;margin-bottom:20px">
      <button class="btn btn-ghost" onclick="loadGallery()">↻ Refresh</button>
    </div>
    <div class="mgmt-grid" id="mgmt-grid">
      <div class="mgmt-empty">Loading…</div>
    </div>
  </div>

</div>
<script>
let adminFiles = [];

// ── Upload preview ────────────────────────────────────────────────────────────
function previewFiles(input) {
  const files = Array.from(input.files);
  const grid  = document.getElementById('admin-preview');
  grid.innerHTML = '';
  adminFiles = [];
  files.forEach(f => {
    adminFiles.push(f);
    const reader = new FileReader();
    reader.onload = e => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.innerHTML = `<img src="${e.target.result}"/>`;
      grid.appendChild(item);
    };
    reader.readAsDataURL(f);
  });
}

// ── Logging ───────────────────────────────────────────────────────────────────
function ts() { return new Date().toTimeString().slice(0,8); }
function log(msg, cls='info') {
  const t = document.getElementById('admin-log');
  const line = document.createElement('span');
  line.className = `log-line ${cls}`;
  line.innerHTML = `<span class="ts">[${ts()}]</span>${msg}`;
  t.appendChild(line);
  t.scrollTop = t.scrollHeight;
}

// ── Base64 helper ─────────────────────────────────────────────────────────────
async function toBase64(file) {
  return new Promise(r => {
    const reader = new FileReader();
    reader.onload = e => r(e.target.result.split(',')[1]);
    reader.readAsDataURL(file);
  });
}

// ── Index ─────────────────────────────────────────────────────────────────────
async function runIndex() {
  if(!adminFiles.length) { log('No files selected.','err'); return; }
  document.getElementById('admin-progress').classList.add('show');
  document.getElementById('btn-index').disabled = true;
  log(`Starting indexing of ${adminFiles.length} image(s)…`, 'dim');

  for(const file of adminFiles) {
    const b64 = await toBase64(file);
    const res  = await fetch('/api/index', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({filename: file.name, image: b64})
    });
    const data = await res.json();
    if(data.error) log(`✕ ${file.name}: ${data.error}`, 'err');
    else           log(`✓ ${file.name} → ${data.faces} face(s) indexed`, 'ok');
  }

  document.getElementById('admin-progress').classList.remove('show');
  document.getElementById('btn-index').disabled = false;
  log('Done.', 'dim');
  loadGallery(); // refresh gallery after indexing
}

// ── Clear all ─────────────────────────────────────────────────────────────────
async function clearDB() {
  if(!confirm('Delete ALL indexed faces? This cannot be undone.')) return;
  const res  = await fetch('/api/clear', {method:'POST'});
  const data = await res.json();
  log(data.message || 'Cleared.', 'err');
  loadGallery();
}

// ── Delete single photo ───────────────────────────────────────────────────────
async function deletePhoto(filename, itemEl) {
  if(!confirm(`Delete "${filename}" and all its faces from the database?`)) return;
  itemEl.classList.add('deleting');
  const res  = await fetch('/api/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({filename})
  });
  const data = await res.json();
  if(data.error) {
    log(`✕ Could not delete ${filename}: ${data.error}`, 'err');
    itemEl.classList.remove('deleting');
  } else {
    log(`✓ Deleted "${filename}" (${data.deleted} record(s) removed)`, 'ok');
    itemEl.remove();
    // Update counts
    const cur = parseInt(document.getElementById('g-photos').textContent) || 0;
    document.getElementById('g-photos').textContent = Math.max(0, cur - 1);
    const curFaces = parseInt(document.getElementById('g-total').textContent) || 0;
    document.getElementById('g-total').textContent = Math.max(0, curFaces - (data.deleted || 0));
  }
}

// ── Load gallery ──────────────────────────────────────────────────────────────
async function loadGallery() {
  const grid = document.getElementById('mgmt-grid');
  grid.innerHTML = '<div class="mgmt-empty">Loading…</div>';
  try {
    const res  = await fetch('/api/gallery');
    const data = await res.json();
    document.getElementById('g-total').textContent  = data.total_faces ?? 0;
    document.getElementById('g-photos').textContent = data.photos ? data.photos.length : 0;

    if(!data.photos || !data.photos.length) {
      grid.innerHTML = '<div class="mgmt-empty">No photos indexed yet.</div>';
      return;
    }

    grid.innerHTML = '';
    data.photos.forEach(p => {
      const item = document.createElement('div');
      item.className = 'mgmt-item';
      const thumb = p.thumbnail
        ? `<img src="data:image/jpeg;base64,${p.thumbnail}" loading="lazy"/>`
        : `<div style="width:100%;height:100%;background:var(--border);display:flex;align-items:center;justify-content:center;font-family:DM Mono,monospace;font-size:.6rem;color:var(--muted)">NO PREVIEW</div>`;
      item.innerHTML = `
        ${thumb}
        <div class="m-faces">${p.face_count ?? '?'} face${p.face_count===1?'':'s'}</div>
        <div class="m-label">${p.file}</div>
        <button class="m-del" title="Delete this photo">✕</button>`;
      item.querySelector('.m-del').onclick = (e) => {
        e.stopPropagation();
        deletePhoto(p.file, item);
      };
      grid.appendChild(item);
    });
  } catch(e) {
    grid.innerHTML = '<div class="mgmt-empty">Failed to load gallery.</div>';
  }
}

loadGallery();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return "ok", 200


@app.route("/")
def index():
    return render_template_string(USER_HTML)


@app.route("/admin")
def admin():
    return render_template_string(ADMIN_HTML)


# ── Public: search ────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search():
    ensure_collection()
    client    = get_db()
    data      = request.json
    threshold = float(data.get("threshold", 0.5))

    img = decode_image(data["image"])
    if img is None:
        return jsonify({"error": "Could not decode image"})

    ml_res = get_embeddings_from_ml(img)
    if "error" in ml_res:
        return jsonify({"error": ml_res["error"]})

    embeddings = ml_res.get("faces", [])
    results    = []

    for i, emb in enumerate(embeddings):
        raw     = client.query_points("faces", query=emb, limit=10).points
        matches = []
        for r in raw:
            score = r.score
            # If this is the exact same source photo, set score to 1.0
            if r.payload.get("file") == data.get("filename"):
                score = 1.0
            if score >= threshold:
                matches.append({
                    "file":       r.payload["file"],
                    "face_index": r.payload["face"],
                    "score":      round(score, 4),
                    "thumbnail":  None
                })
        results.append({"face_index": i, "matches": matches})

    return jsonify({"results": results})


@app.route("/api/gallery")
def api_gallery():
    """Returns unique photos with a small thumbnail (resized from stored payload).
    We keep it simple: return filenames + a placeholder thumbnail.
    For real thumbnails, store them at index time (see admin route below).
    """
    ensure_collection()
    client = get_db()

    total = client.count("faces").count
    pts, _ = client.scroll("faces", limit=2000, with_payload=True, with_vectors=False)

    seen   = {}
    photos = []
    for p in pts:
        fname = p.payload.get("file", "?")
        if fname not in seen:
            seen[fname] = {"thumbnail": p.payload.get("thumbnail", None), "face_count": 0}
        seen[fname]["face_count"] += 1

    for fname, meta in seen.items():
        photos.append({"file": fname, "thumbnail": meta["thumbnail"], "face_count": meta["face_count"]})

    return jsonify({"total_faces": total, "photos": photos})


# ── Admin-only: index & clear ─────────────────────────────────────────────────

@app.route("/api/index", methods=["POST"])
def api_index():

    ensure_collection()
    client = get_db()
    data   = request.json

    img = decode_image(data["image"])
    if img is None:
        return jsonify({"error": "Could not decode image"})

    # Create a thumbnail — larger size, higher quality to preserve face detail
    h, w   = img.shape[:2]
    scale  = 256 / max(h, w)
    thumb  = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
    thumb_b64 = base64.b64encode(buf).decode()

    ml_res = get_embeddings_from_ml(img)
    if "error" in ml_res:
        return jsonify({"error": ml_res["error"]})

    embeddings = ml_res.get("faces", [])
    count      = client.count("faces").count

    from qdrant_client.models import PointStruct
    for i, emb in enumerate(embeddings):
        client.upsert("faces", points=[PointStruct(
            id      = count + i,
            vector  = emb,
            payload = {
                "file":      data["filename"],
                "face":      i,
                "thumbnail": thumb_b64      # ← stored so gallery can show it
            }
        )])

    _save_backup(client)
    return jsonify({"faces": len(embeddings)})


@app.route("/api/delete", methods=["POST"])
def api_delete():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client   = get_db()
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"})
    try:
        # Find all point IDs with this filename
        pts, _ = client.scroll(
            "faces", limit=5000,
            scroll_filter=Filter(must=[FieldCondition(key="file", match=MatchValue(value=filename))]),
            with_payload=False, with_vectors=False
        )
        ids = [p.id for p in pts]
        if ids:
            client.delete("faces", points_selector=ids)
        _save_backup(client)
        return jsonify({"deleted": len(ids)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/clear", methods=["POST"])
def api_clear():

    client = get_db()
    if client.collection_exists("faces"):
        client.delete_collection("faces")
    ensure_collection()
    if BACKUP.exists():
        BACKUP.unlink()
    return jsonify({"message": "Collection cleared and recreated."})


# ── Stats (admin convenience) ─────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():

    ensure_collection()
    client = get_db()
    total  = client.count("faces").count
    pts, _ = client.scroll("faces", limit=200, with_payload=True, with_vectors=False)
    points = [{"id": p.id, "file": p.payload.get("file","?"), "face": p.payload.get("face",0)} for p in pts]
    return jsonify({"total": total, "points": points})


# ═══════════════════════════════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
