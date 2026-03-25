import os
import cv2
import numpy as np
import base64
import json
import shutil
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

app = Flask(__name__)

# 🔥 HuggingFace ML API

ML_API = "https://mohammedlokhandwala-ssweb.hf.space/extract"

DB_PATH   = Path("./qdrant_db")
LOCK_FILE = DB_PATH / ".lock"
BACKUP    = Path("./faces_backup.json")

_db = None

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
_db = QdrantClient(path=str(DB_PATH))
except Exception:
_db = QdrantClient(":memory:")
_restore_backup(_db)
return _db

def _restore_backup(client):
if not BACKUP.exists():
return
try:
data = json.loads(BACKUP.read_text())
if not data:
return
if not client.collection_exists("faces"):
client.create_collection("faces", vectors_config=VectorParams(size=512, distance=Distance.COSINE))
pts = [PointStruct(id=r["id"], vector=r["vector"], payload=r["payload"]) for r in data]
client.upsert("faces", points=pts)
except Exception:
pass

def _save_backup(client):
try:
pts, _ = client.scroll("faces", limit=10000, with_payload=True, with_vectors=True)
data = [{"id": p.id, "vector": p.vector, "payload": p.payload} for p in pts]
BACKUP.write_text(json.dumps(data))
except Exception:
pass

def ensure_collection():
client = get_db()
if not client.collection_exists("faces"):
client.create_collection("faces", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

ensure_collection()

# 🔥 CALL ML API

def get_embeddings_from_ml(img):
_, buffer = cv2.imencode(".jpg", img)
files = {"file": ("image.jpg", buffer.tobytes(), "image/jpeg")}
try:
res = requests.post(ML_API, files=files, timeout=30)
return res.json()
except Exception as e:
return {"error": str(e)}

# ================= HTML (YOUR ORIGINAL UI) =================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SS WebCreation — Face Search</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@200;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #080a0f;
    --surface: #0d1117;
    --card: #111820;
    --border: #1e2a38;
    --accent: #00ffe5;
    --accent2: #ff3c6e;
    --accent3: #7b61ff;
    --text: #e8edf3;
    --muted: #4a5a6a;
    --success: #00e87a;
    --warn: #ffba00;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; min-height: 100vh; overflow-x: hidden; }
  body::before {
    content: ''; position: fixed; inset: 0;
    background-image: linear-gradient(rgba(0,255,229,.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,229,.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
  }
  body::after {
    content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.15) 2px, rgba(0,0,0,.15) 4px);
    pointer-events: none; z-index: 0;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 0 24px; position: relative; z-index: 1; }
  header { border-bottom: 1px solid var(--border); padding: 20px 0; position: sticky; top: 0; background: rgba(8,10,15,.88); backdrop-filter: blur(14px); z-index: 100; }
  .header-inner { display: flex; align-items: center; justify-content: space-between; }
  .brand { display: flex; flex-direction: column; }
  .brand-name {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: .14em;
    color: var(--accent); text-shadow: 0 0 20px rgba(0,255,229,.5), 0 0 60px rgba(0,255,229,.2); line-height: 1;
  }
  .brand-name span { color: var(--accent2); }
  .brand-sub { font-family: 'DM Mono', monospace; font-size: .55rem; letter-spacing: .22em; color: var(--muted); text-transform: uppercase; margin-top: 3px; }
  .status-pill { font-family: 'DM Mono', monospace; font-size: .65rem; letter-spacing: .1em; padding: 4px 12px; border-radius: 20px; border: 1px solid var(--accent); color: var(--accent); display: flex; align-items: center; gap: 6px; }
  .status-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 6px var(--accent); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .tabs { display: flex; margin: 40px 0 0; border-bottom: 1px solid var(--border); }
  .tab { font-family: 'DM Mono', monospace; font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; padding: 14px 28px; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all .2s; user-select: none; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .badge { background: var(--accent2); color: #fff; border-radius: 4px; font-size: .55rem; padding: 1px 5px; margin-left: 6px; vertical-align: middle; }
  .panel { display: none; padding: 40px 0; }
  .panel.active { display: block; animation: fadeUp .35s ease; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
  .drop-zone { border: 1px dashed var(--border); border-radius: 4px; padding: 60px 40px; text-align: center; cursor: pointer; transition: all .25s; position: relative; overflow: hidden; background: var(--card); }
  .drop-zone::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 50% 0%, rgba(0,255,229,.05) 0%, transparent 70%); opacity: 0; transition: opacity .3s; }
  .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); background: rgba(0,255,229,.03); }
  .drop-zone:hover::before, .drop-zone.dragover::before { opacity: 1; }
  .drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .drop-icon { font-size: 2.2rem; margin-bottom: 14px; opacity: .4; display: block; }
  .drop-label { font-family: 'DM Mono', monospace; font-size: .75rem; letter-spacing: .1em; color: var(--muted); text-transform: uppercase; }
  .drop-label strong { color: var(--accent); font-weight: 500; }
  .preview-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; margin-top: 20px; }
  .preview-item { position: relative; border-radius: 3px; overflow: hidden; border: 1px solid var(--border); aspect-ratio: 1; background: var(--card); }
  .preview-item img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .preview-item .remove { position: absolute; top: 4px; right: 4px; width: 20px; height: 20px; background: rgba(255,60,110,.9); border: none; border-radius: 2px; color: #fff; font-size: .7rem; cursor: pointer; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity .2s; }
  .preview-item:hover .remove { opacity: 1; }
  .btn { display: inline-flex; align-items: center; gap: 8px; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; padding: 12px 28px; border-radius: 3px; border: 1px solid; cursor: pointer; transition: all .2s; position: relative; overflow: hidden; }
  .btn::after { content: ''; position: absolute; inset: 0; background: currentColor; opacity: 0; transition: opacity .2s; }
  .btn:hover::after { opacity: .08; }
  .btn-primary { background: transparent; border-color: var(--accent); color: var(--accent); }
  .btn-primary:hover { box-shadow: 0 0 20px rgba(0,255,229,.25); }
  .btn-danger { background: transparent; border-color: var(--accent2); color: var(--accent2); }
  .btn-ghost { background: transparent; border-color: var(--border); color: var(--muted); }
  .btn:disabled { opacity: .35; cursor: not-allowed; }
  .stats-bar { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 4px; margin-bottom: 32px; overflow: hidden; }
  .stat { background: var(--card); padding: 20px 24px; }
  .stat-val { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; line-height: 1; color: var(--accent); text-shadow: 0 0 20px rgba(0,255,229,.3); }
  .stat-key { font-family: 'DM Mono', monospace; font-size: .6rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-top: 4px; }
  .terminal { background: #06090d; border: 1px solid var(--border); border-radius: 4px; padding: 20px; font-family: 'DM Mono', monospace; font-size: .72rem; line-height: 1.8; max-height: 260px; overflow-y: auto; margin-top: 24px; }
  .terminal:empty::before { content: '> awaiting input...'; color: var(--muted); }
  .log-line { display: block; }
  .log-line.ok { color: var(--success); }
  .log-line.err { color: var(--accent2); }
  .log-line.info { color: var(--accent); }
  .log-line.dim { color: var(--muted); }
  .log-line .ts { color: var(--muted); margin-right: 8px; }
  .result-card { border: 1px solid var(--border); border-radius: 4px; overflow: hidden; margin-bottom: 16px; background: var(--card); transition: border-color .2s; }
  .result-card:hover { border-color: var(--accent3); }
  .result-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,.2); }
  .result-query-img { display: flex; align-items: center; gap: 12px; }
  .query-thumb { width: 44px; height: 44px; border-radius: 3px; object-fit: cover; border: 1px solid var(--border); }
  .result-file { font-family: 'DM Mono', monospace; font-size: .75rem; color: var(--text); letter-spacing: .04em; }
  .result-meta { font-size: .65rem; color: var(--muted); margin-top: 2px; }
  .no-match { display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: .65rem; color: var(--muted); background: rgba(255,255,255,.03); border: 1px solid var(--border); border-radius: 20px; padding: 4px 12px; }
  .match-list { padding: 16px 20px; display: flex; flex-direction: column; gap: 10px; }
  .match-row { display: flex; align-items: center; gap: 16px; }
  .match-img-wrap { position: relative; flex-shrink: 0; }
  .match-thumb { width: 52px; height: 52px; border-radius: 3px; object-fit: cover; border: 1px solid var(--border); }
  .match-score-badge { position: absolute; bottom: -6px; left: 50%; transform: translateX(-50%); font-family: 'DM Mono', monospace; font-size: .55rem; white-space: nowrap; padding: 2px 6px; border-radius: 20px; border: 1px solid; background: var(--bg); }
  .match-info { flex: 1; }
  .match-file { font-family: 'DM Mono', monospace; font-size: .72rem; color: var(--text); }
  .match-bar-wrap { margin-top: 6px; height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .match-bar { height: 100%; border-radius: 2px; transition: width .8s cubic-bezier(.16,1,.3,1); }
  .score-high { color: var(--success); border-color: var(--success); }
  .score-high .match-bar { background: var(--success); }
  .score-med  { color: var(--warn); border-color: var(--warn); }
  .score-med  .match-bar { background: var(--warn); }
  .score-low  { color: var(--accent2); border-color: var(--accent2); }
  .score-low  .match-bar { background: var(--accent2); }
  .slider-row { display: flex; align-items: center; gap: 14px; margin-bottom: 24px; }
  .slider-label { font-family: 'DM Mono', monospace; font-size: .68rem; letter-spacing: .08em; color: var(--muted); text-transform: uppercase; white-space: nowrap; }
  input[type=range] { flex: 1; -webkit-appearance: none; height: 2px; background: var(--border); border-radius: 2px; outline: none; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); cursor: pointer; }
  .slider-val { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; color: var(--accent); min-width: 40px; text-align: right; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .mt-md { margin-top: 28px; }
  .section-title { font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
  .section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }
  .progress-wrap { height: 2px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 16px; display: none; }
  .progress-wrap.show { display: block; }
  .progress-bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent3)); width: 0%; transition: width .3s; border-radius: 2px; }
  .progress-bar.indeterminate { animation: indeterminate 1.2s infinite ease-in-out; width: 40% !important; }
  @keyframes indeterminate { 0%{transform:translateX(-100%)} 100%{transform:translateX(350%)} }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
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
  <div class="tabs">
    <div class="tab active" onclick="switchTab('index')">INDEX</div>
    <div class="tab" onclick="switchTab('search')">SEARCH <span class="badge" id="match-badge" style="display:none">!</span></div>
    <div class="tab" onclick="switchTab('database')">DATABASE</div>
  </div>
  <div class="panel active" id="panel-index">
    <p class="section-title">Index faces into the database</p>
    <div class="drop-zone" id="index-drop" onclick="document.getElementById('index-input').click()">
      <input type="file" id="index-input" accept="image/*" multiple onchange="previewFiles(this,'index-preview','index-drop')"/>
      <span class="drop-icon">⬡</span>
      <div class="drop-label">Drop images here or <strong>click to browse</strong></div>
      <div class="drop-label" style="margin-top:6px;font-size:.62rem;">PNG · JPG · WEBP · BMP</div>
    </div>
    <div class="preview-grid" id="index-preview"></div>
    <div class="mt-md row">
      <button class="btn btn-primary" onclick="runIndex()" id="btn-index"><span>⬡</span> INDEX FACES</button>
      <button class="btn btn-danger" onclick="clearCollection()">✕ CLEAR DATABASE</button>
    </div>
    <div class="progress-wrap" id="index-progress"><div class="progress-bar indeterminate"></div></div>
    <div class="terminal" id="index-log"></div>
  </div>
  <div class="panel" id="panel-search">
    <p class="section-title">Search for matching faces</p>
    <div class="slider-row">
      <span class="slider-label">Min Similarity</span>
      <input type="range" min="0" max="100" value="50" id="threshold-slider" oninput="document.getElementById('threshold-val').textContent = this.value + '%'"/>
      <span class="slider-val" id="threshold-val">50%</span>
    </div>
    <div class="drop-zone" id="search-drop" onclick="document.getElementById('search-input').click()">
      <input type="file" id="search-input" accept="image/*" multiple onchange="previewFiles(this,'search-preview','search-drop')"/>
      <span class="drop-icon">◎</span>
      <div class="drop-label">Drop query images here or <strong>click to browse</strong></div>
    </div>
    <div class="preview-grid" id="search-preview"></div>
    <div class="mt-md row">
      <button class="btn btn-primary" onclick="runSearch()" id="btn-search"><span>◎</span> SEARCH FACES</button>
    </div>
    <div class="progress-wrap" id="search-progress"><div class="progress-bar indeterminate"></div></div>
    <div id="search-results" class="mt-md"></div>
  </div>
  <div class="panel" id="panel-database">
    <p class="section-title">Collection status</p>
    <div class="stats-bar">
      <div class="stat"><div class="stat-val" id="stat-total">—</div><div class="stat-key">Faces Indexed</div></div>
      <div class="stat"><div class="stat-val" id="stat-dim">512</div><div class="stat-key">Embedding Dims</div></div>
      <div class="stat"><div class="stat-val" id="stat-metric">COS</div><div class="stat-key">Distance Metric</div></div>
    </div>
    <p class="section-title">Indexed records</p>
    <div id="db-records" class="terminal" style="max-height:420px;font-size:.65rem;"></div>
    <div class="mt-md row">
      <button class="btn btn-ghost" onclick="loadDbStats()">↻ REFRESH</button>
    </div>
  </div>
</div>
<script>
let indexFiles = [], searchFiles = [];

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i)=>{
    t.classList.toggle('active', ['index','search','database'][i]===tab);
  });
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
  if(tab==='database') loadDbStats();
}

function previewFiles(input, gridId) {
  const files = Array.from(input.files);
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  const arr = gridId.includes('index') ? (indexFiles=[]) : (searchFiles=[]);
  files.forEach(f => {
    arr.push(f);
    const reader = new FileReader();
    reader.onload = e => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.innerHTML = `<img src="${e.target.result}"/><button class="remove" onclick="removePreview(this,'${gridId}','${f.name}')">✕</button>`;
      grid.appendChild(item);
    };
    reader.readAsDataURL(f);
  });
}

function removePreview(btn, gridId, name) {
  btn.closest('.preview-item').remove();
  const arr = gridId.includes('index') ? indexFiles : searchFiles;
  const i = arr.findIndex(f=>f.name===name);
  if(i>-1) arr.splice(i,1);
}

function ts() { return new Date().toTimeString().slice(0,8); }

function log(termId, msg, cls='info') {
  const t = document.getElementById(termId);
  const line = document.createElement('span');
  line.className = `log-line ${cls}`;
  line.innerHTML = `<span class="ts">[${ts()}]</span>${msg}`;
  t.appendChild(line);
  t.scrollTop = t.scrollHeight;
}

async function toBase64(file) {
  return new Promise(r=>{
    const reader = new FileReader();
    reader.onload = e => r(e.target.result.split(',')[1]);
    reader.readAsDataURL(file);
  });
}

async function runIndex() {
  if(!indexFiles.length) { log('index-log','No files selected.','err'); return; }
  document.getElementById('index-progress').classList.add('show');
  document.getElementById('btn-index').disabled = true;
  document.getElementById('index-log').innerHTML = '';
  log('index-log',`Starting indexing of ${indexFiles.length} image(s)…`,'dim');
  for(const file of indexFiles) {
    const b64 = await toBase64(file);
    const res = await fetch('/api/index', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({filename: file.name, image: b64}) });
    const data = await res.json();
    if(data.error) log('index-log',`✕ ${file.name}: ${data.error}`,'err');
    else log('index-log',`✓ ${file.name} → ${data.faces} face(s) indexed`,'ok');
  }
  document.getElementById('index-progress').classList.remove('show');
  document.getElementById('btn-index').disabled = false;
  log('index-log','Done.','dim');
}

async function runSearch() {
  if(!searchFiles.length) return;
  const threshold = parseInt(document.getElementById('threshold-slider').value) / 100;
  document.getElementById('search-progress').classList.add('show');
  document.getElementById('btn-search').disabled = true;
  document.getElementById('search-results').innerHTML = '';
  let totalMatches = 0;
  for(const file of searchFiles) {
    const b64 = await toBase64(file);
    const res = await fetch('/api/search', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({filename: file.name, image: b64, threshold}) });
    const data = await res.json();
    renderResult(data, file, `data:image/jpeg;base64,${b64}`);
    totalMatches += data.results ? data.results.reduce((a,r)=>a+r.matches.length,0) : 0;
  }
  document.getElementById('search-progress').classList.remove('show');
  document.getElementById('btn-search').disabled = false;
  document.getElementById('match-badge').style.display = totalMatches ? '' : 'none';
}

function scoreClass(s) {
  if(s >= .8) return 'score-high';
  if(s >= .6) return 'score-med';
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
          const sc = scoreClass(m.score);
          const pct = Math.round(m.score * 100);
          const thumb = m.thumbnail ? `<img class="match-thumb" src="data:image/jpeg;base64,${m.thumbnail}"/>` : `<div class="match-thumb" style="background:var(--border)"></div>`;
          matchesHtml += `<div class="match-row"><div class="match-img-wrap">${thumb}<span class="match-score-badge ${sc}">${pct}%</span></div><div class="match-info"><div class="match-file">${m.file} <span style="color:var(--muted);font-size:.62rem;">· face #${m.face_index}</span></div><div class="match-bar-wrap ${sc}"><div class="match-bar" style="width:${pct}%"></div></div></div></div>`;
        });
      }
    });
    matchesHtml = `<div class="match-list">${matchesHtml}</div>`;
  }
  card.innerHTML = `<div class="result-header"><div class="result-query-img"><img class="query-thumb" src="${previewSrc}"/><div><div class="result-file">${file.name}</div><div class="result-meta">${data.results ? data.results.length : 0} face(s) detected</div></div></div></div>${matchesHtml}`;
  wrap.appendChild(card);
}

async function clearCollection() {
  if(!confirm('Clear all indexed faces?')) return;
  const res = await fetch('/api/clear', {method:'POST'});
  const data = await res.json();
  log('index-log', data.message || 'Collection cleared.', 'err');
}

async function loadDbStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  document.getElementById('stat-total').textContent = data.total ?? '—';
  const rec = document.getElementById('db-records');
  if(!data.points || !data.points.length) { rec.innerHTML = '<span class="log-line dim">> no records found</span>'; return; }
  rec.innerHTML = data.points.map(p => `<span class="log-line"><span class="ts">[id:${p.id}]</span><span class="info"> ${p.file}</span> <span class="dim">· face #${p.face}</span></span>`).join('');
}

['index-drop','search-drop'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('dragover'); });
  el.addEventListener('dragleave', () => el.classList.remove('dragover'));
  el.addEventListener('drop', e => {
    e.preventDefault(); el.classList.remove('dragover');
    const inputId = id==='index-drop' ? 'index-input' : 'search-input';
    const gridId  = id==='index-drop' ? 'index-preview' : 'search-preview';
    const dt = new DataTransfer();
    Array.from(e.dataTransfer.files).forEach(f=>dt.items.add(f));
    document.getElementById(inputId).files = dt.files;
    previewFiles(document.getElementById(inputId), gridId);
  });
});

loadDbStats();
</script>
</body>
</html>"""

# ================= UTILS =================

def decode_image(b64_str):
img_bytes = base64.b64decode(b64_str)
arr = np.frombuffer(img_bytes, np.uint8)
return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_image(img, quality=70):
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
return base64.b64encode(buf).decode()

@app.route("/")
def index():
return render_template_string(HTML)

# ================= INDEX =================

@app.route("/api/index", methods=["POST"])
def api_index():
ensure_collection()
client = get_db()
data = request.json

```
img = decode_image(data["image"])
if img is None:
    return jsonify({"error": "Could not decode image"})

ml_res = get_embeddings_from_ml(img)
if "error" in ml_res:
    return jsonify({"error": ml_res["error"]})

embeddings = ml_res.get("faces", [])
count = client.count("faces").count

for i, emb in enumerate(embeddings):
    client.upsert("faces", points=[PointStruct(
        id=count + i,
        vector=emb,
        payload={"file": data["filename"], "face": i}
    )])

_save_backup(client)
return jsonify({"faces": len(embeddings)})
```

# ================= SEARCH =================

@app.route("/api/search", methods=["POST"])
def api_search():
ensure_collection()
client = get_db()
data = request.json

```
threshold = float(data.get("threshold", 0.5))

img = decode_image(data["image"])
if img is None:
    return jsonify({"error": "Could not decode image"})

ml_res = get_embeddings_from_ml(img)
if "error" in ml_res:
    return jsonify({"error": ml_res["error"]})

embeddings = ml_res.get("faces", [])
results = []

for i, emb in enumerate(embeddings):
    raw = client.query_points("faces", query=emb, limit=10).points
    matches = []

    for r in raw:
        if r.score >= threshold:
            matches.append({
                "file": r.payload["file"],
                "face_index": r.payload["face"],
                "score": round(r.score, 4),
                "thumbnail": None
            })

    results.append({"face_index": i, "matches": matches})

return jsonify({"results": results})
```

# ================= CLEAR =================

@app.route("/api/clear", methods=["POST"])
def api_clear():
client = get_db()
if client.collection_exists("faces"):
client.delete_collection("faces")
ensure_collection()
if BACKUP.exists():
BACKUP.unlink()
return jsonify({"message": "Collection cleared and recreated."})

# ================= STATS =================

@app.route("/api/stats")
def api_stats():
ensure_collection()
client = get_db()

```
total = client.count("faces").count
pts, _ = client.scroll("faces", limit=200, with_payload=True, with_vectors=False)

points = [{"id": p.id, "file": p.payload.get("file","?"), "face": p.payload.get("face",0)} for p in pts]

return jsonify({"total": total, "points": points})
```

if **name** == "**main**":
app.run(debug=False, host="0.0.0.0", port=5050)
