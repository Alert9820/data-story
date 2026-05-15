"""
DataStory — Production-ready AI-style Data Storytelling Dashboard
No external AI APIs. Pure Python analytics + Flask + Plotly.
"""

import os, io, json, traceback, uuid, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy import stats as scipy_stats
from flask import Flask, jsonify, request, send_file, Response

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Numpy-safe JSON encoder — prevents bool_/int64/float64 serialization errors
import json as _json
class _NumpySafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)
app.json_provider_class = type('P', (app.json_provider_class,), {
    'dumps': lambda self, obj, **kw: _json.dumps(obj, cls=_NumpySafeEncoder, **{k:v for k,v in kw.items() if k != 'cls'}),
})
app.json = app.json_provider_class(app)

UPLOAD_DIR = Path("uploads")
CLEANED_DIR = Path("cleaned")
UPLOAD_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)

EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>DataStory — Intelligent Analytics Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root{
    --bg:#060a10;
    --s1:#0c1220;
    --s2:#101828;
    --s3:#172032;
    --border:rgba(56,189,248,0.1);
    --border2:rgba(255,255,255,0.06);
    --blue:#38bdf8;
    --indigo:#818cf8;
    --green:#34d399;
    --amber:#fbbf24;
    --pink:#f472b6;
    --text:#f1f5f9;
    --muted:#64748b;
    --muted2:#94a3b8;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}

  body{
    background:var(--bg);
    color:var(--text);
    font-family:'Syne',sans-serif;
    min-height:100vh;
    overflow-x:hidden;
  }

  /* ─── Noise texture overlay ─── */
  body::after{
    content:'';position:fixed;inset:0;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events:none;z-index:0;opacity:0.4;
  }

  /* ─── Gradient mesh bg ─── */
  .bg-mesh{
    position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;
  }
  .mesh-blob{
    position:absolute;border-radius:50%;filter:blur(140px);opacity:0.12;
    animation:drift 18s ease-in-out infinite alternate;
  }
  .mb1{width:700px;height:700px;background:#1d4ed8;top:-200px;right:-200px;animation-delay:0s}
  .mb2{width:500px;height:500px;background:#0f766e;bottom:-100px;left:-100px;animation-delay:-6s}
  .mb3{width:400px;height:400px;background:#7c3aed;top:40%;left:30%;animation-delay:-12s}

  @keyframes drift{0%{transform:translate(0,0) scale(1)}100%{transform:translate(40px,30px) scale(1.05)}}

  /* ─── Top nav ─── */
  nav{
    position:sticky;top:0;z-index:100;
    background:rgba(6,10,16,0.85);
    backdrop-filter:blur(20px);
    border-bottom:1px solid var(--border2);
    padding:0 40px;
    height:64px;
    display:flex;align-items:center;justify-content:space-between;
  }

  .logo{
    display:flex;align-items:center;gap:10px;
    font-size:1.15rem;font-weight:800;letter-spacing:-0.03em;
  }
  .logo-icon{
    width:34px;height:34px;
    background:linear-gradient(135deg,var(--blue),var(--indigo));
    border-radius:8px;display:flex;align-items:center;justify-content:center;
    font-size:16px;
  }
  .logo-text{
    background:linear-gradient(90deg,var(--blue) 0%,var(--indigo) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }
  .logo-sub{font-size:0.62rem;color:var(--muted);font-family:'JetBrains Mono',monospace;letter-spacing:0.08em}

  .nav-tabs{display:flex;gap:4px}
  .tab{
    padding:6px 16px;border-radius:8px;font-size:0.82rem;font-weight:600;
    cursor:pointer;transition:all 0.2s;color:var(--muted);
    border:1px solid transparent;
  }
  .tab:hover{color:var(--text);background:rgba(255,255,255,0.04)}
  .tab.active{
    color:var(--blue);
    background:rgba(56,189,248,0.08);
    border-color:rgba(56,189,248,0.2);
  }

  .nav-right{display:flex;align-items:center;gap:12px}
  .pill{
    display:inline-flex;align-items:center;gap:6px;
    padding:4px 12px;border-radius:100px;font-size:0.7rem;
    font-family:'JetBrains Mono',monospace;
    background:rgba(52,211,153,0.08);
    border:1px solid rgba(52,211,153,0.2);
    color:var(--green);
  }
  .dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:blink 2s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

  /* ─── Main wrapper ─── */
  .main{
    position:relative;z-index:1;
    max-width:1400px;margin:0 auto;padding:40px 40px 80px;
  }

  /* ─── Upload section ─── */
  .upload-section{
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;min-height:calc(100vh - 140px);
    gap:32px;text-align:center;
  }

  .upload-hero-title{
    font-size:clamp(2rem,5vw,3.5rem);
    font-weight:800;letter-spacing:-0.04em;line-height:1.1;
  }
  .upload-hero-title span{
    background:linear-gradient(135deg,var(--blue),var(--indigo),var(--pink));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }
  .upload-hero-sub{
    font-size:1rem;color:var(--muted2);max-width:500px;
    font-family:'JetBrains Mono',monospace;font-size:0.875rem;
    line-height:1.6;
  }

  .features-row{
    display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-bottom:8px;
  }
  .feature-chip{
    display:flex;align-items:center;gap:6px;
    padding:6px 14px;border-radius:100px;font-size:0.75rem;
    background:var(--s1);border:1px solid var(--border2);
    color:var(--muted2);
  }

  .drop-zone{
    width:100%;max-width:580px;
    border:2px dashed rgba(56,189,248,0.25);
    border-radius:24px;
    background:rgba(12,18,32,0.7);
    backdrop-filter:blur(12px);
    padding:56px 40px;
    cursor:pointer;
    transition:all 0.3s;
    position:relative;overflow:hidden;
  }
  .drop-zone::before{
    content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse at 50% 0%,rgba(56,189,248,0.06),transparent 70%);
    pointer-events:none;
  }
  .drop-zone:hover,.drop-zone.over{
    border-color:var(--blue);
    background:rgba(56,189,248,0.04);
    transform:translateY(-3px);
    box-shadow:0 20px 60px rgba(56,189,248,0.08);
  }
  .drop-zone:hover::before,.drop-zone.over::before{
    background:radial-gradient(ellipse at 50% 0%,rgba(56,189,248,0.12),transparent 70%);
  }

  .drop-icon{
    font-size:3rem;margin-bottom:18px;display:block;
    animation:levitate 3s ease-in-out infinite;
  }
  @keyframes levitate{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}

  .drop-title{font-size:1.2rem;font-weight:700;margin-bottom:8px}
  .drop-sub{font-size:0.8rem;color:var(--muted);font-family:'JetBrains Mono',monospace}

  .format-badges{display:flex;gap:8px;justify-content:center;margin-top:20px}
  .badge{
    padding:4px 14px;border-radius:6px;font-size:0.68rem;font-weight:600;
    font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;
    background:rgba(56,189,248,0.08);
    border:1px solid rgba(56,189,248,0.2);
    color:var(--blue);
  }

  .file-indicator{
    display:none;width:100%;max-width:580px;
    background:var(--s1);border:1px solid var(--border);
    border-radius:16px;padding:16px 24px;
    align-items:center;gap:14px;
  }
  .file-indicator.show{display:flex}
  .fi-icon{font-size:1.4rem}
  .fi-name{font-size:0.85rem;color:var(--blue);font-family:'JetBrains Mono',monospace;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .fi-size{font-size:0.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace}

  /* ─── Buttons ─── */
  .btn{
    display:inline-flex;align-items:center;gap:8px;
    padding:11px 24px;border-radius:12px;font-size:0.875rem;font-weight:700;
    cursor:pointer;transition:all 0.22s;border:none;
    font-family:'Syne',sans-serif;letter-spacing:-0.01em;
  }
  .btn-primary{
    background:linear-gradient(135deg,var(--blue) 0%,#0ea5e9 100%);
    color:#030c14;box-shadow:0 0 24px rgba(56,189,248,0.25);
  }
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(56,189,248,0.4)}
  .btn-primary:disabled{opacity:0.35;cursor:not-allowed;transform:none;box-shadow:none}
  .btn-ghost{
    background:var(--s2);color:var(--muted2);
    border:1px solid var(--border2);
  }
  .btn-ghost:hover{color:var(--blue);border-color:rgba(56,189,248,0.3)}

  /* ─── Dashboard ─── */
  #dashboard{display:none}
  #dashboard.show{display:block;animation:fadeUp 0.5s ease}
  @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

  /* ─── Phase headers ─── */
  .phase-header{
    display:flex;align-items:center;gap:14px;margin:48px 0 24px;
  }
  .phase-badge{
    font-size:0.6rem;font-family:'JetBrains Mono',monospace;font-weight:500;
    letter-spacing:0.15em;padding:5px 14px;border-radius:100px;
  }
  .phase-badge.data{background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.3);color:var(--amber)}
  .phase-badge.viz{background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);color:var(--blue)}
  .phase-badge.narrative{background:rgba(129,140,248,0.12);border:1px solid rgba(129,140,248,0.3);color:var(--indigo)}

  .phase-title{font-size:1.4rem;font-weight:800;letter-spacing:-0.04em}
  .phase-sep{flex:1;height:1px;background:var(--border2);margin-left:8px}

  /* ─── Stats row ─── */
  .stats-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:16px;margin-bottom:28px}

  .stat-card{
    background:var(--s1);border:1px solid var(--border2);
    border-radius:16px;padding:20px;
    position:relative;overflow:hidden;
    animation:popIn 0.4s ease both;
    transition:all 0.25s;
  }
  .stat-card::after{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--blue),var(--indigo));
  }
  .stat-card:hover{transform:translateY(-3px);border-color:var(--border);box-shadow:0 8px 32px rgba(0,0,0,0.4)}
  @keyframes popIn{from{opacity:0;transform:scale(0.94)}to{opacity:1;transform:scale(1)}}

  .stat-label{font-size:0.65rem;font-family:'JetBrains Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px}
  .stat-value{font-size:1.6rem;font-weight:800;letter-spacing:-0.04em;line-height:1}
  .stat-sub{font-size:0.7rem;color:var(--muted);margin-top:6px;font-family:'JetBrains Mono',monospace}
  .stat-blue{background:linear-gradient(135deg,var(--blue),#7dd3fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-green{background:linear-gradient(135deg,var(--green),#6ee7b7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-amber{background:linear-gradient(135deg,var(--amber),#fcd34d);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-red{background:linear-gradient(135deg,#f87171,#fca5a5);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .stat-indigo{background:linear-gradient(135deg,var(--indigo),#a5b4fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}

  /* ─── Quality score bar ─── */
  .quality-card{
    background:var(--s1);border:1px solid var(--border2);
    border-radius:16px;padding:24px 28px;
    display:flex;align-items:center;gap:24px;
    margin-bottom:28px;
    animation:popIn 0.5s 0.1s ease both;
  }
  .quality-ring{position:relative;width:80px;height:80px;flex-shrink:0}
  .ring-svg{transform:rotate(-90deg)}
  .ring-bg{fill:none;stroke:rgba(255,255,255,0.06);stroke-width:6}
  .ring-fill{fill:none;stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset 1.2s ease}
  .quality-score-num{
    position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    font-size:1rem;font-weight:800;letter-spacing:-0.03em;
    font-family:'JetBrains Mono',monospace;
  }
  .quality-score-label{font-size:0.45rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em}
  .quality-info{flex:1}
  .quality-title{font-size:1rem;font-weight:700;margin-bottom:4px}
  .quality-sub{font-size:0.78rem;color:var(--muted2);line-height:1.5}

  /* ─── KPI cards ─── */
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:20px;margin-bottom:32px}

  .kpi-card{
    background:var(--s1);border:1px solid var(--border2);
    border-radius:18px;padding:24px;
    position:relative;overflow:hidden;
    transition:all 0.25s;
    animation:popIn 0.4s ease both;
  }
  .kpi-card:hover{transform:translateY(-4px);box-shadow:0 16px 48px rgba(0,0,0,0.5)}

  .kpi-card-glow{
    position:absolute;top:-30px;right:-30px;
    width:120px;height:120px;border-radius:50%;
    filter:blur(50px);opacity:0.15;
    background:var(--blue);
  }
  .kpi-label{font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px}
  .kpi-value{font-size:2rem;font-weight:800;letter-spacing:-0.05em;line-height:1;
    background:linear-gradient(135deg,#fff 0%,var(--blue) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }
  .kpi-meta{display:flex;align-items:center;gap:10px;margin-top:12px}
  .kpi-avg{font-size:0.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace}
  .kpi-trend{
    display:inline-flex;align-items:center;gap:3px;
    padding:2px 10px;border-radius:100px;font-size:0.68rem;font-weight:700;
    font-family:'JetBrains Mono',monospace;
  }
  .kpi-trend.up{background:rgba(52,211,153,0.1);color:var(--green);border:1px solid rgba(52,211,153,0.2)}
  .kpi-trend.down{background:rgba(248,113,113,0.1);color:#f87171;border:1px solid rgba(248,113,113,0.2)}

  /* ─── Charts grid ─── */
  .charts-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:24px;margin-bottom:32px}

  .chart-card{
    background:var(--s1);border:1px solid var(--border2);
    border-radius:20px;padding:28px;
    transition:all 0.25s;
    animation:popIn 0.4s ease both;
  }
  .chart-card:hover{border-color:var(--border);box-shadow:0 8px 40px rgba(0,0,0,0.4)}
  .chart-card-label{
    font-size:0.65rem;font-family:'JetBrains Mono',monospace;
    color:var(--muted);text-transform:uppercase;letter-spacing:0.12em;
    margin-bottom:16px;display:flex;align-items:center;gap:8px;
  }
  .chart-card-label::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--blue)}

  /* ─── Narrative cards ─── */
  .narrative-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}

  .narr-card{
    background:var(--s1);border:1px solid var(--border2);
    border-radius:18px;padding:24px;
    animation:popIn 0.4s ease both;
    transition:all 0.25s;
  }
  .narr-card:hover{border-color:var(--border)}
  .narr-card.full{grid-column:1/-1}

  .narr-tag{
    display:inline-flex;align-items:center;gap:6px;
    font-size:0.62rem;font-family:'JetBrains Mono',monospace;
    text-transform:uppercase;letter-spacing:0.12em;font-weight:500;
    padding:3px 10px;border-radius:100px;margin-bottom:14px;
  }
  .narr-tag.blue{background:rgba(56,189,248,0.1);color:var(--blue);border:1px solid rgba(56,189,248,0.2)}
  .narr-tag.green{background:rgba(52,211,153,0.1);color:var(--green);border:1px solid rgba(52,211,153,0.2)}
  .narr-tag.amber{background:rgba(251,191,36,0.1);color:var(--amber);border:1px solid rgba(251,191,36,0.2)}
  .narr-tag.indigo{background:rgba(129,140,248,0.1);color:var(--indigo);border:1px solid rgba(129,140,248,0.2)}
  .narr-tag.pink{background:rgba(244,114,182,0.1);color:var(--pink);border:1px solid rgba(244,114,182,0.2)}

  .narr-heading{font-size:0.95rem;font-weight:700;letter-spacing:-0.02em;margin-bottom:12px}
  .narr-body{font-size:0.82rem;color:var(--muted2);line-height:1.75}
  .narr-body strong{color:var(--text)}
  .narr-body p{margin-bottom:8px}

  .insight-list{display:flex;flex-direction:column;gap:10px}
  .insight-item{
    background:var(--s2);border:1px solid var(--border2);
    border-radius:10px;padding:12px 14px;
    font-size:0.8rem;color:var(--muted2);line-height:1.6;
  }
  .insight-item strong{color:var(--text)}

  .rec-list{display:flex;flex-direction:column;gap:10px}
  .rec-item{
    display:flex;align-items:flex-start;gap:10px;
    background:var(--s2);border:1px solid var(--border2);
    border-radius:10px;padding:12px 14px;
    font-size:0.8rem;color:var(--muted2);line-height:1.6;
  }
  .rec-item strong{color:var(--text)}

  /* ─── Table ─── */
  .table-wrap{overflow-x:auto;max-height:380px;overflow-y:auto}
  table{width:100%;border-collapse:collapse;font-size:0.76rem;font-family:'JetBrains Mono',monospace}
  thead th{
    background:var(--s3);padding:10px 14px;text-align:left;
    color:var(--blue);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;
    position:sticky;top:0;z-index:1;white-space:nowrap;font-weight:500;
    border-bottom:1px solid var(--border);
  }
  tbody tr{border-bottom:1px solid rgba(255,255,255,0.02);transition:background 0.15s}
  tbody tr:hover{background:rgba(56,189,248,0.03)}
  tbody td{padding:9px 14px;color:var(--muted2);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  /* ─── Loading overlay ─── */
  #loading{
    position:fixed;inset:0;
    background:rgba(6,10,16,0.95);
    backdrop-filter:blur(12px);
    display:none;flex-direction:column;
    align-items:center;justify-content:center;
    z-index:9999;gap:32px;
  }
  #loading.show{display:flex}

  .loader-pulse{
    width:72px;height:72px;border-radius:50%;
    border:2px solid rgba(56,189,248,0.2);
    border-top-color:var(--blue);
    animation:spin 0.8s linear infinite;
    position:relative;
  }
  .loader-pulse::after{
    content:'';position:absolute;inset:8px;border-radius:50%;
    border:2px solid rgba(129,140,248,0.2);
    border-bottom-color:var(--indigo);
    animation:spin 1.2s linear infinite reverse;
  }
  @keyframes spin{to{transform:rotate(360deg)}}

  .loader-phases{display:flex;flex-direction:column;gap:12px;text-align:center}
  .loader-phase{
    font-size:0.8rem;font-family:'JetBrains Mono',monospace;
    color:var(--muted);transition:all 0.3s;
    display:flex;align-items:center;gap:8px;justify-content:center;
  }
  .loader-phase.active{color:var(--blue);font-weight:500}
  .loader-phase.done{color:var(--green)}
  .phase-dot{width:8px;height:8px;border-radius:50%;background:currentColor;flex-shrink:0}

  /* ─── Toast ─── */
  #toast{
    position:fixed;bottom:28px;right:28px;
    background:var(--s1);border:1px solid var(--border2);
    border-radius:14px;padding:14px 22px;
    font-size:0.82rem;z-index:9998;
    display:none;align-items:center;gap:10px;
    box-shadow:0 16px 48px rgba(0,0,0,0.6);
    animation:toastIn 0.3s ease;
  }
  #toast.show{display:flex}
  #toast.success{border-color:rgba(52,211,153,0.3)}
  #toast.error{border-color:rgba(248,113,113,0.3)}
  @keyframes toastIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

  /* ─── Export bar ─── */
  .export-bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px}

  /* ─── Scrollbar ─── */
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:3px}

  /* ─── Responsive ─── */
  @media(max-width:900px){
    nav{padding:0 20px}
    .main{padding:24px 20px 60px}
    .charts-grid{grid-template-columns:1fr}
    .narrative-grid{grid-template-columns:1fr}
    .stats-row{grid-template-columns:repeat(2,1fr)}
    .nav-tabs{display:none}
    .upload-hero-title{font-size:2rem}
  }
</style>
</head>
<body>

<!-- Background -->
<div class="bg-mesh">
  <div class="mesh-blob mb1"></div>
  <div class="mesh-blob mb2"></div>
  <div class="mesh-blob mb3"></div>
</div>

<!-- Loading -->
<div id="loading">
  <div class="loader-pulse"></div>
  <div>
    <div style="font-size:0.9rem;font-weight:700;text-align:center;margin-bottom:6px;letter-spacing:-0.02em">Analyzing your data</div>
    <div style="font-size:0.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace;text-align:center;margin-bottom:20px">Three-phase storytelling pipeline running...</div>
  </div>
  <div class="loader-phases">
    <div class="loader-phase" id="lp1"><span class="phase-dot"></span> Phase 1 · Data Cleaning &amp; Quality</div>
    <div class="loader-phase" id="lp2"><span class="phase-dot"></span> Phase 2 · Visualization Engine</div>
    <div class="loader-phase" id="lp3"><span class="phase-dot"></span> Phase 3 · Narrative Generation</div>
  </div>
</div>

<!-- Toast -->
<div id="toast"></div>

<!-- Nav -->
<nav>
  <div class="logo">
    <div class="logo-icon">◈</div>
    <div>
      <div class="logo-text">DataStory</div>
      <div class="logo-sub">INTELLIGENT ANALYTICS</div>
    </div>
  </div>
  <div class="nav-tabs" id="nav-tabs" style="display:none">
    <div class="tab active" onclick="scrollTo('phase-data')">① Data</div>
    <div class="tab" onclick="scrollTo('phase-viz')">② Visualizations</div>
    <div class="tab" onclick="scrollTo('phase-narr')">③ Narrative</div>
    <div class="tab" onclick="scrollTo('phase-preview')">Preview</div>
  </div>
  <div class="nav-right">
    <div class="pill"><span class="dot"></span> System Online</div>
    <button class="btn btn-ghost" style="font-size:0.78rem;padding:7px 16px" onclick="resetApp()">↩ New File</button>
  </div>
</nav>

<!-- Main -->
<div class="main">

  <!-- Upload Section -->
  <div id="upload-section" class="upload-section">
    <div>
      <div class="upload-hero-title">Turn raw data into<br/><span>compelling stories</span></div>
    </div>
    <div class="upload-hero-sub">Upload any CSV or Excel file — DataStory automatically cleans, visualizes, and narrates your data through 3 intelligent phases.</div>
    <div class="features-row">
      <div class="feature-chip">① Data Cleaning</div>
      <div class="feature-chip">② Auto Visualizations</div>
      <div class="feature-chip">③ Smart Narrative</div>
    </div>

    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-in').click()">
      <span class="drop-icon">📊</span>
      <div class="drop-title">Drop your dataset here</div>
      <div class="drop-sub">or click to browse files · max 50 MB</div>
      <div class="format-badges">
        <span class="badge">CSV</span>
        <span class="badge">XLSX</span>
        <span class="badge">XLS</span>
      </div>
    </div>

    <input type="file" id="file-in" accept=".csv,.xlsx,.xls" style="display:none"/>

    <div class="file-indicator" id="file-ind">
      <span class="fi-icon">📄</span>
      <span class="fi-name" id="fi-name">—</span>
      <span class="fi-size" id="fi-size"></span>
      <button class="btn btn-primary" id="analyze-btn" onclick="runAnalysis()">▶ Analyze</button>
    </div>
  </div>

  <!-- Dashboard -->
  <div id="dashboard">

    <!-- ── PHASE 1: DATA ── -->
    <div id="phase-data">
      <div class="phase-header">
        <div class="phase-badge data">PHASE 01</div>
        <div class="phase-title">Data Quality & Cleaning</div>
        <div class="phase-sep"></div>
      </div>

      <!-- Stats row -->
      <div class="stats-row" id="stats-row"></div>

      <!-- Quality score card -->
      <div class="quality-card" id="quality-card"></div>

      <!-- KPI cards -->
      <div class="kpi-grid" id="kpi-grid"></div>
    </div>

    <!-- ── PHASE 2: VISUALIZATIONS ── -->
    <div id="phase-viz">
      <div class="phase-header">
        <div class="phase-badge viz">PHASE 02</div>
        <div class="phase-title">Interactive Visualizations</div>
        <div class="phase-sep"></div>
      </div>
      <div class="charts-grid" id="charts-grid"></div>
    </div>

    <!-- ── PHASE 3: NARRATIVE ── -->
    <div id="phase-narr">
      <div class="phase-header">
        <div class="phase-badge narrative">PHASE 03</div>
        <div class="phase-title">Business Narrative & Insights</div>
        <div class="phase-sep"></div>
      </div>
      <div class="narrative-grid" id="narrative-grid"></div>
    </div>

    <!-- ── DATA PREVIEW ── -->
    <div id="phase-preview">
      <div class="phase-header">
        <div class="phase-badge data">PREVIEW</div>
        <div class="phase-title">Cleaned Dataset</div>
        <div class="phase-sep"></div>
      </div>
      <div class="export-bar">
        <button class="btn btn-ghost" onclick="downloadCleaned()">↓ Download Cleaned CSV</button>
      </div>
      <div class="narr-card" style="padding:0;overflow:hidden">
        <div style="padding:16px 24px;border-bottom:1px solid var(--border2);display:flex;align-items:center;justify-content:space-between">
          <span id="table-meta" style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:var(--muted)">—</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead id="tbl-head"></thead>
            <tbody id="tbl-body"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div><!-- /dashboard -->
</div><!-- /main -->

<script>
// ── State ────────────────────────────────────────────────────────────────────
const S = { file: null, cleanedFile: null, ready: false };

// ── Drag & Drop ──────────────────────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) pickFile(f);
});
document.getElementById('file-in').addEventListener('change', e => {
  if (e.target.files[0]) pickFile(e.target.files[0]);
});

function pickFile(f) {
  const ext = f.name.split('.').pop().toLowerCase();
  if (!['csv','xlsx','xls'].includes(ext)) { toast('Only CSV and Excel files are supported', 'error'); return; }
  S.file = f;
  document.getElementById('fi-name').textContent = f.name;
  document.getElementById('fi-size').textContent = (f.size / 1024 / 1024).toFixed(2) + ' MB';
  document.getElementById('file-ind').classList.add('show');
  toast('File ready — click Analyze to begin', 'success');
}

// ── Analysis ──────────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!S.file) { toast('Please select a file first', 'error'); return; }

  showLoading(true);
  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;

  const fd = new FormData();
  fd.append('file', S.file);

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');

    S.cleanedFile = data.cleaned_file;
    S.ready = true;

    render(data);
    showLoading(false);
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('dashboard').classList.add('show');
    document.getElementById('nav-tabs').style.display = 'flex';
    toast('Analysis complete! 3 phases generated.', 'success');
  } catch(err) {
    showLoading(false);
    btn.disabled = false;
    toast('Error: ' + err.message, 'error');
  }
}

// ── Render Pipeline ───────────────────────────────────────────────────────────
function render(data) {
  renderStats(data.clean_report, data.original_shape);
  renderQuality(data.clean_report);
  renderKPIs(data.kpis);
  renderCharts(data.charts);
  renderNarrative(data.narrative);
  renderTable(data.preview, data.original_name);
}

// ── Phase 1: Stats ────────────────────────────────────────────────────────────
function renderStats(cr, orig) {
  const stats = [
    { label:'Rows (Before)', val: orig[0].toLocaleString(), cls:'stat-blue', sub:'Original records' },
    { label:'Rows (After)', val: cr.rows_after.toLocaleString(), cls:'stat-green', sub:`${cr.rows_removed} removed` },
    { label:'Columns', val: cr.cols_after, cls:'stat-indigo', sub:`of ${cr.cols_before} original` },
    { label:'Duplicates', val: cr.duplicates_removed, cls:'stat-amber', sub:'Removed & deduped' },
    { label:'Missing Fixed', val: cr.missing_filled.toLocaleString(), cls:'stat-green', sub:'Imputed via stats' },
    { label:'Outliers', val: cr.outliers_detected.toLocaleString(), cls:'stat-red', sub:`In ${(cr.outlier_cols||[]).length} column(s)` },
    { label:'Date Columns', val: (cr.date_columns||[]).length, cls:'stat-indigo', sub:'Auto-detected' },
  ];
  document.getElementById('stats-row').innerHTML = stats.map((s,i)=>`
    <div class="stat-card" style="animation-delay:${i*0.06}s">
      <div class="stat-label">${s.label}</div>
      <div class="stat-value ${s.cls}">${s.val}</div>
      <div class="stat-sub">${s.sub}</div>
    </div>`).join('');
}

// ── Quality Ring ──────────────────────────────────────────────────────────────
function renderQuality(cr) {
  const score = cr.quality_score;
  const label = score >= 85 ? 'Excellent' : score >= 65 ? 'Good' : score >= 45 ? 'Fair' : 'Poor';
  const color = score >= 85 ? '#34d399' : score >= 65 ? '#38bdf8' : score >= 45 ? '#fbbf24' : '#f87171';
  const r = 34, circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;

  document.getElementById('quality-card').innerHTML = `
    <div class="quality-ring">
      <svg class="ring-svg" width="80" height="80" viewBox="0 0 80 80">
        <circle class="ring-bg" cx="40" cy="40" r="${r}"/>
        <circle class="ring-fill" cx="40" cy="40" r="${r}"
          stroke="${color}"
          stroke-dasharray="${circ}"
          stroke-dashoffset="${circ - dash}"
        />
      </svg>
      <div class="quality-score-num" style="color:${color}">
        ${score}%<div class="quality-score-label">quality</div>
      </div>
    </div>
    <div class="quality-info">
      <div class="quality-title">Dataset Quality: <span style="color:${color}">${label}</span></div>
      <div class="quality-sub">
        Completeness, deduplication, and outlier handling scored automatically. 
        ${score >= 85 ? 'This dataset is production-ready for analytics and modeling.' :
          score >= 65 ? 'Dataset is clean enough for most analytical purposes.' :
          'Consider enriching the source data for more reliable insights.'}
      </div>
    </div>`;
}

// ── KPIs ──────────────────────────────────────────────────────────────────────
function renderKPIs(kpis) {
  if (!kpis || !kpis.length) {
    document.getElementById('kpi-grid').innerHTML = '<div style="color:var(--muted);font-size:0.82rem;padding:16px">No numeric KPIs found in this dataset.</div>';
    return;
  }
  document.getElementById('kpi-grid').innerHTML = kpis.map((k,i)=>`
    <div class="kpi-card" style="animation-delay:${i*0.08}s">
      <div class="kpi-card-glow"></div>
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-meta">
        <span class="kpi-avg">avg ${k.avg}</span>
        <span class="kpi-trend ${k.trend_up ? 'up':'down'}">
          ${k.trend_up ? '▲':'▼'} ${Math.abs(k.trend).toFixed(1)}%
        </span>
      </div>
    </div>`).join('');
}

// ── Charts ────────────────────────────────────────────────────────────────────
const CHART_NAMES = {
  bar:'Bar Chart',line:'Line / Trend',pie:'Pie / Donut',
  histogram:'Histogram',scatter:'Scatter Plot',heatmap:'Correlation Matrix',box:'Box Plot'
};

function renderCharts(charts) {
  const keys = Object.keys(charts);
  if (!keys.length) {
    document.getElementById('charts-grid').innerHTML = '<div style="color:var(--muted);font-size:0.82rem;padding:16px">Not enough data to generate charts.</div>';
    return;
  }
  document.getElementById('charts-grid').innerHTML = keys.map((k,i)=>`
    <div class="chart-card" style="animation-delay:${i*0.08}s">
      <div class="chart-card-label">${CHART_NAMES[k]||k}</div>
      <div id="chart-${k}" style="height:300px"></div>
    </div>`).join('');

  const cfg = { responsive:true, displayModeBar:true, displaylogo:false,
    modeBarButtonsToRemove:['select2d','lasso2d','toImage'] };
  setTimeout(() => {
    keys.forEach(k => {
      const c = charts[k];
      Plotly.react('chart-'+k, c.data, c.layout, cfg);
    });
  }, 100);
}

// ── Narrative ─────────────────────────────────────────────────────────────────
function mdBold(t) {
  return t.replace(/\\*\\*(.*?)\\*\\*/g,'<strong>$1</strong>');
}

function renderNarrative(n) {
  const g = document.getElementById('narrative-grid');
  let html = '';

  // Executive Summary
  html += `
    <div class="narr-card full" style="animation-delay:0s">
      <div class="narr-tag blue">Executive Summary</div>
      <div class="narr-heading">Business Overview</div>
      <div class="narr-body"><p>${mdBold(n.executive_summary||'')}</p></div>
    </div>`;

  // Numeric Insights
  if (n.numeric_insights && n.numeric_insights.length) {
    html += `
      <div class="narr-card" style="animation-delay:0.06s">
        <div class="narr-tag green">Metric Analysis</div>
        <div class="narr-heading">Numeric KPI Breakdown</div>
        <div class="insight-list">
          ${n.numeric_insights.map(i=>`<div class="insight-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Category Insights
  if (n.category_insights && n.category_insights.length) {
    html += `
      <div class="narr-card" style="animation-delay:0.1s">
        <div class="narr-tag amber">Category Intelligence</div>
        <div class="narr-heading">Segment Performance</div>
        <div class="insight-list">
          ${n.category_insights.map(i=>`<div class="insight-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Trends & Correlations
  if (n.trend_insights && n.trend_insights.length) {
    html += `
      <div class="narr-card" style="animation-delay:0.14s">
        <div class="narr-tag indigo">Trend & Correlation</div>
        <div class="narr-heading">Statistical Relationships</div>
        <div class="insight-list">
          ${n.trend_insights.map(i=>`<div class="insight-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Top / Bottom Performers
  if (n.performers && n.performers.length) {
    html += `
      <div class="narr-card" style="animation-delay:0.18s">
        <div class="narr-tag pink">Performance Ranking</div>
        <div class="narr-heading">Top & Bottom Segments</div>
        <div class="insight-list">
          ${n.performers.map(i=>`<div class="insight-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Anomalies
  if (n.anomalies && n.anomalies.length) {
    html += `
      <div class="narr-card" style="animation-delay:0.22s">
        <div class="narr-tag amber">Anomaly Detection</div>
        <div class="narr-heading">Statistical Outliers & Alerts</div>
        <div class="insight-list">
          ${n.anomalies.map(i=>`<div class="insight-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  // Recommendations
  if (n.recommendations && n.recommendations.length) {
    html += `
      <div class="narr-card full" style="animation-delay:0.26s">
        <div class="narr-tag green">Recommendations</div>
        <div class="narr-heading">Actionable Business Recommendations</div>
        <div class="rec-list">
          ${n.recommendations.map(i=>`<div class="rec-item">${mdBold(i)}</div>`).join('')}
        </div>
      </div>`;
  }

  g.innerHTML = html;
}

// ── Table ─────────────────────────────────────────────────────────────────────
function renderTable(preview, name) {
  document.getElementById('table-meta').textContent =
    `${name} · ${preview.rows.length} rows shown · ${preview.columns.length} columns`;
  document.getElementById('tbl-head').innerHTML =
    `<tr>${preview.columns.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  document.getElementById('tbl-body').innerHTML =
    preview.rows.map(r=>`<tr>${r.map(c=>`<td title="${c}">${c}</td>`).join('')}</tr>`).join('');
}

// ── Scroll Nav ────────────────────────────────────────────────────────────────
function scrollTo(id) {
  document.getElementById(id).scrollIntoView({ behavior:'smooth', block:'start' });
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  event.target.classList.add('active');
}

// ── Downloads ─────────────────────────────────────────────────────────────────
function downloadCleaned() {
  if (!S.cleanedFile) { toast('Run analysis first', 'error'); return; }
  window.location.href = '/download/' + S.cleanedFile;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetApp() {
  S.file = null; S.cleanedFile = null; S.ready = false;
  document.getElementById('upload-section').style.display = 'flex';
  document.getElementById('dashboard').classList.remove('show');
  document.getElementById('nav-tabs').style.display = 'none';
  document.getElementById('file-ind').classList.remove('show');
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('file-in').value = '';
}

// ── Loading ───────────────────────────────────────────────────────────────────
let _phaseTimer;
function showLoading(on) {
  document.getElementById('loading').classList.toggle('show', on);
  if (on) {
    const phases = ['lp1','lp2','lp3'];
    phases.forEach(p => {
      const el = document.getElementById(p);
      el.classList.remove('active','done');
    });
    let i = 0;
    _phaseTimer = setInterval(() => {
      if (i > 0) { document.getElementById(phases[i-1]).classList.remove('active'); document.getElementById(phases[i-1]).classList.add('done'); }
      if (i < phases.length) { document.getElementById(phases[i]).classList.add('active'); i++; }
      else clearInterval(_phaseTimer);
    }, 2200);
  } else {
    clearInterval(_phaseTimer);
    ['lp1','lp2','lp3'].forEach(p => {
      document.getElementById(p).classList.remove('active');
      document.getElementById(p).classList.add('done');
    });
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastT;
function toast(msg, type='info') {
  const el = document.getElementById('toast');
  const icons = { success:'✓', error:'✕', info:'ℹ' };
  el.innerHTML = `<span>${icons[type]||'ℹ'}</span><span>${msg}</span>`;
  el.className = `show ${type}`;
  clearTimeout(_toastT);
  _toastT = setTimeout(() => { el.className = ''; }, 4000);
}
</script>
</body>
</html>
"""

ALLOWED_EXT = {"csv", "xlsx", "xls"}

# ── Plotly Theme ──────────────────────────────────────────────────────────────
PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#f472b6", "#a78bfa", "#fbbf24"]
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.04)"
FONT = dict(family="'Syne', sans-serif", color="#cbd5e1", size=12)


def _theme(fig, title="") -> dict:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=FONT,
        colorway=PALETTE,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f8fafc")),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, showline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, showline=False, tickfont=dict(size=10))
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


# ── File Helpers ──────────────────────────────────────────────────────────────
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _load(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", errors="replace")
    return pd.read_excel(path)


# ── Phase 1 — Data Cleaning ───────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    before_rows, before_cols = len(df), len(df.columns)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Drop fully empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Duplicates
    dups = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = dups

    # Detect & convert date columns
    date_cols_found = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(30)
        try:
            conv = pd.to_datetime(sample, errors="coerce")
            if conv.notna().sum() >= len(sample) * 0.75:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                date_cols_found.append(col)
        except Exception:
            pass
    report["date_columns"] = date_cols_found

    # String-encoded numerics
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.replace(r"[\$,€£₹%]", "", regex=True).str.strip()
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().sum() > len(df) * 0.6:
            df[col] = num

    # Missing values
    missing_total = int(df.isnull().sum().sum())
    for col in df.columns:
        if col in date_cols_found:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == object:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else "Unknown")
    report["missing_filled"] = missing_total

    # Outlier detection (IQR capping)
    outlier_cols, total_outliers = [], 0
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            df[col] = df[col].clip(lo, hi)
            outlier_cols.append(col)
            total_outliers += n
    report["outliers_detected"] = total_outliers
    report["outlier_cols"] = outlier_cols

    report["rows_before"] = before_rows
    report["rows_after"] = len(df)
    report["cols_before"] = before_cols
    report["cols_after"] = len(df.columns)
    report["rows_removed"] = before_rows - len(df)

    # Quality score (0-100)
    completeness = max(0, 1 - missing_total / max(before_rows * before_cols, 1))
    dup_penalty = max(0, 1 - dups / max(before_rows, 1))
    outlier_penalty = max(0, 1 - total_outliers / max(before_rows * 2, 1))
    report["quality_score"] = round((completeness * 0.5 + dup_penalty * 0.3 + outlier_penalty * 0.2) * 100, 1)

    return df, report


# ── Column Detection ──────────────────────────────────────────────────────────
def detect_columns(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include="object").columns.tolist()
    date = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "date": date,
        "best_num": numeric[0] if numeric else None,
        "best_cat": categorical[0] if categorical else None,
        "best_date": date[0] if date else None,
    }


# ── Phase 2 — Visualizations ──────────────────────────────────────────────────
def generate_visualizations(df: pd.DataFrame) -> dict:
    cols = detect_columns(df)
    num = cols["numeric"]
    cat = cols["categorical"]
    date = cols["date"]
    charts = {}

    # 1. Bar chart
    if cols["best_cat"] and cols["best_num"]:
        agg = df.groupby(cols["best_cat"])[cols["best_num"]].sum().nlargest(12).reset_index()
        fig = px.bar(
            agg, x=cols["best_cat"], y=cols["best_num"],
            color=cols["best_num"], color_continuous_scale="Blues",
            text=cols["best_num"],
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                          marker_line_width=0, opacity=0.9)
        charts["bar"] = _theme(fig, f"Top {cols['best_cat'].replace('_',' ').title()} by {cols['best_num'].replace('_',' ').title()}")

    # 2. Line / trend chart
    if date and cols["best_num"]:
        ldf = df.sort_values(cols["best_date"])
        fig = px.line(ldf, x=cols["best_date"], y=cols["best_num"], markers=True,
                      color_discrete_sequence=["#38bdf8"])
        fig.update_traces(line_width=2.5, marker_size=5)
        charts["line"] = _theme(fig, f"{cols['best_num'].replace('_',' ').title()} Trend Over Time")
    elif cols["best_cat"] and cols["best_num"]:
        grp = df.groupby(cols["best_cat"])[cols["best_num"]].mean().reset_index()
        fig = px.line(grp, x=cols["best_cat"], y=cols["best_num"], markers=True,
                      color_discrete_sequence=["#38bdf8"])
        fig.update_traces(line_width=2.5, marker_size=6)
        charts["line"] = _theme(fig, f"Avg {cols['best_num'].replace('_',' ').title()} Across Categories")

    # 3. Pie / Donut
    if cols["best_cat"] and cols["best_num"]:
        pdf = df.groupby(cols["best_cat"])[cols["best_num"]].sum().nlargest(8).reset_index()
        fig = px.pie(pdf, names=cols["best_cat"], values=cols["best_num"],
                     hole=0.5, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker_line_width=2, marker_line_color="rgba(0,0,0,0.3)")
        charts["pie"] = _theme(fig, f"Share of {cols['best_num'].replace('_',' ').title()}")

    # 4. Histogram
    if cols["best_num"]:
        fig = px.histogram(df, x=cols["best_num"], nbins=30,
                           color_discrete_sequence=["#818cf8"])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        charts["histogram"] = _theme(fig, f"Distribution of {cols['best_num'].replace('_',' ').title()}")

    # 5. Scatter
    if len(num) >= 2:
        color_col = cat[0] if cat else None
        sdf = df.sample(min(500, len(df)), random_state=42)
        fig = px.scatter(sdf, x=num[0], y=num[1], color=color_col,
                         opacity=0.72, size_max=8,
                         color_discrete_sequence=PALETTE)
        fig.update_traces(marker_size=6)
        charts["scatter"] = _theme(fig, f"{num[0].replace('_',' ').title()} vs {num[1].replace('_',' ').title()}")

    # 6. Correlation heatmap
    if len(num) >= 3:
        corr = df[num[:10]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_traces(textfont_size=10)
        charts["heatmap"] = _theme(fig, "Correlation Matrix")

    # 7. Box plot for spread analysis
    if cols["best_num"] and cols["best_cat"]:
        bdf = df.copy()
        top_cats = bdf[cols["best_cat"]].value_counts().nlargest(6).index
        bdf = bdf[bdf[cols["best_cat"]].isin(top_cats)]
        fig = px.box(bdf, x=cols["best_cat"], y=cols["best_num"],
                     color=cols["best_cat"], color_discrete_sequence=PALETTE)
        fig.update_traces(boxmean=True)
        charts["box"] = _theme(fig, f"{cols['best_num'].replace('_',' ').title()} Distribution by Category")

    return charts


# ── Phase 3 — Narrative (Pure Python) ────────────────────────────────────────
def generate_narrative(df: pd.DataFrame, clean_report: dict) -> dict:
    cols = detect_columns(df)
    num = cols["numeric"]
    cat = cols["categorical"]
    date = cols["date"]
    narrative = {}

    # ── Executive Summary
    quality = clean_report.get("quality_score", 0)
    quality_label = "Excellent" if quality >= 85 else "Good" if quality >= 65 else "Fair"
    rows, total_cols = clean_report["rows_after"], clean_report["cols_after"]

    exec_lines = [
        f"This dataset comprises **{rows:,} records** across **{total_cols} dimensions** — cleaned and validated with a data quality score of **{quality}% ({quality_label})**.",
        f"During preprocessing, **{clean_report['duplicates_removed']} duplicate entries** were eliminated and **{clean_report['missing_filled']} missing values** were imputed using statistical methods.",
    ]
    if clean_report["outliers_detected"]:
        exec_lines.append(f"**{clean_report['outliers_detected']} outliers** across {len(clean_report['outlier_cols'])} columns were identified and capped using the IQR method to ensure robust analysis.")
    if date:
        exec_lines.append(f"Time-series analysis is enabled through **{len(date)} date column(s)**: {', '.join(date)}.")
    narrative["executive_summary"] = " ".join(exec_lines)

    # ── Numeric Insights
    num_insights = []
    for col in num[:4]:
        series = df[col].dropna()
        mean, med, std = series.mean(), series.median(), series.std()
        skew = float(series.skew())
        top_val = series.max()
        skew_label = "positively skewed (right tail — high outliers exist)" if skew > 0.5 else \
                     "negatively skewed (left tail — low values dominate)" if skew < -0.5 else \
                     "approximately normally distributed"
        pct_above_mean = (series > mean).mean() * 100

        insight = f"**{col.replace('_',' ').title()}**: Mean = {mean:,.2f}, Median = {med:,.2f}, Std Dev = {std:,.2f}. "
        insight += f"The distribution is {skew_label}. "
        insight += f"{pct_above_mean:.1f}% of records exceed the average, with a peak value of {top_val:,.2f}."
        num_insights.append(insight)
    narrative["numeric_insights"] = num_insights

    # ── Category Performance
    cat_insights = []
    for col in cat[:3]:
        vc = df[col].value_counts()
        top_cat, top_count = vc.index[0], int(vc.iloc[0])
        bottom_cat, bottom_count = vc.index[-1], int(vc.iloc[-1])
        total = len(df)
        share = top_count / total * 100
        n_unique = int(vc.shape[0])

        insight = f"**{col.replace('_',' ').title()}** has **{n_unique} unique categories**. "
        insight += f"The dominant category is **'{top_cat}'** with {top_count:,} records ({share:.1f}% share). "
        if n_unique > 1:
            insight += f"The least represented is **'{bottom_cat}'** ({bottom_count:,} records). "
        if num and n_unique <= 20:
            try:
                agg = df.groupby(col)[num[0]].sum()
                top_performer = agg.idxmax()
                top_val = agg.max()
                insight += f"By {num[0].replace('_',' ')}, **'{top_performer}'** leads with {top_val:,.2f}."
            except Exception:
                pass
        cat_insights.append(insight)
    narrative["category_insights"] = cat_insights

    # ── Trend & Correlation Insights
    trend_insights = []

    if len(num) >= 2:
        corr_matrix = df[num].corr()
        pairs = []
        for i in range(len(num)):
            for j in range(i + 1, len(num)):
                c = corr_matrix.iloc[i, j]
                pairs.append((num[i], num[j], c))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        for a, b, c in pairs[:3]:
            if abs(c) >= 0.4:
                strength = "strong" if abs(c) >= 0.7 else "moderate"
                direction = "positive" if c > 0 else "negative"
                trend_insights.append(
                    f"**{a.replace('_',' ').title()}** and **{b.replace('_',' ').title()}** share a {strength} {direction} correlation (r = {c:.2f}). "
                    + (f"As {a.replace('_',' ')} increases, {b.replace('_',' ')} tends to {'increase' if c > 0 else 'decrease'} proportionally."
                       if abs(c) >= 0.6 else "")
                )

    if date and num:
        try:
            ts = df.sort_values(cols["best_date"])[[cols["best_date"], num[0]]].dropna()
            ts = ts.set_index(cols["best_date"]).resample("ME").sum()[num[0]].dropna()
            if len(ts) >= 3:
                x = np.arange(len(ts))
                slope, _, r, p, _ = scipy_stats.linregress(x, ts.values)
                direction = "upward" if slope > 0 else "downward"
                significance = "statistically significant" if p < 0.05 else "not statistically significant"
                trend_insights.append(
                    f"Monthly time-series analysis of **{num[0].replace('_',' ').title()}** reveals a **{direction} trend** (slope = {slope:,.2f}/month, p={p:.3f}). "
                    f"This trend is {significance} at the 95% confidence level."
                )
        except Exception:
            pass

    narrative["trend_insights"] = trend_insights if trend_insights else [
        "Insufficient numeric columns for correlation analysis. Enrich the dataset with more quantitative dimensions."
    ]

    # ── Top / Bottom Performers
    performers = []
    if cols["best_cat"] and cols["best_num"]:
        try:
            agg = df.groupby(cols["best_cat"])[cols["best_num"]].agg(["sum", "mean", "count"])
            agg.columns = ["total", "avg", "count"]
            agg = agg.sort_values("total", ascending=False)
            top3 = agg.head(3)
            bot3 = agg.tail(3)

            top_str = ", ".join([f"**{idx}** ({row['total']:,.2f})" for idx, row in top3.iterrows()])
            bot_str = ", ".join([f"**{idx}** ({row['total']:,.2f})" for idx, row in bot3.iterrows()])

            performers.append(f"🏆 **Top Performers** by {cols['best_num'].replace('_',' ')}: {top_str}")
            performers.append(f"⚠️ **Lowest Performers**: {bot_str} — these segments warrant attention or strategic investment.")
        except Exception:
            pass
    narrative["performers"] = performers

    # ── Statistical Anomalies
    anomalies = []
    for col in num[:3]:
        series = df[col].dropna()
        z = np.abs(scipy_stats.zscore(series))
        extreme = int((z > 3).sum())
        if extreme > 0:
            anomalies.append(f"**{col.replace('_',' ').title()}**: {extreme} data points fall beyond 3 standard deviations — flag for manual review.")
    if not anomalies:
        anomalies.append("No extreme statistical anomalies detected. Dataset passes the Z-score sanity check (no values beyond ±3σ).")
    narrative["anomalies"] = anomalies

    # ── Recommendations
    recs = []
    if clean_report["duplicates_removed"] > rows * 0.05:
        recs.append("🔍 **Duplicate Rate Alert**: Over 5% of records were duplicates. Audit your data ingestion pipeline for deduplication logic.")
    if clean_report["missing_filled"] > rows * total_cols * 0.1:
        recs.append("📋 **High Missingness**: More than 10% of values were missing. Consider improving data collection at the source.")
    if len(num) >= 2:
        recs.append(f"📈 **Predictive Modeling**: With {len(num)} numeric features, this dataset is ready for regression or classification models using `{num[0]}` as the target variable.")
    if cat:
        recs.append(f"🎯 **Segmentation Opportunity**: Leverage `{cat[0]}` for cohort analysis and targeted business strategy.")
    if date:
        recs.append("📅 **Time-Series Forecasting**: Date columns detected. Consider building ARIMA or Prophet forecasting models for future trend prediction.")
    recs.append("🔄 **Refresh Cadence**: Establish a weekly data refresh pipeline to keep insights current and actionable.")
    narrative["recommendations"] = recs

    return narrative


# ── KPI Generator ─────────────────────────────────────────────────────────────
def generate_kpis(df: pd.DataFrame) -> list:
    kpis = []
    for col in df.select_dtypes(include=[np.number]).columns[:4]:
        s = df[col].dropna()
        total = s.sum()
        mean = s.mean()
        # Trend: compare first half vs second half
        half = len(s) // 2
        if half > 0:
            trend = ((s.iloc[half:].mean() - s.iloc[:half].mean()) / (s.iloc[:half].mean() + 1e-9)) * 100
        else:
            trend = 0
        kpis.append({
            "label": col.replace("_", " ").title(),
            "value": f"{float(total):,.1f}" if float(total) < 1e7 else f"{float(total)/1e6:,.2f}M",
            "avg": f"{float(mean):,.2f}",
            "trend": float(round(float(trend), 1)),
            "trend_up": bool(trend >= 0),
        })
    return kpis


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return Response(html_path.read_text(encoding="utf-8"), mimetype="text/html")
    # Fallback: embedded HTML (used when templates/ folder missing on deploy)
    return Response(EMBEDDED_HTML, mimetype="text/html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
        if not _allowed(file.filename):
            return jsonify({"error": "Only CSV and Excel files accepted"}), 400

        uid = uuid.uuid4().hex[:10]
        ext = file.filename.rsplit(".", 1)[1].lower()
        fname = f"{uid}.{ext}"
        raw_path = UPLOAD_DIR / fname
        file.save(raw_path)

        # Load
        df_raw = _load(raw_path)
        original_shape = df_raw.shape

        # Phase 1 — Clean
        df, clean_report = clean_data(df_raw.copy())

        # Save cleaned
        cleaned_path = CLEANED_DIR / f"cleaned_{uid}.csv"
        df.to_csv(cleaned_path, index=False)

        # Phase 2 — Visualizations
        charts = generate_visualizations(df)

        # KPIs
        kpis = generate_kpis(df)

        # Phase 3 — Narrative
        narrative = generate_narrative(df, clean_report)

        # Preview (first 100 rows)
        preview_df = df.head(100).fillna("").astype(str)

        return jsonify({
            "success": True,
            "original_name": file.filename,
            "original_shape": list(original_shape),
            "clean_report": clean_report,
            "charts": charts,
            "kpis": kpis,
            "narrative": narrative,
            "cleaned_file": f"cleaned_{uid}.csv",
            "preview": {
                "columns": preview_df.columns.tolist(),
                "rows": preview_df.values.tolist(),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    path = CLEANED_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
