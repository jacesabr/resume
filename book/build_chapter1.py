#!/usr/bin/env python3
# build_chapter1.py — assembles the enhanced chapter1.html
import os

OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

# ── helpers ──────────────────────────────────────────────────────────────────
def kw(s): return f'<span class="kw">{s}</span>'
def fn(s): return f'<span class="fn">{s}</span>'
def st(s): return f'<span class="st">{s}</span>'
def cm(s): return f'<span class="cm">{s}</span>'
def lm(s): return f'<span class="lm">{s}</span>'
def nu(s): return f'<span class="nu">{s}</span>'
def cl(s): return f'<span class="cl">{s}</span>'
def bi(s): return f'<span class="bi">{s}</span>'
def pr(s): return f'<span class="pr">{s}</span>'

# ─────────────────────────────────────────────────────────────────────────────
HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chapter 1 \u2014 Understanding Large Language Models</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#faf9f7; --bg-code:#1d1d1d; --bg-card:#fff; --bg-aside:#f3f1ee;
  --text:#1a1a1a; --text-secondary:#6b6560; --text-code:#e8e4de;
  --accent:#c96442; --accent-light:rgba(201,100,66,.08);
  --blue:#3b6bb5; --blue-light:rgba(59,107,181,.06);
  --green:#4a8c6f; --green-light:rgba(74,140,111,.06);
  --purple:#7c5cbf; --amber:#b87c2a;
  --border:#e8e4de; --border-light:#f0ece6;
  --font-serif:'Instrument Serif',Georgia,serif;
  --font-sans:-apple-system,'Segoe UI',system-ui,sans-serif;
  --font-mono:'JetBrains Mono','SF Mono',monospace;
  --max-w:740px;
}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:16.5px;line-height:1.8;-webkit-font-smoothing:antialiased;}
::selection{background:var(--accent);color:#fff;}
.container{max-width:var(--max-w);margin:0 auto;padding:0 24px;}
.topnav{position:sticky;top:0;z-index:100;background:rgba(250,249,247,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--border-light);padding:14px 0;}
.topnav .container{display:flex;align-items:center;justify-content:space-between;}
.topnav-title{font-family:var(--font-serif);font-size:15px;}
.topnav-chapter{font-size:13px;color:var(--text-secondary);}
.chapter-header{padding:80px 0 60px;border-bottom:1px solid var(--border);}
.chapter-label{font-size:13px;font-weight:500;letter-spacing:1px;text-transform:uppercase;color:var(--accent);margin-bottom:16px;}
.chapter-header h1{font-family:var(--font-serif);font-size:clamp(30px,5vw,44px);font-weight:400;line-height:1.2;margin-bottom:16px;}
.chapter-header>p{font-size:17px;color:var(--text-secondary);max-width:580px;line-height:1.7;}
.toc{padding:32px 0;border-bottom:1px solid var(--border);}
.toc h3{font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:16px;}
.toc-list{list-style:none;display:flex;flex-direction:column;gap:4px;}
.toc-list a{display:flex;align-items:baseline;gap:12px;text-decoration:none;color:var(--text);padding:5px 0;font-size:14px;transition:color .2s;}
.toc-list a:hover{color:var(--accent);}
.toc-num{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);min-width:30px;}
.section{padding:56px 0 0;}
.section-label{font-family:var(--font-mono);font-size:11px;color:var(--accent);letter-spacing:.5px;margin-bottom:8px;}
.section h2{font-family:var(--font-serif);font-size:28px;font-weight:400;line-height:1.3;margin-bottom:20px;}
.section p{margin-bottom:18px;}
.section-divider{border:none;border-top:1px solid var(--border-light);margin:60px 0 0;}
.callout{background:var(--bg-aside);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:18px 22px;margin:24px 0;font-size:15px;line-height:1.7;}
.callout strong{color:var(--accent);}
.callout.blue{border-left-color:var(--blue);}
.callout.blue strong{color:var(--blue);}
.callout.green{border-left-color:var(--green);}
.callout.green strong{color:var(--green);}
.callout.purple{border-left-color:var(--purple);}
.callout.purple strong{color:var(--purple);}
.code-block{background:var(--bg-code);border-radius:10px;padding:24px 28px;margin:24px 0;overflow-x:auto;position:relative;}
.code-block pre{font-family:var(--font-mono);font-size:13px;line-height:1.7;color:var(--text-code);}
.code-label{position:absolute;top:10px;right:14px;font-family:var(--font-mono);font-size:10px;color:#666;letter-spacing:.5px;}
.kw{color:#cf8e6d;} .fn{color:#87ceeb;} .cm{color:#6a737d;font-style:italic;}
.st{color:#6aab73;} .nu{color:#79b4de;} .cl{color:#ffd580;}
.op{color:#c792ea;} .dc{color:#888;} .lm{color:#5b9bd5;}
.bi{color:#56b6c2;} .pr{color:#94c7d9;}
.deepdive{border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin:28px 0;background:var(--bg-card);}
.deepdive-label{font-family:var(--font-mono);font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:14px;}
.deepdive-links{display:flex;flex-direction:column;gap:8px;}
.deepdive-links a{display:flex;align-items:flex-start;gap:10px;text-decoration:none;color:var(--text);padding:6px 0;font-size:13.5px;line-height:1.5;transition:color .2s;}
.deepdive-links a:hover{color:var(--accent);}
.link-icon{color:var(--accent);font-weight:600;flex-shrink:0;margin-top:1px;}
.link-label{font-weight:600;}
.cell{border:1px solid var(--border);border-radius:10px;margin:28px 0;overflow:hidden;}
.cell-header{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;background:var(--bg-aside);border-bottom:1px solid var(--border);}
.cell-label{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);letter-spacing:.5px;}
.run-btn{background:var(--accent);color:#fff;border:none;padding:6px 16px;border-radius:6px;font-family:var(--font-mono);font-size:12px;cursor:pointer;transition:opacity .2s;}
.run-btn:hover{opacity:.85;}
.run-btn:disabled{opacity:.5;cursor:default;}
.cell-code{background:var(--bg-code);padding:20px 24px;}
.cell-code pre{font-family:var(--font-mono);font-size:13px;line-height:1.65;color:var(--text-code);white-space:pre-wrap;}
.cell-output{padding:16px 20px;font-family:var(--font-mono);font-size:13px;line-height:1.6;color:#444;min-height:32px;background:#fff;border-top:1px solid var(--border);}
.cell-output.running{color:var(--accent);}
.cell-output.error{color:#c0392b;}
.formula{background:#1d1d1d;color:#e8e4de;font-family:var(--font-mono);font-size:14px;padding:18px 24px;border-radius:8px;margin:20px 0;text-align:center;letter-spacing:.3px;}
.formula .eq-label{font-size:11px;color:#666;display:block;margin-bottom:6px;text-align:left;}
.ar-loop{display:flex;flex-direction:column;gap:4px;margin:24px 0;}
.ar-tokens{display:flex;gap:4px;flex-wrap:wrap;}
.ar-tok{background:var(--bg-aside);border:1px solid var(--border);border-radius:5px;padding:6px 10px;font-family:var(--font-mono);font-size:13px;}
.ar-tok.new{background:var(--accent);color:#fff;border-color:var(--accent);}
.ar-tok.ctx{background:var(--blue-light);border-color:var(--blue);color:var(--blue);}
.ar-arrow{text-align:center;color:var(--text-secondary);font-size:12px;padding:2px 0;}
.ar-probs{display:flex;gap:6px;align-items:flex-end;height:60px;margin:8px 0;}
.ar-bar-wrap{display:flex;flex-direction:column;align-items:center;gap:3px;flex:1;}
.ar-bar{background:var(--bg-aside);border:1px solid var(--border);border-radius:4px 4px 0 0;width:100%;}
.ar-bar.top{background:var(--accent);}
.ar-bar-label{font-family:var(--font-mono);font-size:10px;color:var(--text-secondary);}
.tblock{border:2px solid var(--border);border-radius:10px;padding:20px;margin:24px 0;position:relative;}
.tblock-title{font-family:var(--font-mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:12px;}
.tblock-layers{display:flex;flex-direction:column;gap:8px;}
.tblock-row{display:flex;align-items:center;gap:10px;}
.tblock-node{flex:1;border-radius:7px;padding:10px 14px;text-align:center;font-size:13px;font-weight:500;}
.tbn-norm{background:#e8f4ec;border:1px solid #4a8c6f;color:var(--green);}
.tbn-attn{background:#e8eef8;border:1px solid #3b6bb5;color:var(--blue);}
.tbn-ffn{background:#fdf0ea;border:1px solid #c96442;color:var(--accent);}
.tbn-add{background:#f5f0ff;border:1px solid #7c5cbf;color:var(--purple);font-size:11px;padding:8px 10px;}
.tblock-residual{position:absolute;left:8px;top:0;bottom:0;display:flex;flex-direction:column;align-items:center;}
.residual-line{width:2px;flex:1;background:repeating-linear-gradient(180deg,var(--blue) 0,var(--blue) 6px,transparent 6px,transparent 10px);}
.residual-label{font-family:var(--font-mono);font-size:9px;color:var(--blue);letter-spacing:.5px;transform:rotate(-90deg);white-space:nowrap;}
.qkv-diag{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;margin:24px 0;}
.qkv-col{display:flex;flex-direction:column;gap:8px;}
.qkv-mat{border-radius:7px;padding:12px 16px;text-align:center;font-family:var(--font-mono);font-size:13px;font-weight:500;}
.qkv-Q{background:#fff8e1;border:1px solid #b87c2a;color:var(--amber);}
.qkv-K{background:var(--blue-light);border:1px solid var(--blue);color:var(--blue);}
.qkv-V{background:var(--green-light);border:1px solid var(--green);color:var(--green);}
.qkv-arrow{text-align:center;color:var(--text-secondary);}
.qkv-result{background:var(--bg-aside);border:1px solid var(--border);border-radius:7px;padding:12px 16px;text-align:center;font-size:12px;}
.qkv-input{background:var(--bg-code);color:var(--text-code);border-radius:7px;padding:10px 16px;font-family:var(--font-mono);font-size:13px;text-align:center;}
.mask-wrap{margin:24px 0;}
.mask-title{font-family:var(--font-mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:8px;}
.mask-grid{display:inline-grid;gap:2px;border:1px solid var(--border);border-radius:6px;padding:8px;background:var(--bg-aside);}
.mk{width:32px;height:32px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:9px;}
.mk-on{background:var(--blue-light);border:1px solid var(--blue);color:var(--blue);}
.mk-off{background:repeating-linear-gradient(45deg,#e8e4de 0,#e8e4de 4px,transparent 4px,transparent 8px);border:1px solid #ddd;color:#ccc;}
.mk-head{background:var(--bg-aside);border:none;color:var(--text-secondary);font-size:9px;}
.mask-legend{display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--text-secondary);}
.mask-legend-item{display:flex;align-items:center;gap:6px;}
.ml-swatch{width:14px;height:14px;border-radius:3px;}
.mha-diag{margin:24px 0;}
.mha-heads{display:flex;gap:8px;margin-bottom:12px;}
.mha-head{flex:1;border-radius:7px;padding:10px;text-align:center;font-size:11px;font-family:var(--font-mono);}
.mh-1{background:#fff8e1;border:1px solid #b87c2a;color:var(--amber);}
.mh-2{background:var(--blue-light);border:1px solid var(--blue);color:var(--blue);}
.mh-3{background:var(--green-light);border:1px solid var(--green);color:var(--green);}
.mh-4{background:#f5f0ff;border:1px solid #7c5cbf;color:var(--purple);}
.mha-concat{height:24px;background:linear-gradient(90deg,#fff8e1 25%,var(--blue-light) 25% 50%,var(--green-light) 50% 75%,#f5f0ff 75%);border:1px solid var(--border);border-radius:5px;margin-bottom:6px;}
.mha-proj{background:var(--bg-aside);border:1px solid var(--border);border-radius:7px;padding:8px;text-align:center;font-size:12px;color:var(--text-secondary);}
.mha-label{font-family:var(--font-mono);font-size:10px;color:var(--text-secondary);text-align:center;margin:4px 0;}
.gqa-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:20px 0;}
.gqa-card{border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center;}
.gqa-card h4{font-size:12px;font-weight:600;margin-bottom:8px;}
.gqa-heads{display:flex;gap:3px;justify-content:center;flex-wrap:wrap;margin-bottom:8px;}
.gqa-hq{width:12px;height:20px;border-radius:3px;background:var(--accent);}
.gqa-hkv{width:12px;height:20px;border-radius:3px;background:var(--blue);}
.gqa-stat{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);}
.pipeline{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin:24px 0;}
.pipe-step{padding:20px;text-align:center;}
.ps-1{background:#f5f0ff;} .ps-2{background:#e8eef8;} .ps-3{background:#fdf0ea;}
.pipe-num{font-family:var(--font-mono);font-size:20px;font-weight:500;margin-bottom:6px;}
.ps-1 .pipe-num{color:var(--purple);} .ps-2 .pipe-num{color:var(--blue);} .ps-3 .pipe-num{color:var(--accent);}
.pipe-name{font-size:13px;font-weight:600;margin-bottom:6px;}
.pipe-desc{font-size:11px;color:var(--text-secondary);line-height:1.5;}
.lora-diag{display:flex;align-items:center;gap:12px;margin:24px 0;flex-wrap:wrap;}
.lora-box{border-radius:7px;padding:16px 20px;text-align:center;font-family:var(--font-mono);font-size:12px;}
.lora-W0{background:#eee;border:2px solid #999;color:#555;min-width:100px;}
.lora-A{background:#fff8e1;border:2px solid var(--amber);color:var(--amber);}
.lora-B{background:var(--blue-light);border:2px solid var(--blue);color:var(--blue);}
.lora-delta{background:var(--green-light);border:2px solid var(--green);color:var(--green);}
.lora-eq{font-size:20px;color:var(--text-secondary);}
.lora-frozen{font-size:9px;display:block;margin-top:4px;opacity:.7;}
.moe-diag{margin:24px 0;}
.moe-router{background:var(--accent);color:#fff;border-radius:7px;padding:10px 20px;text-align:center;font-size:13px;font-weight:500;margin-bottom:12px;}
.moe-experts{display:flex;gap:6px;}
.moe-exp{flex:1;border-radius:7px;padding:12px 8px;text-align:center;font-size:11px;font-family:var(--font-mono);}
.moe-active{background:#e8f4ec;border:2px solid var(--green);color:var(--green);}
.moe-inactive{background:var(--bg-aside);border:1px solid var(--border);color:var(--text-secondary);}
.moe-weights{display:flex;gap:6px;margin-top:6px;}
.moe-w{flex:1;border-radius:4px;padding:4px;font-family:var(--font-mono);font-size:10px;text-align:center;}
.data-bar-wrap{margin:24px 0;}
.data-model-label{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);margin-bottom:4px;}
.data-bar{display:flex;height:28px;border-radius:6px;overflow:hidden;margin-bottom:10px;}
.ds{display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:#fff;}
.ds-web{background:#3b6bb5;} .ds-books{background:#4a8c6f;} .ds-code{background:#c96442;}
.ds-wiki{background:#7c5cbf;} .ds-papers{background:#b87c2a;} .ds-other{background:#888;}
.data-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;margin-top:4px;}
.dl-item{display:flex;align-items:center;gap:5px;}
.dl-sw{width:10px;height:10px;border-radius:2px;}
.mem-stack{display:flex;flex-direction:column;gap:4px;margin:24px 0;}
.mem-tier{border-radius:7px;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;}
.mt-regs{background:#fdf0ea;border:1px solid var(--accent);}
.mt-sram{background:#e8f4ec;border:1px solid var(--green);}
.mt-hbm{background:var(--blue-light);border:1px solid var(--blue);}
.mt-dram{background:var(--bg-aside);border:1px solid var(--border);}
.mem-name{font-weight:600;font-size:13px;}
.mem-sz{font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);}
.mem-bw{font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);}
.kv-diag{margin:24px 0;}
.kv-step{display:flex;gap:6px;margin-bottom:10px;align-items:center;}
.kv-step-label{font-family:var(--font-mono);font-size:10px;color:var(--text-secondary);width:52px;}
.kv-tok{border-radius:4px;padding:6px 8px;font-family:var(--font-mono);font-size:11px;text-align:center;}
.kv-cached{background:#e8f4ec;border:1px solid var(--green);color:var(--green);}
.kv-new{background:var(--accent);color:#fff;}
.kv-kv{background:var(--bg-aside);border:1px solid var(--border);color:var(--text-secondary);font-size:10px;}
.quant-compare{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:24px 0;}
.quant-card{border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center;}
.quant-card h4{font-size:12px;font-weight:600;margin-bottom:6px;}
.quant-bits{font-family:var(--font-mono);font-size:28px;font-weight:700;}
.quant-name{font-size:11px;color:var(--text-secondary);margin:4px 0;}
.quant-loss{font-size:11px;}
.ql-none{color:var(--green);} .ql-low{color:var(--amber);} .ql-med{color:var(--accent);}
.emerge-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:24px 0;}
.emerge-card{border:1px solid var(--border);border-radius:8px;padding:14px;}
.emerge-card h4{font-size:12px;font-weight:600;margin-bottom:6px;color:var(--accent);}
.emerge-list{font-size:12px;line-height:1.7;color:var(--text-secondary);}
.pe-heatmap{display:flex;flex-direction:column;gap:2px;margin:24px 0;}
.pe-row{display:flex;gap:2px;}
.pe-cell{width:14px;height:14px;border-radius:2px;}
.pe-axis{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:9px;color:var(--text-secondary);margin-top:4px;}
.model-table{width:100%;border-collapse:collapse;margin:24px 0;font-size:13px;}
.model-table th{background:var(--bg-aside);padding:10px 14px;text-align:left;font-weight:600;border-bottom:2px solid var(--border);}
.model-table td{padding:10px 14px;border-bottom:1px solid var(--border-light);}
.model-table tr:hover td{background:var(--accent-light);}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-family:var(--font-mono);font-size:10px;font-weight:600;}
.badge-open{background:var(--green-light);color:var(--green);}
.badge-prop{background:var(--accent-light);color:var(--accent);}
.slider-wrap{background:var(--bg-aside);border:1px solid var(--border);border-radius:10px;padding:18px 22px;margin:24px 0;}
.slider-wrap label{display:flex;align-items:center;justify-content:space-between;font-size:14px;margin-bottom:6px;}
.slider-wrap input[type=range]{width:100%;cursor:pointer;}
.slider-output{font-family:var(--font-mono);font-size:13px;background:var(--bg-card);border:1px solid var(--border);border-radius:6px;padding:10px 14px;margin-top:10px;min-height:32px;white-space:pre-wrap;}
#pyodide-status{text-align:center;padding:12px;background:var(--bg-aside);border-radius:8px;font-size:13px;color:var(--text-secondary);margin:20px 0;}
</style>
</head>
<body>
<nav class="topnav">
  <div class="container">
    <span class="topnav-title">Practical LLMs</span>
    <span class="topnav-chapter">Chapter 1 \u2014 Understanding Large Language Models</span>
  </div>
</nav>

<div class="container">
<header class="chapter-header">
  <div class="chapter-label">Chapter 1</div>
  <h1>Understanding Large Language Models</h1>
  <p>From the autoregressive loop to Flash Attention, from BPE tokenisation to Mixture of Experts \u2014 a rigorous, code-first tour grounded in the papers that built modern LLMs.</p>
</header>

<nav class="toc">
  <h3>Contents</h3>
  <ul class="toc-list">
    <li><a href="#s1"><span class="toc-num">1.1</span>What Is a Large Language Model?</a></li>
    <li><a href="#s2"><span class="toc-num">1.2</span>Tokenisation &amp; BPE</a></li>
    <li><a href="#s3"><span class="toc-num">1.3</span>Embeddings &amp; Vector Geometry</a></li>
    <li><a href="#s4"><span class="toc-num">1.4</span>Transformer Architecture</a></li>
    <li><a href="#s5"><span class="toc-num">1.5</span>Self-Attention Mechanism</a></li>
    <li><a href="#s6"><span class="toc-num">1.6</span>Multi-Head Attention &amp; GQA</a></li>
    <li><a href="#s7"><span class="toc-num">1.7</span>Positional Encoding &amp; RoPE</a></li>
    <li><a href="#s8"><span class="toc-num">1.8</span>The Feedforward Sublayer</a></li>
    <li><a href="#s9"><span class="toc-num">1.9</span>Pre-training at Scale &amp; Chinchilla</a></li>
    <li><a href="#s10"><span class="toc-num">1.10</span>BERT vs GPT: Encoder vs Decoder</a></li>
    <li><a href="#s11"><span class="toc-num">1.11</span>KV Cache &amp; Inference Memory</a></li>
    <li><a href="#s12"><span class="toc-num">1.12</span>Flash Attention</a></li>
    <li><a href="#s13"><span class="toc-num">1.13</span>Alignment: RLHF, DPO &amp; LoRA</a></li>
    <li><a href="#s14"><span class="toc-num">1.14</span>Scale &amp; Emergent Abilities</a></li>
    <li><a href="#s15"><span class="toc-num">1.15</span>Mixture of Experts</a></li>
    <li><a href="#s16"><span class="toc-num">1.16</span>Quantization &amp; Limitations</a></li>
    <li><a href="#s17"><span class="toc-num">1.17</span>2025\u201326 Landscape &amp; Roadmap</a></li>
  </ul>
</nav>

<div id="pyodide-status">Loading Python runtime (Pyodide)\u2026</div>

<main>
"""

print(f"HEAD written: {len(HEAD)} chars")
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(HEAD)
print("File started.")
