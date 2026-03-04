#!/usr/bin/env python3
# Appends sections 1.9 (Chinchilla), 1.10 (BERT vs GPT), 1.11 (KV Cache), 1.12 (Flash Attention)
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

lines = []
A = lines.append

A("""
<!-- ============================================================ §1.9 == -->
<section class="section" id="s9">
  <div class="section-label">1.9</div>
  <h2>Pre-training at Scale &amp; Chinchilla</h2>

  <p>LLMs are pretrained on vast, diverse text corpora by minimising next-token prediction loss. The datasets
  have grown dramatically: GPT-3 trained on ~300 billion tokens (mainly Common Crawl, WebText, Books1/2,
  Wikipedia); LLaMA-3 trained on <strong>15 trillion tokens</strong>. Composition matters as much as size:
  code, books, and scientific papers consistently outperform raw web text per token because they contain
  denser reasoning, explicit structure, and factual content.</p>

  <p>The <strong>Chinchilla scaling laws</strong> (Hoffmann et al., 2022, arXiv:2203.15556) upended the field.
  By training over 400 models from 70M to 16B parameters across 5B to 500B token budgets, they found that
  compute-optimal training requires roughly <em>20 tokens per parameter</em>. GPT-3 (175B parameters) should
  have trained on 3.5 trillion tokens &#8212; it used only 300 billion, making it severely undertrained.
  The 70B Chinchilla model, trained on 1.4 trillion tokens, outperforms 280B Gopher on every evaluated
  benchmark while using identical compute. The message: &#8220;current large language models are significantly
  undertrained.&#8221;</p>

  <div class="data-bar-wrap">
    <div class="data-model-label">GPT-3 (~300B tokens, Brown et al. 2020)</div>
    <div class="data-bar">
      <div class="ds ds-web" style="flex:60">Web 60%</div>
      <div class="ds ds-books" style="flex:22">Books 22%</div>
      <div class="ds ds-wiki" style="flex:9">Wiki</div>
      <div class="ds ds-papers" style="flex:5">Papers</div>
      <div class="ds ds-other" style="flex:4"></div>
    </div>
    <div class="data-model-label">LLaMA-3 (~15T tokens, Meta 2024)</div>
    <div class="data-bar">
      <div class="ds ds-web" style="flex:50">Web 50%</div>
      <div class="ds ds-code" style="flex:17">Code 17%</div>
      <div class="ds ds-books" style="flex:15">Books 15%</div>
      <div class="ds ds-wiki" style="flex:8">Wiki</div>
      <div class="ds ds-papers" style="flex:10">Papers 10%</div>
    </div>
    <div class="data-legend">
      <div class="dl-item"><div class="dl-sw" style="background:#3b6bb5;"></div>Web</div>
      <div class="dl-item"><div class="dl-sw" style="background:#4a8c6f;"></div>Books</div>
      <div class="dl-item"><div class="dl-sw" style="background:#c96442;"></div>Code</div>
      <div class="dl-item"><div class="dl-sw" style="background:#7c5cbf;"></div>Wikipedia</div>
      <div class="dl-item"><div class="dl-sw" style="background:#b87c2a;"></div>Papers</div>
    </div>
  </div>

  <div class="callout green">
    <strong>Deduplication reduces memorisation 10&#215;.</strong> Lee et al. (2022) found that training on
    deduplicated data reduces verbatim memorisation of training examples by 10&#215; while improving
    downstream task performance. Most modern pipelines run min-hash or SimHash-based near-deduplication before
    training. LLaMA-3 used aggressive deduplication across its 15T token corpus.
  </div>

  <div class="cell" id="cell-9">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Chinchilla compute-optimal allocation</span>
      <button class="run-btn" onclick="runCell('cell-9')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

# Chinchilla loss formula: L(N, D) = E + A/N^alpha + B/D^beta
# Coefficients from Hoffmann et al. 2022, Table 3 (Approach 3)
E     = 1.69    # irreducible entropy
A     = 406.4   # model-size scaling coefficient
B     = 410.7   # data-size scaling coefficient
alpha = 0.34    # model-size scaling exponent
beta  = 0.28    # data-size scaling exponent

def chinchilla_loss(N, D):
    return E + A / N**alpha + B / D**beta

def optimal_ND(C, steps=2000, lr=0.01):
    # Find compute-optimal N and D for budget C FLOPs
    # FLOPs ~= 6*N*D for a single pass
    N = 1e9
    for _ in range(steps):
        D = C / (6 * N)
        # gradient w.r.t. log(N) (log-scale gradient descent)
        dN = -alpha*A * N**(-alpha-1) + beta*B * (6/C)**beta * N**(beta-1)
        N = max(N * math.exp(-lr * dN * N), 1e6)
    D = C / (6 * N)
    return N, D

print("Chinchilla compute-optimal allocation")
print(f"{'Scenario':22s} {'Opt N':>10s} {'Opt D':>12s} {'D/N ratio':>10s}")
print("-" * 58)

budgets = [
    ("GPT-3 budget",    3.14e23),
    ("LLaMA-3 8B budget", 1.8e24),
    ("10x GPT-3",       3.14e24),
    ("1e25 FLOPs",      1e25),
]
for name, C in budgets:
    N, D = optimal_ND(C)
    ratio = D / N
    print(f"{name:22s} {N/1e9:8.1f}B {D/1e9:10.1f}B {ratio:10.1f} tok/param")

print()
print("Reality check: how undertrained was GPT-3?")
gpt3_actual_loss   = chinchilla_loss(175e9, 300e9)
gpt3_optimal_N, gpt3_optimal_D = optimal_ND(3.14e23)
gpt3_optimal_loss  = chinchilla_loss(gpt3_optimal_N, gpt3_optimal_D)
gpt3_iso_loss      = chinchilla_loss(gpt3_optimal_N, 300e9)   # same budget, more tokens
print(f"  GPT-3 actual (175B, 300B tok):      loss = {gpt3_actual_loss:.4f}")
print(f"  Chinchilla optimal at same compute: loss = {gpt3_optimal_loss:.4f}")
print(f"  Delta: {gpt3_actual_loss - gpt3_optimal_loss:.4f} nats (significant)")
print(f"  Optimal params at GPT-3 budget:  {gpt3_optimal_N/1e9:.1f}B (not 175B)")
print(f"  Optimal tokens at GPT-3 budget:  {gpt3_optimal_D/1e9:.1f}B (not 300B)")

# Try changing the budget to 1e24 and see how optimal N and D change
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2203.15556" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Hoffmann et al., 2022 &#8212; "Training Compute-Optimal Large Language Models" (Chinchilla)</span> &#8212; ~20 tokens per parameter is optimal. Chinchilla 70B on 1.4T tokens beats 280B Gopher. Over 400 models trained.</span></a>
      <a href="https://arxiv.org/abs/2001.08361" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Kaplan et al., 2020 &#8212; "Scaling Laws for Neural Language Models"</span> &#8212; Loss follows power laws in model size, data, and compute across 7+ orders of magnitude.</span></a>
      <a href="https://arxiv.org/abs/2107.06499" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Lee et al., 2022 &#8212; "Deduplicating Training Data Makes Language Models Better"</span> &#8212; 10&#215; less memorisation, improved performance. The case for aggressive deduplication.</span></a>
      <a href="https://arxiv.org/abs/2302.13971" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Touvron et al., 2023 &#8212; "LLaMA: Open and Efficient Foundation Language Models"</span> &#8212; The original LLaMA: 7B trained on 1T tokens outperforms GPT-3 175B on most benchmarks.</span></a>
      <a href="https://blog.eleuther.ai/transformer-math/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">EleutherAI &#8212; "Transformer Math 101"</span> &#8212; Practical guide to FLOPs, compute budgets, memory estimation, and the 6ND rule.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.10 == -->
<section class="section" id="s10">
  <div class="section-label">1.10</div>
  <h2>BERT vs GPT: Encoder vs Decoder</h2>

  <p>Both BERT and GPT are transformer-based, but they differ in attention pattern and training objective.
  <strong>GPT</strong> (decoder-only, Radford et al. 2018) uses causal (unidirectional) attention and is
  trained with <em>causal language modelling</em> (CLM): predict the next token given left context. This makes
  it naturally generative. <strong>BERT</strong> (encoder-only, Devlin et al. 2019, arXiv:1810.04805) uses
  bidirectional attention and is trained with <em>masked language modelling</em> (MLM): 15% of input tokens
  are randomly masked, and the model predicts them using context from both directions.</p>

  <p>The bidirectional context gives BERT richer representations for classification tasks. But the autoregressive
  structure of GPT allows generation &#8212; and at scale, generation proves to be the more powerful paradigm.
  GPT-3 demonstrated that a large enough generative model performs classification by framing it as generation
  (<em>few-shot prompting</em>). By 2023, decoder-only models dominated essentially every NLP benchmark despite
  being &#8220;weaker&#8221; per token due to their restricted context window at each position.</p>

  <div class="callout blue">
    <strong>T5</strong> (Raffel et al., 2020, arXiv:1910.10683) tried a third approach: encoder-decoder with
    a unified text-to-text framework. Every task is framed as text-to-text conversion. T5-XXL (11B) was
    competitive with GPT-3 (175B) on many benchmarks &#8212; a 16&#215; parameter efficiency advantage.
    But the encoder-decoder paradigm hasn't scaled as cleanly as pure decoders, and 2023&#8211;2024 saw
    near-universal consolidation around decoder-only architectures.
  </div>

  <div class="cell" id="cell-10">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; CLM vs MLM: attention mask &amp; training objective</span>
      <button class="run-btn" onclick="runCell('cell-10')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random
random.seed(42)

def softmax_rows(matrix):
    result = []
    for row in matrix:
        mx = max(row)
        exps = [math.exp(x - mx) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result

def make_causal_mask(n):
    # Lower triangular: each position can only see its left context
    return [[1 if j <= i else 0 for j in range(n)] for i in range(n)]

def make_bert_mask(n, mask_prob=0.15, seed=42):
    random.seed(seed)
    attend = [[1]*n for _ in range(n)]   # full bidirectional attention
    masked = [i for i in range(n) if random.random() < mask_prob]
    return attend, masked

T = 6
tokens = ["The", "cat", "[?]", "on", "the", "mat"]

# GPT: causal mask
gpt_mask = make_causal_mask(T)
raw = [[random.gauss(0,1) for _ in range(T)] for _ in range(T)]
gpt_scores = [[raw[i][j] if gpt_mask[i][j] else -1e9 for j in range(T)] for i in range(T)]
gpt_weights = softmax_rows(gpt_scores)

print("=== GPT-style (Causal) Attention ===")
print(f"{'':6s}" + "".join(f"{t:7s}" for t in tokens))
for i, row in enumerate(gpt_weights):
    print(f"{tokens[i]:6s}" + "".join(f"{v:7.3f}" for v in row))
print("Upper triangle = 0 (future masked)")

# BERT: bidirectional
bert_mask, masked_pos = make_bert_mask(T, mask_prob=0.5)
print(f"\n=== BERT-style (Bidirectional) Attention ===")
print("Full bidirectional: all positions attend to all positions.")
print(f"MLM masked positions (predict these): {[tokens[i] for i in masked_pos]}")

# BERT sees all positions but only computes loss on masked tokens
print("\nKey differences:")
rows = [
    ("Architecture",    "Decoder-only",          "Encoder-only"),
    ("Attention mask",  "Causal (lower triangle)","Full bidirectional"),
    ("Training obj.",   "Next-token prediction",  "Masked token prediction (15%)"),
    ("Good for",        "Generation, reasoning",  "Classification, NER, QA"),
    ("Vocab per token", "Sees left only",         "Sees full context"),
    ("2024 dominance",  "Dominant (GPT-4, Claude)","Niche (fine-tuned classifiers)"),
]
print(f"\n{'':22s} {'GPT (decoder)':26s} {'BERT (encoder)':26s}")
print("-" * 76)
for prop, gpt, bert in rows:
    print(f"{prop:22s} {gpt:26s} {bert:26s}")

# Try increasing mask_prob to 0.5 to see more masked tokens
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1810.04805" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Devlin et al., 2019 &#8212; "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"</span> &#8212; MLM + next-sentence prediction; bidirectional attention. BERT-large: 340M params, 16 heads, 24 layers.</span></a>
      <a href="https://arxiv.org/abs/1910.10683" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Raffel et al., 2020 &#8212; "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)</span> &#8212; Encoder-decoder; every NLP task as text-to-text. T5-11B competitive with GPT-3 175B.</span></a>
      <a href="https://arxiv.org/abs/2204.05832" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Wang et al., 2022 &#8212; "What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"</span> &#8212; Systematic comparison: decoder-only wins for zero-shot generalisation.</span></a>
      <a href="https://jalammar.github.io/illustrated-bert/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jay Alammar &#8212; "The Illustrated BERT, ELMo, and co."</span> &#8212; Visual comparison of BERT's bidirectional attention vs GPT's causal attention.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.11 == -->
<section class="section" id="s11">
  <div class="section-label">1.11</div>
  <h2>KV Cache &amp; Inference Memory</h2>

  <p>During inference, each token's K and V projections are computed once and <em>cached</em>. When the
  next token is generated, only the new token's Q, K, V are computed &#8212; then the new K and V are appended
  to the cache, and attention is computed between the new Q and all cached K/V pairs. Without caching, each
  step requires O(N&#178;) re-computation; with the cache, each step is O(N) in total across the context.</p>

  <p>The memory cost is substantial. LLaMA-3 8B: 32 layers, 8 KV-heads, d_head=128, BF16 (2 bytes).
  One token occupies 32 &#215; 8 &#215; 128 &#215; 2 &#215; 2 = 131,072 bytes &#8776; 128 KB.
  At 128K context, that's <strong>16 GB</strong> just for the KV cache &#8212; equal to the model weights.
  At full MHA (32 KV-heads), it would be <strong>64 GB</strong>. This is why GQA (&#167;1.6) is critical for
  long-context serving, and why vLLM's <em>PagedAttention</em> (Kwon et al., 2023) manages KV cache like OS
  virtual memory pages, achieving 24&#215; higher throughput by eliminating fragmentation.</p>

  <div class="kv-diag">
    <div class="kv-step">
      <div class="kv-step-label">Step 1</div>
      <div class="kv-tok kv-new">The</div>
      <div style="margin:0 6px;font-size:11px;color:var(--text-secondary);">&#8594; compute &amp; store K&#8321;V&#8321;</div>
    </div>
    <div class="kv-step">
      <div class="kv-step-label">Step 2</div>
      <div class="kv-tok kv-cached">The</div>
      <div class="kv-tok kv-new">cat</div>
      <div style="margin:0 6px;font-size:11px;color:var(--text-secondary);">&#8594; K&#8321;V&#8321; cached; compute K&#8322;V&#8322;</div>
    </div>
    <div class="kv-step">
      <div class="kv-step-label">Step 3</div>
      <div class="kv-tok kv-cached">The</div>
      <div class="kv-tok kv-cached">cat</div>
      <div class="kv-tok kv-new">sat</div>
      <div style="margin:0 6px;font-size:11px;color:var(--text-secondary);">&#8594; K&#8321;V&#8321;K&#8322;V&#8322; cached; compute K&#8323;V&#8323;</div>
    </div>
  </div>

  <div class="formula">
    <span class="eq-label">KV cache memory formula</span>
    Memory = batch &#215; seq_len &#215; n_layers &#215; kv_heads &#215; d_head &#215; 2 (K+V) &#215; bytes_per_element
  </div>

  <div class="cell" id="cell-11">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; KV cache memory budget at real scales</span>
      <button class="run-btn" onclick="runCell('cell-11')">Run</button>
    </div>
    <div class="cell-code"><pre>def kv_cache_gb(batch, seq, n_layers, kv_heads, d_model, n_heads, bits=16):
    d_head = d_model // n_heads
    return batch * seq * n_layers * kv_heads * d_head * 2 * (bits//8) / 1e9

def model_gb(d, d_ff, L, H, kv, V, bits=16):
    d_head = d // H
    embed  = V * d
    attn   = L * (d*d + d*(kv*d_head)*2 + d*d)
    ffn    = L * (d*d_ff*2 + d_ff*d)
    return (embed + attn + ffn) * (bits//8) / 1e9

print("LLaMA-3 8B: KV cache growth vs model weights (BF16)")
weights = model_gb(4096, 14336, 32, 32, 8, 128256)
print(f"Model weights: {weights:.2f} GB")
print(f"{'seq_len':>10s} {'KV cache':>12s} {'ratio to weights':>18s}")
print("-" * 44)
for seq in [1024, 4096, 16384, 65536, 131072]:
    kv = kv_cache_gb(1, seq, 32, 8, 4096, 32)
    print(f"{seq:10,d} {kv:10.2f} GB {kv/weights*100:16.0f}%")

print()
print("GQA vs MHA at seq=32K, LLaMA-3 8B scale:")
for name, kv_h in [("MHA (32 KV heads)", 32), ("GQA (8 groups)", 8), ("MQA (1 head)", 1)]:
    gb = kv_cache_gb(1, 32768, 32, kv_h, 4096, 32)
    print(f"  {name:22s}: {gb:.2f} GB")

print()
print("Operations saved by KV cache (cumulative):")
print(f"{'decode step':>12s} {'with cache':>14s} {'without cache':>16s} {'speedup':>10s}")
print("-" * 56)
for step in [1, 10, 50, 200, 500]:
    with_c   = step * 64            # O(N) total cost
    without_c = sum(i*64 for i in range(1, step+1))  # O(N^2) total
    print(f"{step:12d} {with_c:12,d} {without_c:14,d} {without_c//with_c:9d}x")

# Try changing kv_h to 32 to see the pre-GQA memory cost
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2309.06180" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Kwon et al., 2023 &#8212; "Efficient Memory Management for Large Language Model Serving with PagedAttention"</span> &#8212; vLLM: KV cache as OS virtual memory. 24&#215; higher throughput by eliminating 60&#8211;80% memory waste from fragmentation.</span></a>
      <a href="https://arxiv.org/abs/2211.05102" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Pope et al., 2023 &#8212; "Efficiently Scaling Transformer Inference"</span> &#8212; Memory-bandwidth bottleneck analysis. KV cache dominates inference cost at long context.</span></a>
      <a href="https://lilianweng.github.io/posts/2023-01-10-inference-optimization/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Lilian Weng, 2023 &#8212; "Large Transformer Model Inference Optimization"</span> &#8212; Comprehensive overview of KV caching, quantization, speculative decoding, and batching strategies.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.12 == -->
<section class="section" id="s12">
  <div class="section-label">1.12</div>
  <h2>Flash Attention</h2>

  <p>Standard attention materialises the full N&#215;N attention matrix in GPU HBM (high-bandwidth memory,
  the GPU's VRAM). For N=4,096 tokens and d=128, that's 4,096&#178; &#215; 4 bytes = 67 MB <em>per head per
  layer</em>. The bottleneck is not compute &#8212; it's memory bandwidth. Reading and writing this matrix to/from
  HBM dominates runtime. Dao et al. (2022, arXiv:2205.14135) observed that on-chip SRAM is 19 TB/s bandwidth
  vs 2 TB/s for HBM, but only ~50 MB total on an A100. Flash Attention tiles Q, K, V into SRAM blocks
  (typically 64&#8211;128 tokens per tile), computes attention within each tile, and never materialises
  the full N&#215;N matrix.</p>

  <p>Flash Attention is <strong>exact</strong>, not approximate &#8212; it produces bit-identical results
  to standard attention. The speedup is purely algorithmic: 2&#8211;4&#215; end-to-end wall-clock reduction
  on GPT-2 training, 7.6&#215; fewer HBM reads/writes per forward pass on A100. Flash Attention-2 (2023)
  reorganises the inner loop to better parallelise across GPU thread blocks, achieving another 2&#215; speedup.
  Flash Attention-3 (2024) exploits H100 async tensor cores for further gains. The technique is now universal:
  PyTorch 2.0 uses FA-2 by default via <code>F.scaled_dot_product_attention()</code>.</p>

  <div class="mem-stack">
    <div class="mem-tier mt-regs">
      <span class="mem-name" style="color:var(--accent);">Registers / L1 cache</span>
      <span class="mem-sz">~256 KB per SM</span>
      <span class="mem-bw">&#8734; (in-register ops)</span>
    </div>
    <div class="mem-tier mt-sram">
      <span class="mem-name" style="color:var(--green);">SRAM (shared mem, L2)</span>
      <span class="mem-sz">~50 MB (A100)</span>
      <span class="mem-bw">19 TB/s</span>
    </div>
    <div class="mem-tier mt-hbm">
      <span class="mem-name" style="color:var(--blue);">HBM (GPU VRAM)</span>
      <span class="mem-sz">40&#8211;80 GB (A100)</span>
      <span class="mem-bw">2 TB/s</span>
    </div>
    <div class="mem-tier mt-dram">
      <span class="mem-name" style="color:var(--text-secondary);">DRAM (CPU system RAM)</span>
      <span class="mem-sz">100s GB</span>
      <span class="mem-bw">50 GB/s</span>
    </div>
  </div>

  <div class="callout purple">
    <strong>Online softmax</strong> is the algorithmic key. Computing attention requires the full row of
    dot-product scores to normalise with softmax. FA computes a running max and sum as it processes each tile,
    using the numerically stable identity: if new max m' &gt; old max m, correct the accumulated sum by
    exp(m &#8722; m') before adding new terms. This allows exact softmax computation tile-by-tile with only
    O(N) memory instead of O(N&#178;).
  </div>

  <div class="formula">
    <span class="eq-label">Flash Attention HBM access complexity (Dao et al. 2022, Theorem 2)</span>
    Standard: O(N&#178; &#183; d) &nbsp;&#8594;&nbsp; Flash: O(N&#178; &#183; d&#178; / M)
    <br>where M = SRAM size. For N=4096, d=128, M=50MB: ~7.6&#215; fewer HBM reads/writes.
  </div>

  <div class="cell" id="cell-12">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Flash Attention tiling simulation (pure Python)</span>
      <button class="run-btn" onclick="runCell('cell-12')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random
random.seed(0)

def softmax_vec(v):
    mx = max(v)
    exps = [math.exp(x - mx) for x in v]
    s = sum(exps)
    return [e/s for e in exps]

def matmul(A, B):
    rA, cA, cB = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k]*B[k][j] for k in range(cA)) for j in range(cB)] for i in range(rA)]

def standard_attention(Q, K, V):
    T, d_k = len(Q), len(Q[0])
    scale = math.sqrt(d_k)
    Kt = [[K[j][i] for j in range(T)] for i in range(d_k)]
    S = matmul(Q, Kt)
    S = [[s/scale for s in row] for row in S]
    P = [softmax_vec(row) for row in S]
    return matmul(P, V)

def flash_attention_tiled(Q, K, V, B=2):
    # Flash-like: tiled computation, no full NxN materialised in memory
    T, d_k = len(Q), len(Q[0])
    d_v = len(V[0])
    scale = math.sqrt(d_k)
    O = [[0.0]*d_v for _ in range(T)]
    l = [0.0]*T       # running softmax denominator
    m = [-1e9]*T      # running max

    for j in range(0, T, B):         # tile over K/V
        Kj = K[j:j+B]
        Vj = V[j:j+B]
        for i in range(0, T, B):     # tile over Q
            Qi = Q[i:i+B]
            Sij = [[sum(Qi[a][k]*Kj[b][k] for k in range(d_k))/scale
                    for b in range(len(Kj))] for a in range(len(Qi))]
            for a in range(len(Qi)):
                row_max = max(Sij[a])
                m_new   = max(m[i+a], row_max)
                Pij_row = [math.exp(s - m_new) for s in Sij[a]]
                lij     = sum(Pij_row)
                # Correct accumulated O[i+a] for new max
                corr = math.exp(m[i+a] - m_new) if m[i+a] > -1e8 else 0.0
                l_new = corr * l[i+a] + lij
                for c in range(d_v):
                    O[i+a][c] = (corr * l[i+a] * O[i+a][c]
                                 + sum(Pij_row[b] * Vj[b][c] for b in range(len(Vj)))) / l_new
                m[i+a] = m_new
                l[i+a] = l_new
    return O

T, d_k, d_v = 8, 4, 4
Q = [[random.gauss(0,1) for _ in range(d_k)] for _ in range(T)]
K = [[random.gauss(0,1) for _ in range(d_k)] for _ in range(T)]
V = [[random.gauss(0,1) for _ in range(d_v)] for _ in range(T)]

out_std   = standard_attention(Q, K, V)
out_flash = flash_attention_tiled(Q, K, V, B=2)

# Check they match
max_diff = max(abs(out_std[i][j] - out_flash[i][j]) for i in range(T) for j in range(d_v))
print(f"Standard vs Flash Attention max difference: {max_diff:.2e}")
print(f"(Should be ~1e-10 floating point noise, not a different result)")

print(f"\nOutput row 0 (standard): {[round(x,4) for x in out_std[0]]}")
print(f"Output row 0 (tiled):    {[round(x,4) for x in out_flash[0]]}")

# HBM access comparison
print("\nHBM reads/writes (proportional) at various context lengths:")
print(f"{'seq_len':>10s} {'Standard (N^2)':>16s} {'Flash (N^2*d/M)':>18s} {'Speedup':>9s}")
print("-" * 57)
for N in [512, 1024, 4096, 16384, 65536]:
    std_ops   = N*N           # proportional to N^2
    flash_ops = N*128         # roughly N * d_k (linear in context for fixed d_k)
    print(f"{N:10,d} {std_ops:14,d} {flash_ops:16,d} {std_ops//flash_ops:8d}x")

# Try changing B (tile size) to 4 or 1 and re-running
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2205.14135" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Dao et al., 2022 &#8212; "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"</span> &#8212; IO-aware tiling: 7.6&#215; fewer HBM accesses, 3&#215; end-to-end speedup on GPT-2 training. Exact, not approximate.</span></a>
      <a href="https://arxiv.org/abs/2307.08691" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Dao, 2023 &#8212; "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"</span> &#8212; 2&#215; additional speedup by reorganising the inner loop across GPU thread blocks.</span></a>
      <a href="https://arxiv.org/abs/1805.02867" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Milakov &amp; Gimelshein, 2018 &#8212; "Online normalizer calculation for softmax"</span> &#8212; The running max/sum trick that enables tiled attention without materialising the full matrix.</span></a>
      <a href="https://github.com/Dao-AILab/flash-attention" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">FlashAttention &#8212; Official GitHub repository (Dao-AILab)</span> &#8212; CUDA implementation with PyTorch integration; drop-in replacement for standard attention.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
""")

with open(OUT, 'a', encoding='utf-8') as f:
    f.write(''.join(lines))
print(f"S9-S12 appended: {sum(len(l) for l in lines)} chars")
