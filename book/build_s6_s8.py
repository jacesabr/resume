#!/usr/bin/env python3
# Appends sections 1.6 (MHA/GQA), 1.7 (Positional Encoding/RoPE), 1.8 (FFN)
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

lines = []
A = lines.append

A("""
<!-- ============================================================ §1.6 == -->
<section class="section" id="s6">
  <div class="section-label">1.6</div>
  <h2>Multi-Head Attention &amp; GQA</h2>

  <p>Running attention once gives the model a single perspective. Multi-head attention (MHA) runs H separate
  attention operations in parallel, each with head dimension d_k = d_model/H. Each head learns to attend to
  different relationship types &#8212; one might track subject-verb agreement, another coreference chains,
  another syntactic dependencies. The outputs are concatenated and projected back to d_model via W_O.</p>

  <p>For large models, MHA's memory cost is prohibitive during inference due to the <strong>KV cache</strong>
  (&#167;1.11). <strong>Multi-Query Attention (MQA)</strong> (Shazeer, 2019) uses a single K and V head shared
  across all Q heads, reducing KV cache by H-fold with some quality cost. <strong>Grouped-Query Attention
  (GQA)</strong> (Ainslie et al., 2023, arXiv:2305.13245) is the sweet spot: G groups of Q heads share one
  K/V head each. LLaMA-3 uses 32 Q-heads and 8 KV-heads (4 Q-heads per KV group), reducing KV cache to 25%
  of MHA with negligible quality loss. Ainslie et al. showed GQA matches MHA quality while approaching MQA
  inference speed.</p>

  <div class="mha-diag">
    <div class="mha-label">Multi-Head Attention: 4 independent Q/K/V sets, all concatenated</div>
    <div class="mha-heads">
      <div class="mha-head mh-1">H1<br>Q K V</div>
      <div class="mha-head mh-2">H2<br>Q K V</div>
      <div class="mha-head mh-3">H3<br>Q K V</div>
      <div class="mha-head mh-4">H4<br>Q K V</div>
    </div>
    <div class="mha-concat"></div>
    <div class="mha-proj">Concat &#8594; W_O &#8594; output (d_model)</div>
  </div>

  <div class="gqa-grid">
    <div class="gqa-card">
      <h4>MHA (GPT-3)</h4>
      <div class="gqa-heads">
        <div class="gqa-hq"></div><div class="gqa-hkv"></div>
        <div class="gqa-hq"></div><div class="gqa-hkv"></div>
        <div class="gqa-hq"></div><div class="gqa-hkv"></div>
        <div class="gqa-hq"></div><div class="gqa-hkv"></div>
      </div>
      <div class="gqa-stat">H Q-heads = H KV-heads<br>KV cache: 100%</div>
    </div>
    <div class="gqa-card">
      <h4>GQA (LLaMA-3)</h4>
      <div class="gqa-heads">
        <div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hkv"></div>
        <div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hkv"></div>
      </div>
      <div class="gqa-stat">32 Q / 8 KV groups<br>KV cache: 25%</div>
    </div>
    <div class="gqa-card">
      <h4>MQA (Falcon)</h4>
      <div class="gqa-heads">
        <div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hq"></div><div class="gqa-hkv"></div>
      </div>
      <div class="gqa-stat">H Q-heads, 1 KV head<br>KV cache: 1/H &#8776; 3%</div>
    </div>
  </div>

  <div class="cell" id="cell-6">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; MHA vs GQA: KV cache memory at scale</span>
      <button class="run-btn" onclick="runCell('cell-6')">Run</button>
    </div>
    <div class="cell-code"><pre>def kv_cache_bytes(batch, seq_len, n_layers, kv_heads, d_model, n_heads, dtype_bytes=2):
    # Total bytes for K and V caches
    d_head = d_model // n_heads
    per_token = n_layers * kv_heads * d_head * 2  # K + V
    return batch * seq_len * per_token * dtype_bytes

configs = [
    ("GPT-3 175B (MHA)",  12288, 96, 96,  96),
    ("LLaMA-3 8B (GQA)",   4096, 32, 32,   8),
    ("LLaMA-3 70B (GQA)",  8192, 80, 64,   8),
    ("Mistral 7B (GQA)",   4096, 32, 32,   8),
    ("Falcon-7B (MQA)",    4096, 32, 71,   1),
]

batch, seq = 1, 8192
print(f"KV cache at batch={batch}, seq_len={seq:,} (BF16 = 2 bytes per element)")
print(f"{'Model':24s} {'Cache (MB)':>12s} {'vs GPT-3 MHA':>14s}")
print("-" * 54)

ref = None
for name, d, L, H, kv in configs:
    mb = kv_cache_bytes(batch, seq, L, kv, d, H) / 1e6
    if ref is None: ref = mb
    pct = mb / ref * 100
    print(f"{name:24s} {mb:10.1f} MB {pct:12.1f}%")

print()
print("LLaMA-3 8B at extended context lengths:")
print(f"{'seq_len':>10s} {'KV cache GB':>14s} {'vs model weights (16GB)':>24s}")
print("-" * 52)
model_gb = 8e9 * 2 / 1e9  # 8B params * 2 bytes BF16
for seq in [4096, 16384, 65536, 131072]:
    kv_gb = kv_cache_bytes(1, seq, 32, 8, 4096, 32) / 1e9
    print(f"{seq:10,d} {kv_gb:12.2f} GB {kv_gb/model_gb*100:22.0f}%")

# Try changing kv_heads to 32 (full MHA) to see memory balloon
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1911.02150" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Shazeer, 2019 &#8212; "Fast Transformer Decoding: One Write-Head is All You Need"</span> &#8212; Multi-Query Attention: single K/V head for all queries. Reduces KV cache by 8&#8211;32&#215; at modest quality cost.</span></a>
      <a href="https://arxiv.org/abs/2305.13245" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Ainslie et al., 2023 &#8212; "GQA: Training Generalized Multi-Query Transformer Models"</span> &#8212; GQA: interpolates between MHA and MQA. Matches MHA quality while approaching MQA inference speed.</span></a>
      <a href="https://arxiv.org/abs/1905.09418" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Voita et al., 2019 &#8212; "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting"</span> &#8212; Most attention heads can be pruned; 2&#8211;5 specialised heads do the meaningful work.</span></a>
      <a href="https://arxiv.org/abs/1906.04341" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Clark et al., 2019 &#8212; "What Does BERT Look At? An Analysis of BERT's Attention"</span> &#8212; Probing heads: specific heads track syntax, coreference, and sentence boundaries.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.7 == -->
<section class="section" id="s7">
  <div class="section-label">1.7</div>
  <h2>Positional Encoding &amp; RoPE</h2>

  <p>Self-attention is permutation-invariant &#8212; if you shuffle all token positions, the attention weights
  change but the <em>form</em> of the computation is identical. Word order must be injected explicitly. The
  original transformer used fixed <strong>sinusoidal encoding</strong>: PE(pos,2i) = sin(pos/10000<sup>2i/d</sup>),
  PE(pos,2i+1) = cos(&#8230;). The wavelengths form a geometric sequence from 2&#960; to 10000&#183;2&#960;.
  The key property: a relative offset PE(pos+k) can be expressed as a linear function of PE(pos), which the
  model can learn to exploit.</p>

  <p><strong>Rotary Position Embedding (RoPE)</strong> (Su et al., 2022, arXiv:2104.09864) is the current
  standard, used in LLaMA, Mistral, GPT-NeoX, Falcon, and most recent models. RoPE encodes position by
  rotating Q and K vectors by an angle proportional to position: q_m = R(m)&#183;q, k_n = R(n)&#183;k.
  When the dot product Q&#183;K is computed, the result depends only on the <em>relative</em> distance (m-n),
  not absolute positions. This gives zero-shot generalisation to unseen sequence lengths &#8212; a critical
  property for context extension.</p>

  <div class="callout purple">
    <strong>YaRN extends LLaMA from 4K &#8594; 128K context</strong> using only 0.1% of original pretraining
    compute (Peng et al., 2023, arXiv:2309.00071). It rescales RoPE frequencies differently for low-frequency
    (high-dimension) components versus high-frequency ones, avoiding the &#8220;lost in the middle&#8221;
    failure mode of naive position interpolation. LLaMA-3 increases the RoPE base from 10,000 (standard) to
    500,000 for its 128K-context variants &#8212; making the wavelengths dramatically longer and more robust
    to extended sequence lengths.
  </div>

  <div style="margin:24px 0;">
    <div style="font-family:var(--font-mono);font-size:10px;color:var(--text-secondary);margin-bottom:6px;">SINUSOIDAL PE HEATMAP &#8212; rows=position (0&#8211;15), cols=embedding dim (0&#8211;31)</div>
    <div class="pe-heatmap" id="pe-heatmap"></div>
    <div class="pe-axis"><span>pos 0</span><span>pos 15</span></div>
  </div>

  <div class="formula">
    <span class="eq-label">RoPE: relative position emerges from the dot product</span>
    q_rot(m) = R(m)&#183;q,&nbsp;&nbsp; k_rot(n) = R(n)&#183;k
    <br>&lt;q_rot(m), k_rot(n)&gt; = f(q, k, m&#8722;n) &#8212; depends only on relative offset
  </div>

  <div class="cell" id="cell-7">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; RoPE: relative position from dot product</span>
      <button class="run-btn" onclick="runCell('cell-7')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random
random.seed(3)

def rope_rotate(x, pos, base=10000):
    # Apply RoPE to a vector x at position pos
    d = len(x)
    half = d // 2
    result = [0.0] * d
    for i in range(half):
        theta = 1.0 / (base ** (2*i / d))   # frequency for dimension i
        angle = pos * theta
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Rotate pair (x[i], x[i+half]) by angle
        result[i]      = x[i] * cos_a - x[i+half] * sin_a
        result[i+half] = x[i] * sin_a + x[i+half] * cos_a
    return result

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

d = 8
q = [random.gauss(0, 1) for _ in range(d)]
k = [random.gauss(0, 1) for _ in range(d)]

print("RoPE dot products: same relative distance -> similar value")
print(f"{'Positions':20s} {'Offset':>8s} {'Dot product':>14s}")
print("-" * 46)
for m, n in [(0,0), (1,1), (5,5), (3,0), (8,5), (20,17), (100,97)]:
    qr = rope_rotate(q, m)
    kr = rope_rotate(k, n)
    d_val = dot(qr, kr)
    print(f"  pos ({m:3d},{n:3d})       {m-n:+6d}     {d_val:12.5f}")

print()
print("Absolute position is invisible; only relative offset (m-n) matters.")

# Show wavelength spectrum: high dims have very long wavelengths
print(f"\nRoPE wavelength per dimension (d={d}, base=10000):")
print("(longer wavelength = changes more slowly with position)")
for i in range(d//2):
    theta = 1.0 / (10000 ** (2*i / d))
    wavelength = 2 * math.pi / theta
    print(f"  dim {i:2d}: theta={theta:.6f}  wavelength={wavelength:10.1f} positions")

# LLaMA-3 uses base=500000 for long context
print(f"\nWith base=500000 (LLaMA-3 128K context), dim 0 wavelength:")
theta_new = 1.0 / (500000 ** (0 / d))
wl_new = 2 * math.pi / (1.0 / (500000 ** (0 / d))) if theta_new != 0 else float('inf')
# Actually compute it properly:
theta_base10k  = 1.0 / (10000  ** (0 / d))  # = 1.0 for i=0
theta_base500k = 1.0 / (500000 ** (0 / d))  # = 1.0 for i=0
# For i>0:
i = 1
t_10k  = 1.0 / (10000  ** (2*i / d))
t_500k = 1.0 / (500000 ** (2*i / d))
print(f"  dim {i}, base=10000:  wavelength={2*math.pi/t_10k:.1f}")
print(f"  dim {i}, base=500000: wavelength={2*math.pi/t_500k:.1f}")
print("  Longer wavelengths prevent position confusion at 128K+ tokens.")

# Try changing base to 500000 and the position pairs to see long-range consistency
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2104.09864" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Su et al., 2022 &#8212; "RoFormer: Enhanced Transformer with Rotary Position Embedding"</span> &#8212; RoPE: relative position via vector rotation. Relative distance emerges from dot product naturally.</span></a>
      <a href="https://arxiv.org/abs/2108.12409" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Press et al., 2022 &#8212; "Train Short, Test Long: Attention with Linear Biases" (ALiBi)</span> &#8212; Alternative to RoPE: bias attention scores by linear position penalty. No positional vectors needed.</span></a>
      <a href="https://arxiv.org/abs/2309.00071" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Peng et al., 2023 &#8212; "YaRN: Efficient Context Window Extension of Large Language Models"</span> &#8212; Extends LLaMA from 4K to 128K context using 0.1% pretraining compute by rescaling RoPE frequencies.</span></a>
      <a href="https://arxiv.org/abs/2306.15595" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Chen et al., 2023 &#8212; "Extending Context Window via Positional Interpolation"</span> &#8212; Simple linear scaling of RoPE position indices; enables length extrapolation with fine-tuning.</span></a>
      <a href="https://blog.eleuther.ai/rotary-embeddings/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">EleutherAI &#8212; "Rotary Embeddings: A Relative Revolution"</span> &#8212; Detailed mathematical walkthrough of RoPE's rotation derivation and its advantages over learned PE.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.8 == -->
<section class="section" id="s8">
  <div class="section-label">1.8</div>
  <h2>The Feedforward Sublayer</h2>

  <p>After attention routes information between positions, the feedforward network (FFN) processes each token
  independently and in parallel. The original FFN applies two linear projections with a nonlinearity:
  FFN(x) = W&#8322; &#183; GELU(W&#8321; &#183; x). The intermediate dimension is 4&#215; d_model:
  for LLaMA-3 8B that's 4,096 &#8594; 16,384 &#8594; 4,096.</p>

  <p>Geva et al. (2021, arXiv:2012.14913) made a striking discovery: <strong>FFN layers act as key-value
  memories</strong>. The first projection computes &#8220;keys&#8221;; the GELU selects which keys are active;
  the second projection reads out corresponding &#8220;values.&#8221; Specific factual knowledge
  (&#8220;Paris is the capital of France&#8221;) can be traced to specific neurons in specific FFN layers.
  Lower layers store syntactic patterns; upper layers store semantic facts. Meng et al. (2022, ROME) exploited
  this to <em>directly edit factual associations</em> by modifying a single FFN weight row.</p>

  <div class="callout">
    <strong>SwiGLU</strong> (Shazeer, 2020, arXiv:2002.05202) replaces GELU with a gated variant:
    FFN(x) = W&#8322; &#183; (SiLU(W&#8321;&#183;x) &#8855; W&#8323;&#183;x). The gate acts as a learned
    router. SwiGLU requires 3 matrices instead of 2, so the intermediate dimension is reduced to approximately
    2/3 of the standard 4&#215; value to keep parameter counts equal. LLaMA-3 8B uses d_ff=14,336
    (&#8776; 2/3 &#215; 4 &#215; 4,096, rounded to a multiple of 256 for GPU efficiency). Shazeer showed
    GLU variants consistently outperform standard GELU/ReLU across model scales.
  </div>

  <div class="cell" id="cell-8">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; FFN as key-value memory + SwiGLU comparison</span>
      <button class="run-btn" onclick="runCell('cell-8')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random
random.seed(42)

def gelu(x):
    # GELU: Gaussian Error Linear Unit (Hendrycks & Gimpel 2016)
    return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715*x**3)))

def silu(x):
    # SiLU (Sigmoid Linear Unit): x * sigmoid(x)
    return x / (1 + math.exp(-x))

def randn(): return random.gauss(0, 0.02)

d, d_ff = 8, 32   # small demo (real: 4096, 14336)

W1 = [[randn() for _ in range(d_ff)] for _ in range(d)]
W2 = [[randn() for _ in range(d)]    for _ in range(d_ff)]
W3 = [[randn() for _ in range(d_ff)] for _ in range(d)]  # SwiGLU gate

def matvec(W, x):
    return [sum(W[i][j]*x[j] for j in range(len(x))) for i in range(len(W))]

x = [random.gauss(0, 1) for _ in range(d)]

# Standard GELU FFN
h_gelu  = [gelu(v) for v in matvec(W1, x)]
out_gelu = matvec([[W2[i][j] for i in range(d_ff)] for j in range(d)], h_gelu)
# (using transposed W2 for W2*h)
W2t = [[W2[i][j] for i in range(d_ff)] for j in range(d)]
out_gelu2 = [sum(W2t[j][i]*h_gelu[i] for i in range(d_ff)) for j in range(d)]

# SwiGLU
gate = [silu(v)  for v in matvec(W1, x)]   # gating path
up   = matvec(W3, x)                        # value path
h_swiglu = [gate[i]*up[i] for i in range(d_ff)]
W2t2 = [[W2[i][j] for i in range(d_ff)] for j in range(d)]
out_swiglu = [sum(W2t2[j][i]*h_swiglu[i] for i in range(d_ff)) for j in range(d)]

print(f"FFN demo: d={d}, d_ff={d_ff} ({d_ff//d}x expansion)")
print()

# Key-value memory: which neurons fire?
active_gelu   = sum(1 for v in h_gelu   if v > 0.01)
active_swiglu = sum(1 for v in h_swiglu if abs(v) > 0.01)
print("Neurons active after nonlinearity (Geva et al. 2021 insight):")
print(f"  GELU:   {active_gelu:3d}/{d_ff} active ({active_gelu/d_ff:.0%})")
print(f"  SwiGLU: {active_swiglu:3d}/{d_ff} active ({active_swiglu/d_ff:.0%})")
print("  (Sparse activation = only relevant 'memories' accessed)")

print()
print("Parameter counts (same d, d_ff):")
std_params    = d*d_ff + d_ff*d
swiglu_params = d*d_ff*2 + d_ff*d   # gate + up + down
print(f"  Standard (W1+W2):      {std_params:6d} params")
print(f"  SwiGLU (W1+W2+W3):     {swiglu_params:6d} params (+{(swiglu_params-std_params)/std_params:.0%})")

print()
print("LLaMA-3 8B FFN: why d_ff=14336 instead of 4*4096=16384?")
d_model = 4096
d_ff_std   = d_model * 4
d_ff_swiglu = int(2/3 * d_ff_std)  # naive 2/3
d_ff_rounded = (d_ff_swiglu // 256) * 256  # round to 256 for GPU efficiency
print(f"  Standard 4x:            d_ff = {d_ff_std}")
print(f"  SwiGLU 2/3 x 4x:        d_ff = {d_ff_swiglu}")
print(f"  Rounded to 256 boundary: d_ff = {d_ff_rounded}")
print(f"  LLaMA-3 actual:          d_ff = 14336  (manually tuned)")
print()
std_llama   = d_model*d_ff_std   + d_ff_std*d_model
swi_llama   = d_model*14336*2    + 14336*d_model
print(f"  Params/layer standard FFN: {std_llama/1e6:.1f}M")
print(f"  Params/layer SwiGLU FFN:   {swi_llama/1e6:.1f}M")

# Try changing d_ff to 16384 (standard 4x) to see param inflation
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2012.14913" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Geva et al., 2021 &#8212; "Transformer Feed-Forward Layers Are Key-Value Memories"</span> &#8212; FFN lower layers store syntactic patterns; upper layers store semantic facts. Neurons are human-interpretable.</span></a>
      <a href="https://arxiv.org/abs/2002.05202" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Shazeer, 2020 &#8212; "GLU Variants Improve Transformer"</span> &#8212; SwiGLU: gated activation consistently outperforms GELU/ReLU. Used in LLaMA, PaLM, GPT-4 (likely).</span></a>
      <a href="https://arxiv.org/abs/2202.05262" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Meng et al., 2022 &#8212; "Locating and Editing Factual Associations in GPT" (ROME)</span> &#8212; Edit specific facts by modifying a single FFN weight row in the right layer. Validates the key-value hypothesis.</span></a>
      <a href="https://arxiv.org/abs/2104.08696" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Dai et al., 2022 &#8212; "Knowledge Neurons in Pretrained Transformers"</span> &#8212; Locates specific neurons that store individual factual associations; activating them retrieves the fact.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
""")

with open(OUT, 'a', encoding='utf-8') as f:
    f.write(''.join(lines))
print(f"S6+S7+S8 appended: {sum(len(l) for l in lines)} chars")
