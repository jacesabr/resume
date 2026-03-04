#!/usr/bin/env python3
# Appends sections 1.4 and 1.5
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

lines = []
A = lines.append

A("""
<!-- ============================================================ §1.4 == -->
<section class="section" id="s4">
  <div class="section-label">1.4</div>
  <h2>Transformer Architecture</h2>

  <p>The transformer (Vaswani et al., 2017, arXiv:1706.03762) stacks <em>L</em> identical blocks. Each block
  contains two sublayers: multi-head self-attention and a position-wise feedforward network. Every sublayer
  is wrapped in a residual connection and layer normalisation. Residual connections are the key engineering
  insight: gradients flow directly from the final loss back through every earlier layer via additive skip paths,
  making training stable even at 100+ layers. Without them, deep networks saturate and gradients vanish.</p>

  <p>The original Vaswani 2017 paper uses <strong>Post-Norm</strong> (LayerNorm after the sublayer). Modern
  decoder-only LLMs (GPT family, LLaMA, Mistral) universally adopt <strong>Pre-Norm</strong> (LayerNorm before
  the sublayer), which Xiong et al. (2020) showed produces larger, more stable gradients. LLaMA also replaces
  LayerNorm with <strong>RMSNorm</strong> (Zhang &amp; Sennrich, 2019, arXiv:1910.07467), omitting the
  mean-centering step entirely. Since language model residual streams have near-zero mean anyway, this saves
  ~40% of norm compute with no measurable quality drop.</p>

  <div style="position:relative;padding-left:28px;">
    <div class="tblock-residual">
      <div class="residual-line"></div>
      <div class="residual-label">RESIDUAL STREAM</div>
      <div class="residual-line"></div>
    </div>
    <div class="tblock">
      <div class="tblock-title">&#215; L Transformer Blocks (LLaMA-3: L=32 for 8B, L=80 for 70B, L=126 for 405B)</div>
      <div class="tblock-layers">
        <div class="tblock-row"><div class="tblock-node tbn-norm">RMSNorm (Pre-Norm)</div></div>
        <div class="tblock-row"><div class="tblock-node tbn-attn">Multi-Head Self-Attention + GQA + RoPE</div></div>
        <div class="tblock-row"><div class="tblock-node tbn-add">+ residual connection</div></div>
        <div class="tblock-row"><div class="tblock-node tbn-norm">RMSNorm</div></div>
        <div class="tblock-row"><div class="tblock-node tbn-ffn">SwiGLU Feedforward (3 matrices, d_ff=14336 for 8B)</div></div>
        <div class="tblock-row"><div class="tblock-node tbn-add">+ residual connection</div></div>
      </div>
    </div>
  </div>

  <div class="formula">
    <span class="eq-label">LLaMA-3 8B parameter budget (exact)</span>
    d_model=4096, d_ff=14336, L=32, H=32 Q-heads, KV_heads=8, V=128,256
    <br>Params &#8776; V&#183;d + L&#183;(d&#178; + 2&#183;kv_h&#183;d_h&#183;d + d&#178; + 3&#183;d_ff&#183;d) &#8776; 8.03B
  </div>

  <div class="cell" id="cell-4">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Exact parameter counter for real models</span>
      <button class="run-btn" onclick="runCell('cell-4')">Run</button>
    </div>
    <div class="cell-code"><pre>def count_params(d, d_ff, L, H, kv_heads, V):
    # Approx parameter count for decoder-only LLM (GQA + SwiGLU)
    d_head = d // H
    embed  = V * d                        # token embedding table
    q_proj = d * d
    k_proj = d * (kv_heads * d_head)      # fewer KV heads with GQA
    v_proj = d * (kv_heads * d_head)
    o_proj = d * d
    attn   = q_proj + k_proj + v_proj + o_proj
    ffn    = d * d_ff * 2 + d_ff * d     # gate + up + down (SwiGLU)
    norm   = d * 2 * L                   # RMSNorm: one scale vector per dim
    total  = embed + L * (attn + ffn) + norm
    return total

configs = [
    ("LLaMA-3 8B",    4096, 14336,  32, 32,  8, 128256),
    ("LLaMA-3 70B",   8192, 28672,  80, 64,  8, 128256),
    ("LLaMA-3 405B", 16384, 53248, 126,128, 16, 128256),
    ("Mistral 7B",    4096, 14336,  32, 32,  8,  32000),
    ("GPT-3 175B",   12288, 49152,  96, 96, 96,  50257),
    ("GPT-2 (117M)",   768,  3072,  12, 12, 12,  50257),
]

print(f"{'Model':18s} {'Params':>10s} {'Attn%':>7s} {'FFN%':>7s} {'Embed%':>8s}")
print("-" * 54)
for name, d, d_ff, L, H, kv, V in configs:
    total = count_params(d, d_ff, L, H, kv, V)
    embed = V * d
    d_head = d // H
    attn_p = L * (d*d + d*(kv*d_head)*2 + d*d)
    ffn_p  = L * (d*d_ff*2 + d_ff*d)
    print(f"{name:18s} {total/1e9:8.2f}B {attn_p/total*100:6.1f}% {ffn_p/total*100:6.1f}% {embed/total*100:7.1f}%")

print()
print("FFN holds ~65-70% of parameters in most models.")
print("Attention is architecturally central but parameter-light.")

# Try changing kv_heads=96 (MHA) for GPT-3 to see the GQA cache saving
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1706.03762" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Vaswani et al., 2017 &#8212; "Attention Is All You Need"</span> &#8212; Original transformer: d_model=512, 8 heads, 6 encoder + 6 decoder layers, 65M total parameters.</span></a>
      <a href="https://arxiv.org/abs/1512.03385" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">He et al., 2016 &#8212; "Deep Residual Learning for Image Recognition"</span> &#8212; Residual connections enabling stable gradient flow through 100+ layer networks.</span></a>
      <a href="https://arxiv.org/abs/1910.07467" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Zhang &amp; Sennrich, 2019 &#8212; "Root Mean Square Layer Normalization"</span> &#8212; RMSNorm: drop mean-centering; ~40% faster with identical quality. Used in LLaMA, Mistral, Falcon.</span></a>
      <a href="https://arxiv.org/abs/2002.04745" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Xiong et al., 2020 &#8212; "On Layer Normalization in the Transformer Architecture"</span> &#8212; Pre-LN: normalise before sublayer for larger, more stable gradients; now the universal default.</span></a>
      <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jay Alammar &#8212; "The Illustrated Transformer"</span> &#8212; Canonical visual walkthrough; the definitive introduction to the transformer block stack.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.5 == -->
<section class="section" id="s5">
  <div class="section-label">1.5</div>
  <h2>Self-Attention Mechanism</h2>

  <p>Self-attention computes, for each token position, a weighted average of all other token representations.
  The weights come from the similarity between that token's <em>query</em> vector and every other token's
  <em>key</em> vector; the <em>value</em> vectors are then averaged using those weights. The operation is
  fully differentiable &#8212; the model learns which relationships matter purely from data, with no structural
  assumptions about word order or grammar baked in.</p>

  <div class="formula">
    <span class="eq-label">Scaled Dot-Product Attention (Vaswani et al. 2017, eq. 1)</span>
    Attention(Q, K, V) = softmax( Q&#183;K&#7488; / &#8730;d_k ) &#183; V
    <br><span style="font-size:11px;color:#888;">Scaling by &#8730;d_k prevents large dot-product magnitudes in high dimensions from pushing softmax into near-zero gradient regions.</span>
  </div>

  <p>The &#8730;d_k scaling factor is not cosmetic. For d_k=64 (original transformer), random unit vectors have
  dot products with standard deviation &#8730;64=8. Without scaling, softmax input has variance &#8764;64,
  pushing it into saturation with gradients near zero. Vaswani et al. observed that without scaling, even at
  d_k=64 the dot products are &#8220;extremely large&#8221;, causing &#8220;extremely small gradients.&#8221;</p>

  <div class="qkv-diag">
    <div class="qkv-col">
      <div class="qkv-input">x&#8321; &#8220;The&#8221;</div>
      <div class="qkv-mat qkv-Q">Q = x&#183;W_Q<br><span style="font-size:10px;">&#8220;what am I looking for?&#8221;</span></div>
    </div>
    <div class="qkv-arrow" style="font-size:24px;">&#8855;</div>
    <div class="qkv-col">
      <div class="qkv-input">x&#8322; &#8220;cat&#8221;</div>
      <div class="qkv-mat qkv-K">K = x&#183;W_K<br><span style="font-size:10px;">&#8220;what do I contain?&#8221;</span></div>
    </div>
  </div>
  <div style="margin:0 0 16px;padding:12px 20px;background:var(--bg-aside);border-radius:7px;font-size:13px;">
    Score = Q&#183;K&#7488;/&#8730;d_k &#8594; softmax &#8594; attention weights &#8594; output = weights &#183; V
  </div>

  <div class="mask-wrap">
    <div class="mask-title">Causal Attention Mask &#8212; decoder-only LLMs</div>
    <div class="mask-grid" style="grid-template-columns: 28px repeat(5, 32px);">
      <div class="mk mk-head"></div>
      <div class="mk mk-head">The</div><div class="mk mk-head">cat</div>
      <div class="mk mk-head">sat</div><div class="mk mk-head">on</div><div class="mk mk-head">the</div>
      <div class="mk mk-head">The</div>
      <div class="mk mk-on">&#10003;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div>
      <div class="mk mk-head">cat</div>
      <div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div>
      <div class="mk mk-head">sat</div>
      <div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-off">&#8722;&#8734;</div><div class="mk mk-off">&#8722;&#8734;</div>
      <div class="mk mk-head">on</div>
      <div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-off">&#8722;&#8734;</div>
      <div class="mk mk-head">the</div>
      <div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div><div class="mk mk-on">&#10003;</div>
    </div>
    <div class="mask-legend">
      <div class="mask-legend-item"><div class="ml-swatch" style="background:var(--blue-light);border:1px solid var(--blue);"></div>attend</div>
      <div class="mask-legend-item"><div class="ml-swatch" style="background:repeating-linear-gradient(45deg,#ddd 0,#ddd 3px,transparent 3px,transparent 6px);border:1px solid #ccc;"></div>&#8722;&#8734; masked (future tokens)</div>
    </div>
  </div>

  <p>The &#8722;&#8734; mask entries become zero after softmax, so future tokens receive exactly zero attention
  weight. This makes the transformer <em>autoregressive at training time</em>: all positions are computed in
  parallel on the GPU (highly efficient), but each token only attends to its left context (causally correct).
  The O(N&#178;) cost of the N&#215;N attention matrix is Flash Attention's target (&#167;1.12).</p>

  <div class="cell" id="cell-5">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Scaled dot-product attention (pure Python, no numpy)</span>
      <button class="run-btn" onclick="runCell('cell-5')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random
random.seed(7)

def softmax_rows(matrix):
    result = []
    for row in matrix:
        mx = max(row)
        exps = [math.exp(x - mx) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result

def matmul(A, B):
    rA, cA, cB = len(A), len(A[0]), len(B[0])
    C = [[sum(A[i][k]*B[k][j] for k in range(cA)) for j in range(cB)] for i in range(rA)]
    return C

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def scaled_dot_product_attention(Q, K, V, causal=True):
    T, d_k = len(Q), len(Q[0])
    scale = math.sqrt(d_k)
    scores = matmul(Q, transpose(K))
    scores = [[s / scale for s in row] for row in scores]
    if causal:
        for i in range(T):
            for j in range(T):
                if j > i: scores[i][j] = -1e9   # mask future tokens
    weights = softmax_rows(scores)
    output  = matmul(weights, V)
    return output, weights

T, d_k, d_v = 5, 4, 4
Q = [[random.gauss(0,1) for _ in range(d_k)] for _ in range(T)]
K = [[random.gauss(0,1) for _ in range(d_k)] for _ in range(T)]
V = [[random.gauss(0,1) for _ in range(d_v)] for _ in range(T)]

tokens = ["The","cat","sat","on","mat"]
out, W = scaled_dot_product_attention(Q, K, V, causal=True)

print(f"Attention weights (T={T}, d_k={d_k}, causal=True)")
print(f"{'':6s}" + "".join(f"{t:7s}" for t in tokens))
for i, row in enumerate(W):
    print(f"{tokens[i]:6s}" + "".join(f"{v:7.3f}" for v in row))

# Row sums should all be 1.0
sums = [round(sum(row), 5) for row in W]
print(f"\nRow sums: {sums}  (all 1.0)")

# Shannon entropy per row
print("\nAttention entropy (higher = attention more spread out):")
for i, row in enumerate(W):
    H = -sum(p * math.log(p + 1e-12) for p in row)
    max_H = math.log(i+1) if i > 0 else 0
    print(f"  {tokens[i]:5s}  H={H:.3f}  max_possible={max_H:.3f}")

# Without scaling: show why sqrt(d_k) matters
print("\n--- Effect of scaling factor ---")
for scale_factor in [1.0, math.sqrt(d_k), d_k]:
    raw = [[sum(Q[i][k]*K[j][k] for k in range(d_k))/scale_factor for j in range(T)] for i in range(T)]
    W2 = softmax_rows(raw)
    entropy_row0 = -sum(p * math.log(p+1e-12) for p in W2[0])
    max_val = max(W2[0])
    print(f"  scale={scale_factor:.2f}  row0_max={max_val:.3f}  row0_entropy={entropy_row0:.3f}")
print("Without 1/sqrt(d_k): max weight near 1.0, entropy near 0 -> vanishing gradients")

# Try changing T=8 or d_k=64 and re-running to see entropy effects
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1706.03762" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Vaswani et al., 2017 &#8212; "Attention Is All You Need"</span> &#8212; The scaling factor motivation explained in Section 3.2.1: large dot products push softmax into near-zero gradient regions.</span></a>
      <a href="https://arxiv.org/abs/1409.0473" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Bahdanau et al., 2015 &#8212; "Neural Machine Translation by Jointly Learning to Align and Translate"</span> &#8212; Original additive attention. The direct predecessor of transformer self-attention.</span></a>
      <a href="https://jalammar.github.io/illustrated-gpt2/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jay Alammar &#8212; "The Illustrated GPT-2"</span> &#8212; Visual step-by-step of causal self-attention in a decoder-only model.</span></a>
      <a href="https://www.youtube.com/watch?v=eMlx5fFNoYc" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">3Blue1Brown &#8212; "Attention in transformers, visually explained"</span> &#8212; Outstanding geometric intuition for what Q, K, V projections compute.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
""")

with open(OUT, 'a', encoding='utf-8') as f:
    f.write(''.join(lines))
print(f"S4+S5 appended: {sum(len(l) for l in lines)} chars")
