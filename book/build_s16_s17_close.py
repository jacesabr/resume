#!/usr/bin/env python3
# Appends sections 1.16, 1.17, and the closing JS/Pyodide + </main></body></html>
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

lines = []
A = lines.append

A("""
<!-- ============================================================ §1.16 == -->
<section class="section" id="s16">
  <div class="section-label">1.16</div>
  <h2>Quantisation &amp; Deployment Efficiency</h2>

  <p>A full-precision (float32) parameter occupies 4 bytes; float16/bfloat16 cuts this to 2 bytes; int8 to 1 byte; int4 to 0.5 bytes. LLaMA-3 70B in float16 requires ~140 GB of GPU VRAM — four A100-40GB cards with nothing to spare. In 4-bit quantisation (GGUF, GPTQ, AWQ), the same model fits in ~35 GB — a single A100 or two 24 GB consumer GPUs. The quality cost is surprisingly small: perplexity typically degrades &lt;1% at int8, &lt;2–3% at int4 with modern algorithms.</p>

  <p><strong>Post-Training Quantisation (PTQ)</strong> converts a trained float16 model without re-training. Dettmers et al. (2022, arXiv:2208.07339) introduced LLM.int8(), using vector-wise quantisation with mixed-precision decomposition for emergent outlier features. Outlier features (a small fraction of dimensions with magnitudes 60–100× larger than average) blow up INT8 quantisation — LLM.int8() detects these dimensions and keeps them in float16 while quantising the rest to int8.</p>

  <p><strong>NF4 — Normal Float 4-bit.</strong> Dettmers et al. (2023, arXiv:2305.14314, QLoRA paper) introduced NF4, an information-theoretically optimal 4-bit data type for normally-distributed weights. Standard int4 has uniformly-spaced bins; NF4 places bins so that each quantile of a standard normal distribution occupies equal probability mass. For weights that are (approximately) normally distributed, NF4 minimises the expected quantisation error. QLoRA = NF4 quantisation + LoRA fine-tuning + double quantisation of the quantisation constants = fine-tuning a 65B model on a single 48 GB A40.</p>

  <div class="callout">
    <strong>NF4 quantile levels</strong> — 16 values covering the standard normal distribution's quantiles (each covers 1/16 of the probability mass). Contrast with uniform int4 [−8,−7,...,+7]: NF4 clusters values near zero (where most weights lie) and spreads more near tails.
    <br><br>Dequantisation: each 4-bit code maps to its NF4 centroid, scaled by the block's absmax value. Block size = 64 (64 weights share one absmax constant). Double quantisation: the absmax constants themselves are further quantised to 8-bit, saving ~0.5 bits/parameter.
  </div>

  <p><strong>GGUF &amp; llama.cpp.</strong> For CPU/GPU inference on consumer hardware, llama.cpp (Gerganov, 2023) implements its own quantisation formats: Q4_K_M means 4-bit, K-quant (groups of 256 with enhanced precision on important weights), M=medium (some layers kept at 6-bit). A llama.cpp Q4_K_M of LLaMA-3 70B fits in ~42 GB; Q5_K_M fits in ~52 GB with meaningfully better quality. Inference speed on an M2 MacBook Pro: ~8 tokens/second for 8B, ~1–2 t/s for 70B.</p>

  <p><strong>AWQ (Activation-Aware Weight Quantisation).</strong> Lin et al. (2023, arXiv:2306.00978) observed that not all weights matter equally — 0.1% of channels account for most quantisation error because input activations are large at those channels. AWQ protects these salient channels via per-channel scaling before quantisation, achieving better quality than GPTQ at similar speed.</p>

  <div class="ar-loop" style="padding:16px;">
    <div style="margin-bottom:8px; color:#888; font-size:0.75rem;">LLAMA-3 70B: VRAM REQUIREMENTS BY QUANTISATION</div>
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:6px; font-size:0.72rem; text-align:center;">
""")

quant_data = [
    ("float32", "4 B/param", "280 GB", "7×A100", "#333"),
    ("float16", "2 B/param", "140 GB", "4×A100-40G", "#444"),
    ("int8 LLM.int8()", "1 B/param", "70 GB", "2×A100-40G", "#4ec9b0"),
    ("NF4 QLoRA", "0.5 B/param", "~40 GB", "1×A100-80G", "#569cd6"),
    ("Q4_K_M GGUF", "~0.5 B/param", "~42 GB", "M2 Mac", "#c586c0"),
]
for name, bpp, vram, hw, color in quant_data:
    A(f"""      <div style="background:#1a1a2e; border:1px solid {color}; border-radius:4px; padding:8px;">
        <div style="color:{color}; font-weight:700; margin-bottom:4px; font-size:0.68rem;">{name}</div>
        <div style="color:#ccc;">{bpp}</div>
        <div style="color:#fff; font-weight:600;">{vram}</div>
        <div style="color:#888; font-size:0.65rem;">{hw}</div>
      </div>
""")

A("""    </div>
  </div>

  <div class="code-block">
    <span class="code-label">Python</span>
    <pre><span class="kw">import</span> math

<span class="lm"># NF4 quantile levels (16 values, each covers 1/16 of N(0,1) probability mass)</span>
<span class="kw">def</span> <span class="fn">normal_ppf</span>(p):
    <span class="lm"># ↑ Beasley-Springer-Moro rational approximation of inverse normal CDF</span>
    <span class="kw">if</span> p <= <span class="nu">0.0</span> <span class="kw">or</span> p >= <span class="nu">1.0</span>: <span class="kw">raise</span> ValueError
    c = [<span class="nu">2.515517</span>, <span class="nu">0.802853</span>, <span class="nu">0.010328</span>]
    d = [<span class="nu">1.432788</span>, <span class="nu">0.189269</span>, <span class="nu">0.001308</span>]
    <span class="kw">if</span> p < <span class="nu">0.5</span>: sign, q = -<span class="nu">1</span>, math.sqrt(-<span class="nu">2</span>*math.log(p))
    <span class="kw">else</span>:       sign, q = +<span class="nu">1</span>, math.sqrt(-<span class="nu">2</span>*math.log(<span class="nu">1</span>-p))
    num = c[<span class="nu">0</span>] + c[<span class="nu">1</span>]*q + c[<span class="nu">2</span>]*q*q
    den = <span class="nu">1</span> + d[<span class="nu">0</span>]*q + d[<span class="nu">1</span>]*q*q + d[<span class="nu">2</span>]*q*q*q
    <span class="kw">return</span> sign * (q - num/den)

nf4_levels = [normal_ppf(i/<span class="nu">16</span> + <span class="nu">1</span>/<span class="nu">32</span>) <span class="kw">for</span> i <span class="kw">in</span> <span class="bi">range</span>(<span class="nu">16</span>)]
<span class="lm"># ↑ midpoints of 16 equal-probability quantile buckets</span>
nf4_norm = [v / <span class="bi">max</span>(<span class="bi">abs</span>(v) <span class="kw">for</span> v <span class="kw">in</span> nf4_levels) <span class="kw">for</span> v <span class="kw">in</span> nf4_levels]   <span class="lm"># ↑ scale to [-1, +1]</span>
print(<span class="st">"NF4 quantile levels (16 bins, normalized to [-1, +1]):"</span>)
<span class="kw">for</span> i, v <span class="kw">in</span> <span class="bi">enumerate</span>(nf4_norm):
    bar = <span class="st">"#"</span> * <span class="bi">int</span>(<span class="bi">abs</span>(v) * <span class="nu">20</span>)
    print(f<span class="st">f"  [{i:2d}] {v:+.4f}  {bar}"</span>)</pre>
  </div>

  <div class="cell" id="cell-16">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; NF4 quantile bins vs uniform int4 + VRAM calculator</span>
      <button class="run-btn" onclick="runCell('cell-16')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

# Approximate inverse normal CDF (Beasley-Springer-Moro)
def normal_ppf(p):
    if p <= 0.0 or p >= 1.0: return 0.0
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    if p < 0.5:
        sign, q = -1, math.sqrt(-2*math.log(p))
    else:
        sign, q = 1, math.sqrt(-2*math.log(1-p))
    num = c[0] + c[1]*q + c[2]*q**2
    den = 1 + d[0]*q + d[1]*q**2 + d[2]*q**3
    return sign * (q - num/den)

# NF4: 16 bins at quantiles of N(0,1), normalized to [-1,+1]
nf4_raw = [normal_ppf(i/16 + 1/32) for i in range(16)]
scale = max(abs(v) for v in nf4_raw)
nf4 = [v / scale for v in nf4_raw]

# Uniform int4: 16 bins evenly spaced in [-1, +1]
int4 = [-1.0 + i * (2.0/15) for i in range(16)]

print("Bin comparison: NF4 vs Uniform INT4")
print(f"  {'Bin':>3}  {'NF4':>8}  {'INT4':>8}  {'Diff':>8}")
for i in range(16):
    diff = nf4[i] - int4[i]
    print(f"  {i:>3d}  {nf4[i]:>8.4f}  {int4[i]:>8.4f}  {diff:>+8.4f}")

print()
print("Key insight: NF4 clusters bins near 0 (where most weights live)")
print("INT4 wastes precision on extreme values that are rare")
print()

# VRAM calculator
print("VRAM usage for LLaMA-3 models:")
models = [
    ("8B",  8e9),
    ("70B", 70e9),
    ("405B", 405e9),
]
formats = [
    ("float32", 4.0),
    ("bfloat16", 2.0),
    ("int8", 1.0),
    ("NF4 (4-bit)", 0.5),
]
for model_name, params in models:
    print(f"\n  LLaMA-3 {model_name}:")
    for fmt_name, bytes_per in formats:
        vram_gb = params * bytes_per / 1e9
        a100s = math.ceil(vram_gb / 80)
        print(f"    {fmt_name:<14}  {vram_gb:>6.1f} GB  ({a100s}× A100-80G)")

# Try changing bytes_per to 0.375 for Q3 to see 3-bit quantisation impact</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2305.14314" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Dettmers et al., 2023 &#8212; QLoRA: Efficient Finetuning of Quantized LLMs</span> &#8212; Introduces NF4, double quantisation, paged optimizers. Fine-tunes 65B model on single 48GB GPU. Guanaco-65B beats ChatGPT on human eval.</span></a>
      <a href="https://arxiv.org/abs/2208.07339" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Dettmers et al., 2022 &#8212; LLM.int8(): 8-bit Matrix Multiplication at Scale</span> &#8212; Decomposed quantisation handling outlier features. &lt;1% degradation on all benchmarks at 8-bit. Integrated into bitsandbytes library.</span></a>
      <a href="https://arxiv.org/abs/2306.00978" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Lin et al., 2023 &#8212; AWQ: Activation-aware Weight Quantization for LLM Compression</span> &#8212; Finds 0.1% salient channels via activation statistics. Protects them with per-channel scaling. Better quality than GPTQ at 4-bit.</span></a>
      <a href="https://github.com/ggerganov/llama.cpp" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Gerganov, 2023 &#8212; llama.cpp (GitHub)</span> &#8212; Pure C/C++ inference engine. Supports CPU, Metal, CUDA, Vulkan. K-quant formats (Q4_K_M etc.) with mixed-precision blocks. Runs LLaMA-3 8B at real-time speed on a MacBook.</span></a>
      <a href="https://huggingface.co/docs/transformers/quantization" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">HuggingFace &#8212; Quantization Concepts (2024)</span> &#8212; Practical guide to GPTQ, AWQ, bitsandbytes, GGUF. Code examples for loading quantised models. Decision tree for which format to use.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.17 == -->
<section class="section" id="s17">
  <div class="section-label">1.17</div>
  <h2>The 2025–26 Landscape &amp; Open Problems</h2>

  <p>As of mid-2025, the frontier has shifted from raw capability to efficiency, reasoning, and multimodality. The GPT-4 release (March 2023) established a new capability ceiling; subsequent frontier models — Claude 3 Opus, Gemini 1.5 Pro, Llama 3.1 405B — have approached or matched it on most benchmarks at lower cost. The open-weight ecosystem has compressed the capability gap dramatically: Llama 3.1 405B (released July 2024) performs on par with GPT-4-class models while being freely downloadable.</p>

  <p><strong>Context length.</strong> The practical context window has expanded from 4K tokens (GPT-3) to 128K (GPT-4 Turbo), 1M (Gemini 1.5 Pro), and 2M in research settings. Key enabling techniques: Ring Attention (Liu et al., 2023, distributes attention across devices), YaRN (Peng et al., 2023, RoPE scaling for extrapolation), and sliding window attention (Mistral). The bottleneck is now KV cache memory, not model architecture.</p>

  <p><strong>Reasoning models.</strong> OpenAI's o1/o3, Google's Gemini 2.0 Flash Thinking, and DeepSeek-R1 (January 2025, arXiv:2501.12948) implement chain-of-thought reasoning at inference time — generating and evaluating multiple solution paths before returning an answer. DeepSeek-R1 achieved GPT-4o-level math and coding performance through pure reinforcement learning from outcome feedback (no human preference data), a landmark demonstrating that RLHF is not necessary for reasoning alignment.</p>

  <p><strong>Multimodality.</strong> GPT-4V, Gemini 1.5 Pro, Claude 3.5 Sonnet, and Llama 3.2 90B Vision accept images alongside text. The dominant architecture is a ViT (Vision Transformer) encoder whose patch embeddings are projected into the LLM's residual stream — the same architecture as CLIP (Radford et al., 2021). Native image generation within a unified model (not a separate diffusion model) remains an active research area (Chameleon, Transfusion, 2024).</p>

  <p><strong>Open problems at the 2025 frontier:</strong></p>

  <div class="ar-loop" style="padding:16px;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.8rem;">
      <div style="background:#1a1a2e; border:1px solid #c586c0; border-radius:6px; padding:12px;">
        <div style="color:#c586c0; font-weight:700; margin-bottom:6px;">Hallucination &amp; Calibration</div>
        <div>LLMs generate plausible-sounding falsehoods with high confidence. RAG, grounding via tool use, and constitutional AI partially mitigate but don't solve. Factuality remains the #1 deployment blocker for high-stakes applications (medicine, law).</div>
      </div>
      <div style="background:#1a1a2e; border:1px solid #4ec9b0; border-radius:6px; padding:12px;">
        <div style="color:#4ec9b0; font-weight:700; margin-bottom:6px;">Long-Context Faithfulness</div>
        <div>"Lost in the middle" (Liu et al., 2023): LLMs reliably use information at the start and end of a 128K context but systematically miss information buried in the middle. This is an attention sink problem, not a context-length problem per se.</div>
      </div>
      <div style="background:#1a1a2e; border:1px solid #569cd6; border-radius:6px; padding:12px;">
        <div style="color:#569cd6; font-weight:700; margin-bottom:6px;">Sample Efficiency</div>
        <div>Chinchilla-optimal training requires 20 tokens/parameter. GPT-3 175B needed 300B tokens; LLaMA-3 405B used 15T tokens. Humans acquire language from ~100M words (≈ 1B tokens) over childhood. Current pre-training is 10,000× less sample-efficient than biological learning.</div>
      </div>
      <div style="background:#1a1a2e; border:1px solid #dcdcaa; border-radius:6px; padding:12px;">
        <div style="color:#dcdcaa; font-weight:700; margin-bottom:6px;">Test-Time Compute Scaling</div>
        <div>DeepSeek-R1 and OpenAI-o3 show that spending more inference compute (extended chain-of-thought, self-consistency) improves performance on hard tasks. A second scaling law axis beyond training compute. Optimal allocation of training vs inference budget is an open research question.</div>
      </div>
    </div>
  </div>

  <p><strong>Model comparison (mid-2025 snapshot):</strong></p>

  <div class="ar-loop" style="padding:10px; overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-size:0.78rem;">
      <thead>
        <tr style="border-bottom:1px solid #444; color:#888;">
          <th style="text-align:left; padding:6px 10px;">Model</th>
          <th style="text-align:right; padding:6px 10px;">Params</th>
          <th style="text-align:right; padding:6px 10px;">Context</th>
          <th style="text-align:right; padding:6px 10px;">Training tokens</th>
          <th style="text-align:left; padding:6px 10px;">Access</th>
          <th style="text-align:left; padding:6px 10px;">Notable</th>
        </tr>
      </thead>
      <tbody>
""")

table_rows = [
    ("GPT-4 Turbo", "~1T MoE*", "128K", "~13T*", "API", "Multimodal, function calling"),
    ("Claude 3.5 Sonnet", "~70B*", "200K", "~8T*", "API", "Best coding/agentic (2024)"),
    ("Gemini 1.5 Pro", "MoE*", "1M", "~7T*", "API", "Video, audio native"),
    ("Llama 3.1 405B", "405B dense", "128K", "15T", "Open weight", "GPT-4 parity, free"),
    ("Llama 3.1 8B", "8B dense", "128K", "15T", "Open weight", "Fits on consumer GPU"),
    ("Mixtral 8×7B", "46.7B total", "32K", "~8T*", "Open weight", "12.9B active, fast"),
    ("DeepSeek-R1", "671B MoE", "128K", "14.8T", "Open weight", "Reasoning, RL-only training"),
    ("Mistral 7B v0.3", "7.3B dense", "32K", "~8T*", "Open weight", "Sliding window attention"),
]
for i, (name, params, ctx, tokens, access, note) in enumerate(table_rows):
    bg = "#0d0d0d" if i % 2 == 0 else "#111"
    A(f"""        <tr style="background:{bg}; border-bottom:1px solid #222;">
          <td style="padding:6px 10px; font-weight:600; color:#dcdcaa;">{name}</td>
          <td style="padding:6px 10px; text-align:right;">{params}</td>
          <td style="padding:6px 10px; text-align:right;">{ctx}</td>
          <td style="padding:6px 10px; text-align:right;">{tokens}</td>
          <td style="padding:6px 10px; color:#4ec9b0;">{access}</td>
          <td style="padding:6px 10px; color:#888; font-size:0.72rem;">{note}</td>
        </tr>
""")

A("""      </tbody>
    </table>
    <div style="margin-top:6px; color:#555; font-size:0.68rem;">* Unofficial estimates. MoE = Mixture of Experts. All figures as of mid-2025.</div>
  </div>

  <div class="cell" id="cell-17">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Compute scaling: training cost vs inference cost tradeoff</span>
      <button class="run-btn" onclick="runCell('cell-17')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

# --- Training compute budget breakdown ---
# Rule of thumb: C ≈ 6 * N * D FLOPs (Kaplan 2020)
# Factor of 6 = 2 (forward) + 4 (backward, ~2x forward)
def training_flops(N_params, D_tokens):
    return 6 * N_params * D_tokens

# --- Inference compute per token ---
# ~2 * N FLOPs per token (matrix-vector products dominate)
def inference_flops_per_token(N_params):
    return 2 * N_params

print("Training compute (in FLOPs) for selected models:")
models = [
    ("GPT-3 175B", 175e9, 300e9),
    ("Llama 3.1 8B", 8e9, 15e12),
    ("Llama 3.1 70B", 70e9, 15e12),
    ("Llama 3.1 405B", 405e9, 15e12),
    ("DeepSeek-R1 671B", 671e9, 14.8e12),
]
for name, N, D in models:
    C_train = training_flops(N, D)
    C_inf = inference_flops_per_token(N)
    tokens_to_equal_training = C_train / C_inf
    print(f"\n  {name}:")
    print(f"    Training FLOPs:     {C_train:.2e}")
    print(f"    Inference FLOPs/tok:{C_inf:.2e}")
    print(f"    Equiv. inference:   {tokens_to_equal_training:.2e} tokens to match training cost")
    # At 100 tokens/second, how many years to match training cost?
    years = tokens_to_equal_training / (100 * 60 * 60 * 24 * 365)
    print(f"    At 100 tok/s:       {years:.1f} years of inference = 1 training run")

print()
print("Test-time compute scaling (o1/R1 style chain-of-thought):")
print("More thinking tokens at inference -> better on hard tasks")
base_acc = 0.4  # accuracy without extended thinking
for think_tokens in [0, 100, 500, 2000, 8000, 32000]:
    # Diminishing returns sigmoid model
    if think_tokens == 0:
        acc = base_acc
    else:
        x = math.log10(think_tokens)
        acc = base_acc + (0.58) / (1 + math.exp(-1.5 * (x - 2.0)))
    cost_multiplier = 1 + think_tokens / 100
    bar = "#" * int(acc * 40)
    print(f"  thinking_tokens={think_tokens:>6}  acc={acc:.3f}  cost={cost_multiplier:>6.1f}x  {bar}")

# Try changing base_acc to 0.8 to see diminishing returns in high-accuracy regime</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2501.12948" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">DeepSeek-AI, 2025 &#8212; DeepSeek-R1: Incentivizing Reasoning Capability via RL</span> &#8212; Pure RL training (GRPO, no SFT cold-start for final model) achieves GPT-4o parity on AIME 2024. 671B MoE model open-weighted.</span></a>
      <a href="https://arxiv.org/abs/2407.21783" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Meta AI, 2024 &#8212; The Llama 3 Herd of Models</span> &#8212; 8B, 70B, 405B. 15T tokens. GQA, RoPE base=500,000, 128K context. 405B matches GPT-4 on most benchmarks. Full training details published.</span></a>
      <a href="https://arxiv.org/abs/2403.05530" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Google DeepMind, 2024 &#8212; Gemini 1.5: Unlocking Multimodal Understanding</span> &#8212; 1M-token context via ring attention. Needle-in-a-haystack near-perfect at 1M tokens. Retrieves hidden audio in 22 hours of video.</span></a>
      <a href="https://arxiv.org/abs/2307.03172" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Liu et al., 2023 &#8212; Lost in the Middle: How Language Models Use Long Contexts</span> &#8212; Multi-document QA: accuracy peaks at context start and end, drops sharply in middle. Consistent across GPT-3.5, GPT-4, Claude. An attention recency/primacy bias.</span></a>
      <a href="https://karpathy.ai/zero-to-hero.html" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Andrej Karpathy &#8212; Zero to Hero Neural Networks (2024)</span> &#8212; Video series building GPT from scratch in PyTorch. micrograd, makemore, nanoGPT, llama2.c. Essential companion to this chapter for practitioners.</span></a>
      <a href="https://simonwillison.net/2024/Apr/17/ai-for-data-journalism/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Simon Willison &#8212; Tracking the AI landscape (ongoing blog)</span> &#8212; The most consistently accurate and current survey of model releases, benchmark results, and deployment considerations. Updated almost daily.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

""")

# Closing JS block
A("""
</main>
</div><!-- #layout -->

<!-- ============================================================ JS == -->
<script>
// ── Vocabulary size calculator (§1.2 slider) ─────────────────────────────
function updateVocabCalc() {
  var v = parseInt(document.getElementById('vocabSlider').value, 10);
  document.getElementById('vocabVal').textContent = v.toLocaleString();
  var embDim = 4096;
  var embMB  = (v * embDim * 2 / 1e6).toFixed(1);
  document.getElementById('embSize').textContent = embMB + ' MB';
  var logitMB = (v * embDim * 2 / 1e6).toFixed(1);
  document.getElementById('logitSize').textContent = logitMB + ' MB';
  var tokPerSec = Math.round(50000 / (v / 32000));
  document.getElementById('tokPerSec').textContent = tokPerSec.toLocaleString() + ' tok/s (approx)';
}

// ── PE heatmap (§1.7) ────────────────────────────────────────────────────
function drawPEHeatmap() {
  var canvas = document.getElementById('pe-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var maxPos = 64, maxDim = 64;
  var cw = W / maxDim, ch = H / maxPos;
  for (var pos = 0; pos < maxPos; pos++) {
    for (var dim = 0; dim < maxDim; dim++) {
      var freq = 1 / Math.pow(10000, (2 * Math.floor(dim / 2)) / maxDim);
      var val;
      if (dim % 2 === 0) {
        val = Math.sin(pos * freq);
      } else {
        val = Math.cos(pos * freq);
      }
      var norm = (val + 1) / 2; // 0..1
      var r = Math.round(norm * 80);
      var g = Math.round(norm * 140 + (1-norm) * 40);
      var b = Math.round(norm * 200 + (1-norm) * 100);
      ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      ctx.fillRect(dim * cw, pos * ch, cw + 0.5, ch + 0.5);
    }
  }
  // Axes
  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.fillText('dim →', 4, 12);
  ctx.save();
  ctx.translate(12, H/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('position ↓', 0, 0);
  ctx.restore();
}

// ── Pyodide in-browser Python runtime ────────────────────────────────────
var pyodide = null;
var pyodideLoading = false;

async function initPyodide() {
  if (pyodide) return pyodide;
  if (pyodideLoading) {
    // Wait until loaded
    while (!pyodide) await new Promise(r => setTimeout(r, 100));
    return pyodide;
  }
  pyodideLoading = true;
  // Show global loading notice
  var notice = document.createElement('div');
  notice.id = 'pyodide-notice';
  notice.style.cssText = 'position:fixed;bottom:16px;right:16px;background:#252526;border:1px solid #4ec9b0;padding:10px 16px;border-radius:6px;color:#4ec9b0;font-family:monospace;font-size:0.8rem;z-index:9999;';
  notice.textContent = 'Loading Python runtime (Pyodide ~8 MB)…';
  document.body.appendChild(notice);
  try {
    pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/' });
    notice.textContent = 'Python ready ✓';
    setTimeout(() => notice.remove(), 1800);
  } catch(e) {
    notice.style.borderColor = '#f44';
    notice.style.color = '#f44';
    notice.textContent = 'Pyodide load failed: ' + e.message;
    pyodideLoading = false;
    throw e;
  }
  pyodideLoading = false;
  return pyodide;
}

async function runCell(cellId) {
  var cell = document.getElementById(cellId);
  if (!cell) return;
  var codeEl = cell.querySelector('.cell-code pre');
  var outEl  = cell.querySelector('.cell-output');
  var btn    = cell.querySelector('.run-btn');
  if (!codeEl || !outEl) return;

  var code = codeEl.textContent;
  outEl.textContent = 'Loading Python runtime…';
  outEl.style.color = '#888';
  btn.disabled = true;
  btn.textContent = 'Running…';

  try {
    var py = await initPyodide();
    // Capture stdout
    py.runPython('import sys, io\\nclass _Capture:\\n  def __init__(self): self.buf=""\\n  def write(self,s): self.buf+=s\\n  def flush(self): pass\\n_cap=_Capture()\\nsys.stdout=_cap');
    try {
      py.runPython(code);
    } catch(runErr) {
      outEl.textContent = '⚠ ' + runErr.message;
      outEl.style.color = '#f44';
      return;
    }
    var output = py.runPython('sys.stdout=sys.__stdout__;_cap.buf');
    outEl.textContent = output || '(no output)';
    outEl.style.color = '#d4d4d4';
  } catch(e) {
    outEl.textContent = '⚠ Runtime error: ' + e.message;
    outEl.style.color = '#f44';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run';
  }
}

// ── Navigation active-section tracking ───────────────────────────────────
(function() {
  var sections = document.querySelectorAll('section.section[id]');
  var navLinks = document.querySelectorAll('.toc a');

  function getActiveSectionId() {
    var scrollY = window.scrollY + 120;
    var active = null;
    sections.forEach(function(sec) {
      if (sec.offsetTop <= scrollY) active = sec.id;
    });
    return active;
  }

  function updateNav() {
    var id = getActiveSectionId();
    navLinks.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + id);
    });
  }

  window.addEventListener('scroll', updateNav, { passive: true });
  updateNav();
})();

// ── Smooth scroll for TOC links ───────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(function(a) {
  a.addEventListener('click', function(e) {
    var target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── Initialise on page load ───────────────────────────────────────────────
window.addEventListener('load', function() {
  drawPEHeatmap();
  // Preload Pyodide in background after 3s idle
  setTimeout(function() {
    if (!pyodide) initPyodide().catch(function(){});
  }, 3000);
});
</script>

</body>
</html>
""")

content = "".join(lines)
with open(OUT, 'a', encoding='utf-8') as f:
    f.write(content)
print(f"S16-S17 + closing JS appended: {len(content)} chars")
