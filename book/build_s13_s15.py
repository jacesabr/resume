#!/usr/bin/env python3
# Appends sections 1.13, 1.14, 1.15
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

lines = []
A = lines.append

A("""
<!-- ============================================================ §1.13 == -->
<section class="section" id="s13">
  <div class="section-label">1.13</div>
  <h2>Alignment: RLHF, DPO &amp; LoRA</h2>

  <p>Pre-training produces a model that predicts text; it does not produce a model that helpfully follows instructions. Bridging that gap — <em>alignment</em> — is the central engineering challenge of deploying LLMs. Three techniques dominate: Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and the more recent Direct Preference Optimisation (DPO). Low-Rank Adaptation (LoRA) makes all three computationally feasible without retraining billions of parameters.</p>

  <p><strong>Stage 1 — SFT.</strong> A curated set of (prompt, ideal response) pairs is collected — InstructGPT used 13,000 prompts written by 40 contractors (Ouyang et al., 2022, arXiv:2203.02155). The pre-trained model is fine-tuned with standard cross-entropy loss on these examples. SFT teaches the response format but cannot capture nuanced human preferences about tone, safety, or helpfulness.</p>

  <p><strong>Stage 2 — Reward Model Training.</strong> Contractors rank model outputs for the same prompt. A separate model R<sub>φ</sub>(x, y) is trained to predict the human preference score. For a preferred response y<sub>w</sub> vs rejected response y<sub>l</sub>, the reward model loss is:</p>

  <div class="callout">
    <strong>Reward model loss:</strong>
    &#x2112;(φ) = &minus;E<sub>(x,y<sub>w</sub>,y<sub>l</sub>)~D</sub>[ log σ(R<sub>φ</sub>(x,y<sub>w</sub>) &minus; R<sub>φ</sub>(x,y<sub>l</sub>)) ]
    <br><br>This is a Bradley-Terry preference model: the probability that humans prefer y<sub>w</sub> equals σ(R<sub>w</sub> &minus; R<sub>l</sub>). Training pushes the preferred response's score above the rejected one's.
  </div>

  <p><strong>Stage 3 — PPO.</strong> The SFT model is fine-tuned with proximal policy optimisation to maximise the reward model's score, subject to a KL-divergence penalty that prevents the model drifting too far from the SFT base. The PPO update is:</p>

  <div class="callout">
    <strong>RLHF objective:</strong>
    max<sub>π</sub> E<sub>x~D, y~π</sub>[ R<sub>φ</sub>(x,y) ] &minus; β · KL[π(y|x) ∥ π<sub>SFT</sub>(y|x)]
    <br><br>&beta; is typically 0.01–0.1. Without the KL penalty the model collapses to reward-hacking — e.g., producing very long outputs that the reward model was not trained to penalise.
  </div>

  <p><strong>DPO — a simpler alternative.</strong> Rafailov et al. (2023, arXiv:2305.18290) showed that the RLHF objective has a closed-form optimal policy. Rather than training a separate reward model + running PPO, DPO directly fine-tunes the LLM on preference pairs:</p>

  <div class="callout">
    <strong>DPO loss:</strong>
    &#x2112;<sub>DPO</sub>(θ) = &minus;E<sub>(x,y<sub>w</sub>,y<sub>l</sub>)</sub>[ log σ( β log π<sub>θ</sub>(y<sub>w</sub>|x)/π<sub>ref</sub>(y<sub>w</sub>|x) &minus; β log π<sub>θ</sub>(y<sub>l</sub>|x)/π<sub>ref</sub>(y<sub>l</sub>|x) ) ]
    <br><br>Interpretation: DPO increases the probability of preferred responses relative to the reference model, while decreasing rejected responses — all in a single supervised loss. No PPO, no separate reward model, 2&times; less VRAM at training time.
  </div>

  <p><strong>LoRA — efficient fine-tuning.</strong> Full fine-tuning of a 70B model requires ~280 GB of GPU VRAM for fp16 weights alone. Hu et al. (2021, arXiv:2106.09685) introduced Low-Rank Adaptation: instead of updating W directly, add a low-rank decomposition Δ W = BA where B ∈ ℝ<sup>d×r</sup>, A ∈ ℝ<sup>r×k</sup>, and r ≪ min(d,k). For a 4096×4096 attention projection with r=16, LoRA uses 4096×16 + 16×4096 = 131,072 parameters instead of 16,777,216 — a 128× reduction. With r=64 on a 7B model, LoRA reduces trainable parameters from 7B to ~4M: a 1,750× reduction. Only the A,B matrices are updated; W is frozen. At inference, Δ W = BA is merged into W — zero latency overhead.</p>

  <div class="ar-loop" style="grid-template-columns:1fr 1fr 1fr; gap:12px; padding:16px;">
    <div style="text-align:center; padding:12px; background:#1a1a2e; border:1px solid #444; border-radius:6px;">
      <div style="color:#888; font-size:0.7rem; margin-bottom:6px;">PRE-TRAINING</div>
      <div style="font-size:0.85rem;">Next-token prediction<br>300B–15T tokens<br>Cross-entropy loss</div>
    </div>
    <div style="text-align:center; padding:12px; background:#1a1a2e; border:1px solid #4ec9b0; border-radius:6px;">
      <div style="color:#4ec9b0; font-size:0.7rem; margin-bottom:6px;">SFT</div>
      <div style="font-size:0.85rem;">13K curated pairs<br>Format + style<br>Standard CE loss</div>
    </div>
    <div style="text-align:center; padding:12px; background:#1a1a2e; border:1px solid #c586c0; border-radius:6px;">
      <div style="color:#c586c0; font-size:0.7rem; margin-bottom:6px;">RLHF / DPO</div>
      <div style="font-size:0.85rem;">Human preference pairs<br>Reward shaping<br>KL-constrained update</div>
    </div>
  </div>

  <div class="code-block">
    <span class="code-label">Python</span>
    <pre><span class="kw">import</span> math

<span class="lm"># DPO loss — single preferred/rejected pair</span>
<span class="kw">def</span> <span class="fn">dpo_loss</span>(log_pi_w, log_ref_w, log_pi_l, log_ref_l, beta=<span class="nu">0.1</span>):
    <span class="lm"># ↑ beta controls how far policy can deviate from reference</span>
    ratio_w = log_pi_w - log_ref_w   <span class="lm"># ↑ log-ratio for preferred response</span>
    ratio_l = log_pi_l - log_ref_l   <span class="lm"># ↑ log-ratio for rejected response</span>
    logit   = beta * (ratio_w - ratio_l)   <span class="lm"># ↑ margin between preferred and rejected</span>
    loss    = -math.log(1 / (1 + math.exp(-logit)))   <span class="lm"># ↑ -log σ(logit)</span>
    <span class="kw">return</span> loss

<span class="lm"># LoRA parameter budget analysis</span>
<span class="kw">def</span> <span class="fn">lora_params</span>(d, k, rank):
    <span class="lm"># ↑ d=input dim, k=output dim, rank=LoRA rank (e.g. 4, 8, 16, 64)</span>
    full_params  = d * k
    lora_trainable = d * rank + rank * k   <span class="lm"># ↑ A matrix + B matrix</span>
    reduction = full_params / lora_trainable
    <span class="kw">return</span> full_params, lora_trainable, reduction

<span class="kw">for</span> d, k, label <span class="kw">in</span> [(4096, 4096, <span class="st">"LLaMA-3 8B Q,K,V proj"</span>),
                       (8192, 8192, <span class="st">"LLaMA-3 70B Q,K,V proj"</span>),
                       (14336, 4096, <span class="st">"LLaMA-3 8B down_proj"</span>)]:
    <span class="kw">for</span> r <span class="kw">in</span> [8, 16, 64]:
        full, trainable, reduction = lora_params(d, k, r)
        print(f<span class="st">"{label} r={r:2d}: {trainable:,} trainable / {full:,} full ({reduction:.0f}× smaller)"</span>)</pre>
  </div>

  <div class="cell" id="cell-13">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; DPO loss landscape + LoRA parameter budgets</span>
      <button class="run-btn" onclick="runCell('cell-13')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dpo_loss(log_pi_w, log_ref_w, log_pi_l, log_ref_l, beta=0.1):
    ratio_w = log_pi_w - log_ref_w
    ratio_l = log_pi_l - log_ref_l
    logit = beta * (ratio_w - ratio_l)
    return -math.log(sigmoid(logit))

# Show how DPO loss changes as model learns to prefer w over l
print("DPO loss as margin (ratio_w - ratio_l) grows:")
print(f"  {'margin':>8}  {'DPO loss':>10}  {'sigma':>8}")
for margin in [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]:
    # ratio_w - ratio_l = margin (ref cancels out for demonstration)
    loss = -math.log(sigmoid(0.1 * margin))
    sig = sigmoid(0.1 * margin)
    bar = "#" * int(sig * 30)
    print(f"  {margin:>8.1f}  {loss:>10.4f}  {sig:>8.4f}  {bar}")

print()
print("LoRA parameter savings:")
configs = [
    ("LLaMA-3 8B Q proj", 4096, 4096),
    ("LLaMA-3 70B Q proj", 8192, 8192),
    ("Mistral 7B Q proj", 4096, 4096),
]
for name, d, k in configs:
    full = d * k
    for r in [8, 16, 64]:
        train = d * r + r * k
        ratio = full / train
        print(f"  {name} r={r:2d}: {train:>8,} trainable ({ratio:5.0f}x reduction from {full:,})")
    print()

# Try changing beta (0.01-1.0) and re-running to see how loss sensitivity changes</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2203.02155" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Ouyang et al., 2022 &#8212; InstructGPT</span> &#8212; RLHF applied to GPT-3. 40 contractors, 13K SFT prompts, 33K reward model comparisons. 1.3B InstructGPT rated better than 175B GPT-3 by labellers.</span></a>
      <a href="https://arxiv.org/abs/2305.18290" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Rafailov et al., 2023 &#8212; Direct Preference Optimization (DPO)</span> &#8212; Shows the closed-form optimal policy implicit in RLHF. Eliminates reward model + PPO with a single supervised loss. Llama-2 fine-tuning in &lt;1 day on consumer hardware.</span></a>
      <a href="https://arxiv.org/abs/2106.09685" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Hu et al., 2021 &#8212; LoRA: Low-Rank Adaptation of Large Language Models</span> &#8212; Full fine-tuning baseline on GPT-3: 175B trainable params. LoRA r=4: 4.7M trainable. Quality matches full fine-tuning on GLUE, E2E, WebNLG benchmarks.</span></a>
      <a href="https://arxiv.org/abs/2309.00267" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Cui et al., 2023 &#8212; UltraFeedback: Boosting Language Models with Scaled AI Feedback</span> &#8212; 256K AI-generated preference pairs from GPT-4 scoring. Shows AI feedback can match human labelling at 1/100th the cost. Used to train Zephyr-7B-beta.</span></a>
      <a href="https://huggingface.co/blog/rlhf" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">HuggingFace Blog &#8212; Illustrating RLHF (2022)</span> &#8212; Clear visual walkthrough of all three RLHF stages with code examples using the TRL library.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.14 == -->
<section class="section" id="s14">
  <div class="section-label">1.14</div>
  <h2>Scale &amp; Emergent Abilities</h2>

  <p>The Kaplan scaling laws (Kaplan et al., 2020, arXiv:2001.08361) were the first systematic study of how loss decreases with scale. They found:</p>

  <div class="callout">
    <strong>Kaplan power laws:</strong>
    L(N) ≈ (N<sub>c</sub>/N)<sup>α<sub>N</sub></sup> &nbsp;&nbsp; L(D) ≈ (D<sub>c</sub>/D)<sup>α<sub>D</sub></sup> &nbsp;&nbsp; L(C) ≈ (C<sub>c</sub>/C)<sup>α<sub>C</sub></sup>
    <br><br>Where N=parameters, D=tokens, C=compute (FLOPs). Exponents: α<sub>N</sub>≈0.076, α<sub>D</sub>≈0.095, α<sub>C</sub>≈0.050. The laws hold across 7 orders of magnitude — from 768-parameter models to 1.5B parameters. Chinchilla (2022) later revised the compute-optimal frontier.
  </div>

  <p><strong>Emergent abilities</strong> are capabilities that appear abruptly at a threshold scale and are near-zero below it. Wei et al. (2022, arXiv:2206.07682) catalogued ~137 tasks that show emergence across PaLM (540B), GPT-3 (175B), and other large models. Examples: 3-digit addition emerges at ~100B parameters; chain-of-thought reasoning emerges at ~60B; word-in-context disambiguation emerges at ~130B. The abruptness is partly a measurement artifact: on accuracy metrics, capability transitions look sudden; on continuous loss metrics they look smoother. But the phenomenon is real — some capabilities genuinely require sufficient representational capacity before they can be expressed.</p>

  <p>Srivastava et al.'s BIG-Bench (2022, arXiv:2206.04615) benchmark of 204 tasks specifically targets abilities believed to be beyond current models. Of 150 tasks evaluated, 65 showed emergence patterns — flat near random performance at small scale, sudden jump at large scale. The jump often occurs at 10<sup>23</sup>–10<sup>24</sup> training FLOPs, corresponding roughly to GPT-3 scale.</p>

  <div class="ar-loop" style="padding:16px;">
    <div style="margin-bottom:10px; color:#888; font-size:0.75rem;">EMERGENT ABILITY THRESHOLD EXAMPLES (approximate parameter counts)</div>
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px; font-size:0.78rem;">
      <div style="background:#1a1a2e; border:1px solid #333; border-radius:4px; padding:10px;">
        <div style="color:#4ec9b0; margin-bottom:4px;">~1B params</div>
        <div>Basic arithmetic (2-digit)</div>
        <div>Simple code completion</div>
        <div>Translation (high-resource)</div>
      </div>
      <div style="background:#1a1a2e; border:1px solid #569cd6; border-radius:4px; padding:10px;">
        <div style="color:#569cd6; margin-bottom:4px;">~10B params</div>
        <div>3-digit arithmetic</div>
        <div>Analogical reasoning</div>
        <div>Multi-step word problems</div>
      </div>
      <div style="background:#1a1a2e; border:1px solid #c586c0; border-radius:4px; padding:10px;">
        <div style="color:#c586c0; margin-bottom:4px;">~100B+ params</div>
        <div>Chain-of-thought reasoning</div>
        <div>Word-in-context disambiguation</div>
        <div>Novel concept composition</div>
      </div>
    </div>
  </div>

  <p>An important nuance: scaling improves average performance but <em>reduces calibration</em> on some tasks. Larger models are more confident in wrong answers (Anthropic, 2022). This is why RLHF is needed even after capability scaling — raw performance and alignment are not the same axis.</p>

  <div class="code-block">
    <span class="code-label">Python</span>
    <pre><span class="kw">import</span> math

<span class="lm"># Kaplan power-law loss prediction</span>
<span class="kw">def</span> <span class="fn">kaplan_loss</span>(N, D, C_ref=<span class="nu">1e21</span>):
    <span class="lm"># ↑ N = parameters, D = training tokens</span>
    N_c, alpha_N = <span class="nu">8.8e13</span>, <span class="nu">0.076</span>   <span class="lm"># ↑ fitted constants from Kaplan et al.</span>
    D_c, alpha_D = <span class="nu">5.4e13</span>, <span class="nu">0.095</span>
    L_N = (N_c / N) ** alpha_N       <span class="lm"># ↑ loss from model size alone</span>
    L_D = (D_c / D) ** alpha_D       <span class="lm"># ↑ loss from data size alone</span>
    L   = L_N + L_D                  <span class="lm"># ↑ approximate total loss (simplified)</span>
    <span class="kw">return</span> L_N, L_D, L

<span class="lm"># Compare several well-known models</span>
models = [
    (<span class="st">"GPT-2 small"</span>,  <span class="nu">117e6</span>,  <span class="nu">10e9</span>),
    (<span class="st">"GPT-3"</span>,        <span class="nu">175e9</span>, <span class="nu">300e9</span>),
    (<span class="st">"LLaMA-3 8B"</span>,    <span class="nu">8e9</span>,  <span class="nu">15e12</span>),
    (<span class="st">"LLaMA-3 70B"</span>,  <span class="nu">70e9</span>,  <span class="nu">15e12</span>),
    (<span class="st">"PaLM-540B"</span>,   <span class="nu">540e9</span>, <span class="nu">780e9</span>),
]
print(f"{'Model':<18} {'Params':>10} {'Tokens':>10}  {'L_N':>6}  {'L_D':>6}  {'L_total':>8}")
<span class="kw">for</span> name, N, D <span class="kw">in</span> models:
    L_N, L_D, L = kaplan_loss(N, D)
    print(f<span class="st">"{name:<18} {N:>10.2e} {D:>10.2e}  {L_N:>6.3f}  {L_D:>6.3f}  {L:>8.3f}"</span>)</pre>
  </div>

  <div class="cell" id="cell-14">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Kaplan scaling laws + emergence simulation</span>
      <button class="run-btn" onclick="runCell('cell-14')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

# Kaplan et al. 2020 power-law constants (Table 1)
N_c, alpha_N = 8.8e13, 0.076
D_c, alpha_D = 5.4e13, 0.095

def loss_from_params(N):
    # Predict validation loss from param count (infinite data limit)
    return (N_c / N) ** alpha_N

def loss_from_data(D):
    return (D_c / D) ** alpha_D

# Show loss reduction across orders of magnitude
print("Loss vs. model size (infinite data):")
for exp in range(6, 12):
    N = 10 ** exp
    L = loss_from_params(N)
    pct = (L / loss_from_params(1e6) - 1) * 100
    bar = "#" * int((3.5 - L) * 30) if L < 3.5 else ""
    print(f"  N=10^{exp} ({N:.0e} params)  L={L:.3f}  {bar}")

print()
print("Chinchilla-optimal: ~20 tokens per parameter")
print("(Hoffmann et al. 2022 revised Kaplan; Chinchilla 70B > Gopher 280B)")
print()

# Simulate emergence: capability appears abruptly at threshold
# (toy model: capability = 0 below threshold, jumps at threshold)
print("Emergence simulation (3-digit arithmetic task):")
print("Parameter count  |  Accuracy  |  Bar")
threshold = 5e10  # ~50B parameters where 3-digit arithmetic emerges
import random
random.seed(7)
for exp_tenths in range(80, 120, 4):
    N = 10 ** (exp_tenths / 10)
    if N < threshold:
        acc = random.uniform(0.01, 0.08)  # near-random below threshold
    else:
        # sigmoid rise above threshold
        x = math.log10(N / threshold)
        acc = 0.15 + 0.78 / (1 + math.exp(-5 * x))
    bar = "#" * int(acc * 40)
    print(f"  {N:.1e}  |  {acc:.3f}     |  {bar}")

# Try changing the threshold to see how emergence looks at different scales</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2001.08361" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Kaplan et al., 2020 &#8212; Scaling Laws for Neural Language Models</span> &#8212; Power-law exponents α<sub>N</sub>=0.076, α<sub>D</sub>=0.095 measured across 6 orders of magnitude. The paper that justified training ever-larger models.</span></a>
      <a href="https://arxiv.org/abs/2206.07682" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Wei et al., 2022 &#8212; Emergent Abilities of Large Language Models</span> &#8212; Catalogs 137 tasks showing emergence. Defines emergence as "not present in smaller models and present in larger models." Critical reading for anyone reasoning about scale.</span></a>
      <a href="https://arxiv.org/abs/2206.04615" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Srivastava et al., 2022 &#8212; BIG-Bench: Beyond the Imitation Game</span> &#8212; 204 tasks, 450 researchers, 132 institutions. Tasks deliberately chosen to be hard for current models. BIG-Bench-Hard (23 tasks) still challenges GPT-4.</span></a>
      <a href="https://arxiv.org/abs/2304.15004" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Schaeffer et al., 2023 &#8212; Are Emergent Abilities of LLMs a Mirage?</span> &#8212; Argues that emergence is partly an artifact of nonlinear metrics (accuracy). On perplexity, capability increases smoothly. A necessary counterpoint to the emergence narrative.</span></a>
      <a href="https://ai.googleblog.com/2022/11/characterizing-emergent-phenomena-in.html" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Google AI Blog &#8212; Characterizing Emergent Phenomena in LLMs (2022)</span> &#8212; Visualisations of which BIG-Bench tasks show sharp vs. gradual improvement. Includes interactive plots of PaLM's performance across scales.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.15 == -->
<section class="section" id="s15">
  <div class="section-label">1.15</div>
  <h2>Mixture of Experts (MoE)</h2>

  <p>A standard ("dense") transformer activates <em>all</em> parameters for every token. A Mixture-of-Experts model sparsely routes each token to a small subset of specialised FFN networks ("experts"), keeping compute constant while multiplying total capacity. Shazeer et al. (2017, arXiv:1701.06538) introduced MoE to NLP; Fedus et al. (2022, arXiv:2101.03961) scaled it to the Switch Transformer at 1.6 trillion parameters — the largest language model published to date — using only the compute of a 7B dense model.</p>

  <p><strong>Architecture.</strong> In an MoE layer, the standard FFN sublayer is replaced by E expert networks (typically E=8 or E=64) plus a learned router. The router is a small linear layer that produces a probability distribution over experts. For each token, the top-K experts (typically K=1 or K=2) are selected and their outputs are linearly combined:</p>

  <div class="callout">
    <strong>MoE forward pass (Top-1 routing):</strong>
    <br>h = Router(x) = Softmax(x W<sub>r</sub>) &nbsp;&nbsp;&nbsp; [E-dimensional]
    <br>i* = argmax(h) &nbsp;&nbsp;&nbsp; [selected expert index]
    <br>output = h[i*] &middot; FFN<sub>i*</sub>(x) &nbsp;&nbsp;&nbsp; [weighted expert output]
    <br><br>Mistral's Mixtral-8x7B (Jiang et al., 2024, arXiv:2401.04088) uses K=2 of E=8 experts per token. Total parameters: 46.7B. Active parameters per token: only 12.9B — matching a ~13B dense model's compute at 3.6× the capacity.
  </div>

  <p><strong>Load balancing.</strong> Without regularisation, the router collapses: one expert receives all tokens (routing collapse). The auxiliary load-balancing loss encourages uniform expert utilisation:</p>

  <div class="callout">
    <strong>Auxiliary loss (Switch Transformer):</strong>
    &#x2112;<sub>aux</sub> = α · E · &sum;<sub>i</sub> f<sub>i</sub> · P<sub>i</sub>
    <br><br>f<sub>i</sub> = fraction of tokens routed to expert i (discrete, non-differentiable)
    <br>P<sub>i</sub> = average router probability for expert i (differentiable)
    <br>The product f<sub>i</sub>·P<sub>i</sub> is differentiable w.r.t. router weights and minimized when all experts receive equal load.
  </div>

  <p><strong>Expert capacity.</strong> In distributed training, each expert is placed on a different device. A "capacity factor" C limits how many tokens can be processed per expert per batch — overflow tokens are "dropped" (passed through as residual unchanged). C=1.0 means no slack; C=1.25 is typical. Dropped tokens are a training inefficiency but acceptable at scale.</p>

  <p><strong>Real-world MoE models in production:</strong> Mixtral-8x7B (Mistral, 2024) — open-weight, outperforms LLaMA-2 70B at ~13B active params. GPT-4 — rumored 8 experts ×220B (1.76T total), though not officially confirmed. Gemini 1.5 Pro (Google, 2024, arXiv:2403.05530) — MoE with reported 1M-token context.</p>

  <div class="ar-loop" style="padding:16px;">
    <div style="margin-bottom:8px; color:#888; font-size:0.75rem;">MIXTRAL-8x7B: 8 EXPERTS, TOP-2 ROUTING</div>
    <div style="display:flex; gap:6px; justify-content:center; flex-wrap:wrap;">
""")

# Add expert boxes
for i in range(8):
    if i < 2:
        A(f"""      <div style="width:70px; height:60px; background:#1a1a2e; border:2px solid #4ec9b0; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; text-align:center; color:#4ec9b0;">Expert<br>{i+1}<br><span style="color:#fff; font-size:0.65rem;">ACTIVE</span></div>
""")
    else:
        A(f"""      <div style="width:70px; height:60px; background:#0d0d0d; border:1px solid #333; border-radius:4px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; text-align:center; color:#555;">Expert<br>{i+1}</div>
""")

A("""    </div>
    <div style="margin-top:8px; text-align:center; font-size:0.75rem; color:#888;">Each token activates 2 of 8 experts → 12.9B active params of 46.7B total (28% density)</div>
  </div>

  <div class="code-block">
    <span class="code-label">Python</span>
    <pre><span class="kw">import</span> math, random

<span class="lm"># Simplified MoE routing: softmax router, Top-K selection</span>
<span class="kw">def</span> <span class="fn">softmax</span>(x):
    m = <span class="bi">max</span>(x)
    e = [math.exp(v - m) <span class="kw">for</span> v <span class="kw">in</span> x]
    s = <span class="bi">sum</span>(e)
    <span class="kw">return</span> [v/s <span class="kw">for</span> v <span class="kw">in</span> e]

<span class="kw">def</span> <span class="fn">top_k_routing</span>(token_vec, router_weights, k=<span class="nu">2</span>):
    <span class="lm"># ↑ router_weights: (E, d) matrix; E experts, d-dim tokens</span>
    E = <span class="bi">len</span>(router_weights)
    logits = [<span class="bi">sum</span>(token_vec[j] * router_weights[i][j]
               <span class="kw">for</span> j <span class="kw">in</span> <span class="bi">range</span>(<span class="bi">len</span>(token_vec)))
              <span class="kw">for</span> i <span class="kw">in</span> <span class="bi">range</span>(E)]   <span class="lm"># ↑ dot product of token with each expert gate</span>
    probs = softmax(logits)
    ranked = <span class="bi">sorted</span>(<span class="bi">enumerate</span>(probs), key=<span class="kw">lambda</span> x: -x[<span class="nu">1</span>])
    selected = ranked[:k]    <span class="lm"># ↑ top-K experts selected</span>
    total = <span class="bi">sum</span>(p <span class="kw">for</span> _, p <span class="kw">in</span> selected)
    weights = [(i, p/total) <span class="kw">for</span> i, p <span class="kw">in</span> selected]   <span class="lm"># ↑ renormalize weights</span>
    <span class="kw">return</span> weights</pre>
  </div>

  <div class="cell" id="cell-15">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; MoE routing simulation with load balancing</span>
      <button class="run-btn" onclick="runCell('cell-15')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random

random.seed(99)

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v/s for v in e]

def top_k_route(logits, k=2):
    probs = softmax(logits)
    ranked = sorted(enumerate(probs), key=lambda x: -x[1])
    selected = ranked[:k]
    total = sum(p for _, p in selected)
    return [(i, p/total) for i, p in selected], probs

# Simulate 8-expert MoE routing for 20 tokens
E, K, d = 8, 2, 4
# Random router weights (E x d)
router = [[random.gauss(0, 0.5) for _ in range(d)] for _ in range(E)]
# Random token embeddings
tokens = [[random.gauss(0, 1) for _ in range(d)] for _ in range(20)]

expert_counts = [0] * E
expert_load = [0.0] * E

print(f"MoE routing: {E} experts, Top-{K} selection, 20 tokens")
print(f"{'Token':>5}  {'Selected experts':30}  Probabilities")
for t_idx, token in enumerate(tokens):
    logits = [sum(token[j] * router[i][j] for j in range(d)) for i in range(E)]
    selected, probs = top_k_route(logits, K)
    for i, w in selected:
        expert_counts[i] += 1
        expert_load[i] += w
    exp_str = " + ".join(f"E{i+1}({w:.2f})" for i, w in selected)
    print(f"  T{t_idx+1:02d}:  {exp_str:<30}")

print()
print("Expert utilisation (routing collapse check):")
print(f"  {'Expert':>8}  {'Tokens':>6}  {'Load':>8}  Balance bar")
avg_count = sum(expert_counts) / E
for i in range(E):
    bar = "#" * int(expert_counts[i] * 3)
    imbalance = "OVERLOADED" if expert_counts[i] > avg_count * 1.5 else ""
    print(f"  Expert {i+1:1d}  {expert_counts[i]:>6d}  {expert_load[i]:>8.2f}  {bar} {imbalance}")

print(f"\nIdeal tokens/expert: {sum(expert_counts)/E:.1f}")
print(f"Std dev: {(sum((c - sum(expert_counts)/E)**2 for c in expert_counts)/E)**0.5:.2f}")
print("(Lower std dev = better load balance = less routing collapse)")
# Try changing K from 2 to 1 to see more routing collapse</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/2101.03961" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Fedus et al., 2022 &#8212; Switch Transformers: Scaling to Trillion Parameter Models</span> &#8212; Top-1 routing, 1.6T parameters, 7× faster pre-training vs T5-11B at equal compute. Introduced capacity factor and auxiliary loss.</span></a>
      <a href="https://arxiv.org/abs/2401.04088" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jiang et al., 2024 &#8212; Mixtral of Experts</span> &#8212; 8×7B MoE, 46.7B total, 12.9B active. Outperforms LLaMA-2 70B on most benchmarks. 5× faster inference vs dense model of equal quality. Open weights.</span></a>
      <a href="https://arxiv.org/abs/2403.05530" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Google DeepMind, 2024 &#8212; Gemini 1.5: Unlocking Multimodal Understanding</span> &#8212; MoE architecture with 1M-token context via ring attention. 10.5M-token context achieved in research setting.</span></a>
      <a href="https://arxiv.org/abs/1701.06538" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Shazeer et al., 2017 &#8212; Outrageously Large Neural Networks (original MoE paper)</span> &#8212; LSTM + MoE achieving 1000× capacity increase. Introduced noisy top-K gating and the load balancing problem. Foundation for all modern MoE work.</span></a>
      <a href="https://huggingface.co/blog/moe" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">HuggingFace Blog &#8212; Mixture of Experts Explained (2024)</span> &#8212; Visual guide to MoE routing, capacity factors, and expert specialisation. Includes profiling of which tokens go to which expert in Mixtral.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
""")

content = "".join(lines)
with open(OUT, 'a', encoding='utf-8') as f:
    f.write(content)
print(f"S13-S15 appended: {len(content)} chars")
