#!/usr/bin/env python3
# Appends section 1.1 — What Is a Large Language Model?
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

S1 = """
<!-- ============================================================ §1.1 == -->
<section class="section" id="s1">
  <div class="section-label">1.1</div>
  <h2>What Is a Large Language Model?</h2>

  <p>A large language model is, at its core, a function that maps a sequence of tokens to a probability distribution over the next token. That deceptively simple objective — predict what comes next — when applied at sufficient scale gives rise to systems that translate, reason, write code, and hold coherent conversations. The beauty and strangeness of LLMs is that <em>no capability was directly programmed</em>: every emergent behaviour is a consequence of gradient descent minimising cross-entropy loss on raw text.</p>

  <p>The word "large" is doing two jobs. It refers to <strong>parameter count</strong> — GPT-3 (Brown et al., 2020, arXiv:2005.14165) introduced 175 billion parameters, then a record; GPT-4 is estimated at roughly 1 trillion — and to <strong>training data</strong>, measured in tokens. GPT-3 consumed 300 billion tokens; LLaMA-3 (Meta, 2024) consumed 15 trillion tokens across its full training run. Neither alone is sufficient: the Chinchilla scaling laws (Hoffmann et al., 2022) formalised that model size and token count must scale together, roughly 20 tokens per parameter, for compute-optimal training.</p>

  <div class="callout">
    <strong>The autoregressive loop.</strong> At inference time the model is called repeatedly. The output token is appended to the input, the model runs again — until an end-of-sequence token is produced or a length limit is hit. Every output token is a one-way commitment: there is no backtracking. This is why streaming output is possible in real-time chat interfaces — each token is genuinely computed one at a time.
  </div>

  <div class="ar-loop">
    <div class="ar-tokens">
      <div class="ar-tok ctx">The</div>
      <div class="ar-tok ctx">cat</div>
      <div class="ar-tok ctx">sat</div>
      <div class="ar-tok ctx">on</div>
      <div class="ar-tok ctx">the</div>
      <div class="ar-tok new">mat</div>
    </div>
    <div class="ar-arrow">&#8595; softmax over vocabulary (~100,257 tokens for GPT-4's cl100k_base)</div>
    <div class="ar-probs">
      <div class="ar-bar-wrap"><div class="ar-bar top" style="height:55px;"></div><div class="ar-bar-label">mat</div></div>
      <div class="ar-bar-wrap"><div class="ar-bar" style="height:28px;"></div><div class="ar-bar-label">floor</div></div>
      <div class="ar-bar-wrap"><div class="ar-bar" style="height:18px;"></div><div class="ar-bar-label">roof</div></div>
      <div class="ar-bar-wrap"><div class="ar-bar" style="height:10px;"></div><div class="ar-bar-label">sofa</div></div>
      <div class="ar-bar-wrap"><div class="ar-bar" style="height:6px;"></div><div class="ar-bar-label">&#8230;</div></div>
    </div>
    <div class="ar-arrow">&#8594; sample or argmax &#8594; append &#8594; repeat</div>
  </div>

  <p>Choosing <em>how</em> to select the next token from the probability distribution is the decoding strategy, and it matters enormously. <strong>Greedy decoding</strong> (always take the highest-probability token) is fast but produces repetitive, hollow text. <strong>Beam search</strong> — long the gold standard in machine translation — maintains B candidate sequences simultaneously and picks the highest-probability full sequence. Holtzman et al. (2020, arXiv:1904.09751) showed that beam search produces text that is systematically less surprising than human text, falling into repetitive loops. Their solution, <strong>nucleus (top-p) sampling</strong>, truncates the vocabulary to the smallest set of tokens whose cumulative probability mass exceeds <em>p</em>, then samples from that nucleus. At p=0.9 only the top 5–20 tokens typically remain; at p=1.0 the full vocabulary is active. This small change produces dramatically more natural, diverse text.</p>

  <p><strong>Temperature</strong> scales the logits before softmax: <em>temperature &lt; 1</em> sharpens the distribution (more deterministic), <em>temperature &gt; 1</em> flattens it (more random). Most production deployments use temperature 0.7–1.0 combined with top-p=0.9. Setting temperature to exactly 0 (greedy) is common for benchmarking and reasoning tasks where reproducibility matters more than diversity.</p>

  <div class="code-block">
    <span class="code-label">Python</span>
    <pre><span class="kw">import</span> math, random

<span class="kw">def</span> <span class="fn">softmax</span>(logits, temperature=<span class="nu">1.0</span>):
    <span class="lm"># &#8593; temperature &gt; 1 flattens the distribution (more random)</span>
    <span class="lm"># &#8593; temperature &lt; 1 sharpens it (more greedy)</span>
    scaled = [x / temperature <span class="kw">for</span> x <span class="kw">in</span> logits]
    max_v  = <span class="bi">max</span>(scaled)                      <span class="lm"># &#8593; subtract max for numerical stability</span>
    exps   = [math.exp(x - max_v) <span class="kw">for</span> x <span class="kw">in</span> scaled]
    total  = <span class="bi">sum</span>(exps)
    <span class="kw">return</span> [e / total <span class="kw">for</span> e <span class="kw">in</span> exps]

<span class="kw">def</span> <span class="fn">top_p_sample</span>(probs, p=<span class="nu">0.9</span>):
    <span class="lm"># &#8593; sort tokens by probability (highest first)</span>
    sorted_pairs = <span class="bi">sorted</span>(<span class="bi">enumerate</span>(probs), key=<span class="kw">lambda</span> x: -x[<span class="nu">1</span>])
    cumulative, nucleus = <span class="nu">0.0</span>, []
    <span class="kw">for</span> idx, prob <span class="kw">in</span> sorted_pairs:
        nucleus.append((idx, prob))
        cumulative += prob
        <span class="kw">if</span> cumulative &gt;= p: <span class="kw">break</span>          <span class="lm"># &#8593; stop once we cover p of the mass</span>
    total = <span class="bi">sum</span>(p2 <span class="kw">for</span> _, p2 <span class="kw">in</span> nucleus)
    r = random.random() * total
    <span class="kw">for</span> idx, prob <span class="kw">in</span> nucleus:             <span class="lm"># &#8593; sample from renormalised nucleus</span>
        r -= prob
        <span class="kw">if</span> r &lt;= <span class="nu">0</span>: <span class="kw">return</span> idx
    <span class="kw">return</span> nucleus[-<span class="nu">1</span>][<span class="nu">0</span>]</pre>
  </div>

  <div class="cell" id="cell-1">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Sampling strategies compared</span>
      <button class="run-btn" onclick="runCell('cell-1')">Run</button>
    </div>
    <div class="cell-code"><pre>import math, random

random.seed(42)

vocab = ["mat","floor","roof","sofa","table","bed","ground","rug","deck","wall"]
logits = [3.2, 2.1, 1.4, 1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]

def softmax(logits, temperature=1.0):
    scaled = [x / temperature for x in logits]
    max_v = max(scaled)
    exps = [math.exp(x - max_v) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def top_p_sample(probs, p=0.9):
    pairs = sorted(enumerate(probs), key=lambda x: -x[1])
    cum, nucleus = 0.0, []
    for idx, prob in pairs:
        nucleus.append((idx, prob))
        cum += prob
        if cum >= p: break
    total = sum(p2 for _, p2 in nucleus)
    r = random.random() * total
    for idx, prob in nucleus:
        r -= prob
        if r <= 0: return idx
    return nucleus[-1][0]

probs = softmax(logits)
print("Token probabilities (temp=1.0):")
for w, p in zip(vocab, probs):
    bar = "#" * int(p * 60)
    print(f"  {w:8s} {p:.4f}  {bar}")

print(f"\nGreedy (argmax): {vocab[probs.index(max(probs))]}")

samples = [vocab[top_p_sample(softmax(logits, 1.0))] for _ in range(10)]
print(f"\nTop-p=0.9, temp=1.0 (10 samples): {samples}")

samples_sharp = [vocab[top_p_sample(softmax(logits, 0.3))] for _ in range(10)]
print(f"\nTop-p=0.9, temp=0.3 (sharper):    {samples_sharp}")

samples_rand = [vocab[top_p_sample(softmax(logits, 2.0))] for _ in range(10)]
print(f"\nTop-p=0.9, temp=2.0 (more random): {samples_rand}")

# Show nucleus size at various p values
print("\nNucleus size at various p cutoffs:")
for p in [0.5, 0.7, 0.9, 0.95, 1.0]:
    pairs = sorted(enumerate(probs), key=lambda x: -x[1])
    cum, n = 0, 0
    for _, prob in pairs:
        cum += prob; n += 1
        if cum >= p: break
    print(f"  p={p:.2f}  nucleus={n} tokens ({n}/{len(vocab)} of vocab)")

# Try changing temperature and re-running to see distribution shift
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1706.03762" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Vaswani et al., 2017 &#8212; "Attention Is All You Need"</span> &#8212; The paper that started it all. The original transformer: 6-layer encoder-decoder, d_model=512, 8 heads, ~65M params.</span></a>
      <a href="https://arxiv.org/abs/1904.09751" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Holtzman et al., 2020 &#8212; "The Curious Case of Neural Text Degeneration"</span> &#8212; Explains why greedy/beam search fails and introduces nucleus (top-p) sampling with human evaluation evidence.</span></a>
      <a href="https://arxiv.org/abs/2005.14165" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Brown et al., 2020 &#8212; "Language Models are Few-Shot Learners" (GPT-3)</span> &#8212; 175B parameters, 300B training tokens, in-context learning without fine-tuning. The paper that defined the modern era.</span></a>
      <a href="https://arxiv.org/abs/2203.15556" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Hoffmann et al., 2022 &#8212; "Training Compute-Optimal Large Language Models" (Chinchilla)</span> &#8212; 70B model on 1.4T tokens beats 280B Gopher. The ~20 tokens/parameter rule.</span></a>
      <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jay Alammar &#8212; "The Illustrated Transformer" (2018)</span> &#8212; The best visual walkthrough of the transformer architecture. Featured in courses at Stanford, MIT, Harvard, and CMU.</span></a>
      <a href="https://lilianweng.github.io/posts/2018-06-24-attention/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Lilian Weng &#8212; "Attention? Attention!" (2018, updated 2023)</span> &#8212; Comprehensive survey of all major attention mechanisms from Bahdanau to transformers.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
"""

with open(OUT, 'a', encoding='utf-8') as f:
    f.write(S1)
print(f"S1 appended: {len(S1)} chars")
