#!/usr/bin/env python3
# Appends sections 1.2 (Tokenisation) and 1.3 (Embeddings)
import os
OUT = os.path.join(os.path.dirname(__file__), 'chapter1.html')

S2 = """
<!-- ============================================================ §1.2 == -->
<section class="section" id="s2">
  <div class="section-label">1.2</div>
  <h2>Tokenisation &amp; BPE</h2>

  <p>Models do not see characters, words, or bytes in isolation &#8212; they see <em>tokens</em>, the subword units discovered by a compression algorithm. The dominant approach is <strong>Byte-Pair Encoding (BPE)</strong>, introduced by Sennrich et al. (2016, arXiv:1508.07909) for neural machine translation and later adapted by OpenAI for GPT-2 onwards. BPE begins with the individual bytes of the UTF-8 encoding (256 initial symbols) and iteratively merges the most frequent adjacent pair until a target vocabulary size is reached. GPT-4 uses the <code>cl100k_base</code> encoding with exactly <strong>100,257 tokens</strong>.</p>

  <p>The choice of vocabulary size involves a real trade-off. A vocabulary of 50,000 tokens (GPT-2) means fewer unique IDs but more tokens per sentence; 100,000+ tokens (GPT-4, LLaMA-3's 128,256) means more IDs to embed but shorter sequences and cheaper attention. LLaMA-3 uses <strong>SentencePiece</strong> (Kudo &amp; Richardson, 2018), which operates on raw bytes without language-specific pre-tokenisation, making it language-agnostic &#8212; crucial for a multilingual model.</p>

  <div class="callout blue">
    <strong>Token fertility &#8212; not all languages are equal.</strong> English averages roughly 1 token per 4 characters. Turkish, Finnish, and other agglutinative languages often use 2&#8211;3 tokens per word because their rich morphology produces many rare compound forms not in the BPE vocabulary. This means processing Turkish text costs 2&#8211;3&#215; more compute, and early LLMs are correspondingly worse at it.
  </div>

  <p>BPE has famous failure modes that reveal important principles. The string <code>SolidGoldMagikarp</code> is a Reddit username that appeared so infrequently in GPT-3's training corpus that its token embedding was never updated, causing the model to output erratic responses when prompted with it. Numbers tokenise inconsistently: <code>1234</code> might be one token, but <code>12345</code> could be three, which explains why LLMs struggle with multi-digit arithmetic &#8212; they cannot see the positional relationships between digits. Spaces before words get their own tokens (&#8220; the&#8221; is different from &#8220;the&#8221;), which is why prompt formatting subtleties matter more than they should.</p>

  <div class="code-block">
    <span class="code-label">BPE from scratch (pure Python)</span>
    <pre><span class="kw">from</span> collections <span class="kw">import</span> Counter

<span class="kw">def</span> <span class="fn">get_pairs</span>(vocab):
    <span class="lm"># &#8593; count every adjacent symbol pair across all words</span>
    pairs = Counter()
    <span class="kw">for</span> word, freq <span class="kw">in</span> vocab.items():
        syms = word.split()
        <span class="kw">for</span> i <span class="kw">in</span> <span class="bi">range</span>(<span class="bi">len</span>(syms) - <span class="nu">1</span>):
            pairs[(syms[i], syms[i+<span class="nu">1</span>])] += freq    <span class="lm"># &#8593; weighted by word frequency</span>
    <span class="kw">return</span> pairs

<span class="kw">def</span> <span class="fn">merge_vocab</span>(pair, vocab):
    <span class="lm"># &#8593; replace the most frequent pair everywhere in the vocabulary</span>
    bigram  = <span class="st">" "</span>.join(pair)
    merged  = <span class="st">""</span>.join(pair)
    <span class="kw">return</span> {w.replace(bigram, merged): f <span class="kw">for</span> w, f <span class="kw">in</span> vocab.items()}

<span class="lm"># Start: every character is its own symbol</span>
<span class="lm"># &lt;/w&gt; marks end-of-word (boundary token)</span>
vocab = {<span class="st">"l o w &lt;/w&gt;"</span>: <span class="nu">5</span>, <span class="st">"l o w e r &lt;/w&gt;"</span>: <span class="nu">2</span>,
         <span class="st">"n e w e s t &lt;/w&gt;"</span>: <span class="nu">6</span>, <span class="st">"w i d e s t &lt;/w&gt;"</span>: <span class="nu">3</span>}

<span class="kw">for</span> i <span class="kw">in</span> <span class="bi">range</span>(<span class="nu">10</span>):
    pairs = get_pairs(vocab)
    best  = <span class="bi">max</span>(pairs, key=pairs.get)   <span class="lm"># &#8593; greedily pick most frequent pair</span>
    vocab = merge_vocab(best, vocab)
    <span class="bi">print</span>(<span class="fn">f</span><span class="st">f"Merge {i+1:2d}: {best[0]+' '+best[1]:14s} &#8594; {''.join(best)}"</span>)</pre>
  </div>

  <div class="cell" id="cell-2">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; BPE merge trace + token fertility</span>
      <button class="run-btn" onclick="runCell('cell-2')">Run</button>
    </div>
    <div class="cell-code"><pre>from collections import Counter

def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        syms = word.split()
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = " ".join(pair)
    merged = "".join(pair)
    return {w.replace(bigram, merged): f for w, f in vocab.items()}

vocab = {"l o w </w>": 5, "l o w e r </w>": 2,
         "n e w e s t </w>": 6, "w i d e s t </w>": 3}

print("=== BPE Merge Trace (Sennrich et al. 2016 example) ===")
print(f"Initial vocab: {list(vocab.keys())}")
print()
for i in range(10):
    pairs = get_pairs(vocab)
    best = max(pairs, key=pairs.get)
    count = pairs[best]
    vocab = merge_vocab(best, vocab)
    print(f"Merge {i+1:2d}: '{best[0]+' '+best[1]}' -> '{''.join(best)}'  (freq={count})")

print(f"\nFinal vocab tokens: {list(vocab.keys())}")

# Token fertility simulation (rough estimate via BPE-like logic)
print("\n=== Token Fertility by Language ===")
print("(Approximate tokens per word; real tokenisers vary)")
print(f"{'Language':16s} {'Example word':22s} {'~Tokens':>8s}")
print("-" * 50)
words = [
    ("English",    "internationalization",  3),
    ("English",    "cat",                   1),
    ("Turkish",    "gidebilecekmisiniz",    5),
    ("Finnish",    "juoksennellessanikaan", 7),
    ("Chinese",    "\u6211\u7231\u4e2d\u6587",  2),
    ("Arabic",     "sa-aktub",              3),
]
for lang, word, tok in words:
    bar = "#" * tok
    print(f"  {lang:14s} {word:22s} ~{tok}  {bar}")

print("\nKey insight: richer morphology = more tokens = more compute cost.")
print("LLaMA-3's 128K vocab reduces fertility vs GPT-2's 50K vocab.")

# Try changing the vocabulary example words above and re-running
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <!-- Interactive: vocabulary size slider -->
  <div class="slider-wrap">
    <div style="font-family:var(--font-mono);font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:var(--text-secondary);margin-bottom:12px;">Interactive &#8212; Vocabulary Size Trade-off</div>
    <label>Vocabulary size: <strong id="vocab-val">50,000</strong> tokens</label>
    <input type="range" id="vocab-slider" min="10000" max="150000" step="5000" value="50000" oninput="updateVocabCalc()">
    <div class="slider-output" id="vocab-output">Move the slider to compute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1508.07909" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Sennrich et al., 2016 &#8212; "Neural Machine Translation of Rare Words with Subword Units"</span> &#8212; The BPE paper. Byte-pair encoding for open-vocabulary NMT; iterative greedy merging.</span></a>
      <a href="https://arxiv.org/abs/1808.06226" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Kudo &amp; Richardson, 2018 &#8212; "SentencePiece: A simple and language independent subword tokenizer"</span> &#8212; Language-agnostic subword tokeniser used by LLaMA-3, T5, and most modern multilingual LLMs.</span></a>
      <a href="https://arxiv.org/abs/1810.04805" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Devlin et al., 2019 &#8212; "BERT"</span> &#8212; Uses WordPiece tokenisation: maximises training corpus likelihood rather than raw merge frequency.</span></a>
      <a href="https://www.youtube.com/watch?v=zduSFxRajkE" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Andrej Karpathy &#8212; "Let's build the GPT Tokenizer" (2024, 2h video)</span> &#8212; Builds BPE from scratch in Python including the GPT-4 tiktoken encoding. Highest-signal tutorial available.</span></a>
      <a href="https://github.com/karpathy/minbpe" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Karpathy &#8212; minbpe (GitHub)</span> &#8212; Minimal clean BPE implementation; readable 200-line reference code.</span></a>
      <a href="https://huggingface.co/docs/tokenizers/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Hugging Face &#8212; Tokenizers documentation</span> &#8212; Production-grade library supporting BPE, WordPiece, and Unigram with Rust-speed implementations.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">

<!-- ============================================================ §1.3 == -->
<section class="section" id="s3">
  <div class="section-label">1.3</div>
  <h2>Embeddings &amp; Vector Geometry</h2>

  <p>Each of the 100,257 tokens in GPT-4's vocabulary is mapped to a dense vector in &#8477;<sup>d</sup> &#8212; its <em>embedding</em>. These vectors are the model's first representation of language, and they are learned entirely from data. The geometry of the resulting space encodes semantic relationships: similar tokens cluster together, and vector arithmetic sometimes mirrors conceptual arithmetic. The famous result <em>king &#8722; man + woman &#8776; queen</em> (Mikolov et al., 2013) is a property that emerges automatically from the distributional statistics of text &#8212; nobody programmed it in.</p>

  <p>Modern LLMs use embedding dimensions of <strong>4,096</strong> (LLaMA-3 8B, Mistral 7B) to <strong>12,288</strong> (GPT-3 175B). The embedding table alone is substantial: LLaMA-3's 128,256 tokens &#215; 4,096 dims &#215; 2 bytes (BF16) = <strong>1.05 GB</strong> just for the embedding matrix. This is why <strong>weight tying</strong> (Press &amp; Wolf, 2017) &#8212; using the same matrix for the input embedding and the output unembedding projection &#8212; saves memory and improves performance by ensuring input and output token representations are aligned.</p>

  <div class="callout green">
    <strong>The antonym paradox.</strong> "hot" and "cold" have high cosine similarity despite being conceptual opposites, because they appear in identical syntactic contexts ("it was very ___", "the temperature is ___"). Word embeddings capture <em>distributional</em> similarity (same contexts), not semantic opposition. This is a fundamental limitation: embedding geometry encodes co-occurrence, not meaning. Transformer contextual representations partially overcome this, but the base embedding space is blind to negation.
  </div>

  <div class="cell" id="cell-3">
    <div class="cell-header">
      <span class="cell-label">&#9654; RUNNABLE &#8212; Vector arithmetic &amp; cosine similarity</span>
      <button class="run-btn" onclick="runCell('cell-3')">Run</button>
    </div>
    <div class="cell-code"><pre>import math

# 6-dimensional structured embeddings (not real, illustrative)
# dims: [royalty, gender_masc, animate, place, abstract, negative]
words = {
    "king":   [0.9, 0.9, 0.8, 0.0, 0.1, 0.0],
    "queen":  [0.9, 0.1, 0.8, 0.0, 0.1, 0.0],
    "man":    [0.1, 0.9, 0.8, 0.0, 0.0, 0.0],
    "woman":  [0.1, 0.1, 0.8, 0.0, 0.0, 0.0],
    "paris":  [0.2, 0.5, 0.0, 0.9, 0.1, 0.0],
    "france": [0.2, 0.5, 0.0, 0.8, 0.3, 0.0],
    "london": [0.2, 0.5, 0.0, 0.9, 0.1, 0.0],
    "hot":    [0.0, 0.0, 0.0, 0.0, 0.7, 0.0],
    "cold":   [0.0, 0.0, 0.0, 0.0, 0.7, 0.4],
}

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def cosine(a, b):
    n = norm(a) * norm(b)
    return dot(a, b) / n if n > 0 else 0.0

def vadd(a, b):
    return [x + y for x, y in zip(a, b)]

def vsub(a, b):
    return [x - y for x, y in zip(a, b)]

def nearest(vec, exclude=[]):
    scores = {w: cosine(vec, v) for w, v in words.items() if w not in exclude}
    return sorted(scores.items(), key=lambda x: -x[1])[:3]

# Classic word analogy: king - man + woman
result = vadd(vsub(words["king"], words["man"]), words["woman"])
print("king - man + woman =>")
for w, s in nearest(result, ["king", "man", "woman"]):
    print(f"  {w:8s}  cosine={s:.4f}")

# Geometry: paris - france + london (capital analogy)
result2 = vadd(vsub(words["paris"], words["france"]), words["london"])
print("\nparis - france + london =>")
for w, s in nearest(result2, ["paris", "france", "london"]):
    print(f"  {w:8s}  cosine={s:.4f}")

# The antonym paradox
print("\nCosine similarities (antonym paradox):")
print(f"  hot  vs cold  : {cosine(words['hot'],  words['cold']):.4f}  <- high despite opposites!")
print(f"  king vs queen : {cosine(words['king'], words['queen']):.4f}")
print(f"  king vs paris : {cosine(words['king'], words['paris']):.4f}")
print(f"  man  vs woman : {cosine(words['man'],  words['woman']):.4f}")

# Embedding table memory
print("\nEmbedding table memory (bytes):")
for name, V, d, bits in [
    ("GPT-2",    50257,  768,  16),
    ("GPT-3",    50257, 12288, 16),
    ("LLaMA-3",128256,  4096, 16),
    ("GPT-4 est",100257, 8192, 16),
]:
    mb = V * d * (bits // 8) / 1e6
    print(f"  {name:12s} V={V:7d} d={d:6d} -> {mb:8.1f} MB")

# Try changing the word vectors above to explore different semantic spaces
</pre></div>
    <div class="cell-output">Click Run to execute.</div>
  </div>

  <div class="deepdive">
    <div class="deepdive-label">Deep Dive &#8212; Learn More</div>
    <div class="deepdive-links">
      <a href="https://arxiv.org/abs/1301.3781" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Mikolov et al., 2013 &#8212; "Efficient Estimation of Word Representations in Vector Space"</span> &#8212; Word2Vec skip-gram and CBOW. The king&#8722;man+woman=queen result that launched modern embeddings.</span></a>
      <a href="https://arxiv.org/abs/1408.5882" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Pennington et al., 2014 &#8212; "GloVe: Global Vectors for Word Representation"</span> &#8212; Co-occurrence matrix factorisation; combines global statistics with local context.</span></a>
      <a href="https://arxiv.org/abs/1608.05859" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Press &amp; Wolf, 2017 &#8212; "Using the Output Embedding to Improve Language Models"</span> &#8212; Weight tying: reuse the input embedding matrix for the output projection; saves memory and improves quality.</span></a>
      <a href="https://jalammar.github.io/illustrated-word2vec/" target="_blank"><span class="link-icon">&#8599;</span> <span><span class="link-label">Jay Alammar &#8212; "The Illustrated Word2Vec"</span> &#8212; Visual explanation of skip-gram training, negative sampling, and the king-queen geometry.</span></a>
    </div>
  </div>
</section>
<hr class="section-divider">
"""

with open(OUT, 'a', encoding='utf-8') as f:
    f.write(S2)
print(f"S2+S3 appended: {len(S2)} chars")
