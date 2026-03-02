# Book Authoring Instructions
## *Building With Large Language Models* — by Jace Sabr

> This file is the spec for generating every chapter of the book.
> Hand it to Claude with the prompt: **"Write chapter N using BOOK_INSTRUCTIONS.md as your spec."**

---

## The Book

A practitioner's guide to understanding and building large language models — from the mathematics of next-token prediction through fine-tuning, deployment, and production. Each chapter explains one layer of the stack deeply, with annotated code the reader can run and links to every primary source.

**Target reader:** Someone technical enough to read Python, curious enough to want the real math, but not necessarily a researcher. A senior developer, a data scientist pivoting to LLMs, or a technically-minded builder who wants to go beyond the API.

**Not this book:** A tour of prompt engineering tips. A list of LLM tools. A surface-level "AI for beginners." The goal is genuine understanding, not just familiarity.

---

## Chapter Plan

| # | Title | Core Concepts |
|---|-------|---------------|
| 1 | Understanding Large Language Models | Next-token prediction, tokenisation, embeddings, transformer architecture, self-attention, pretraining, BERT vs GPT, RLHF/LoRA, scale/emergence, limitations, open-source landscape |
| 2 | Working with Text Data | Tokenisers in depth (BPE, SentencePiece, tiktoken), building a BPE tokenizer from scratch, data cleaning, dataset loading, batching, the DataLoader pattern |
| 3 | Attention Mechanisms | Scaled dot-product attention from scratch, multi-head attention, causal mask, Flash Attention intuition, RoPE, GQA, KV caching |
| 4 | The GPT Architecture | Full decoder-only transformer: token + position embeddings, stacked transformer blocks, pre-norm, weight tying, parameter counting, GPT-2 architecture deep-dive |
| 5 | Pre-training | Training loop, gradient clipping, learning rate scheduling (cosine with warmup), mixed precision (BF16), gradient checkpointing, evaluation on validation loss, loading real pretrained weights |
| 6 | Fine-tuning for Classification | Adding a classification head, training on a downstream dataset, evaluation metrics (F1, ROC-AUC), overfitting, early stopping |
| 7 | Fine-tuning for Instruction Following | SFT dataset format (Alpaca, ShareGPT, ChatML), training with masked loss, DPO, evaluation with benchmarks, deploying a local assistant |

---

## Writing Style

### Voice
- **Authoritative but not academic.** Write like a senior engineer explaining something to a smart colleague, not like a textbook.
- **Precise.** Never say "the model learns patterns" without immediately explaining *which* patterns and *how*. Every claim should be traceable to a number, a paper, or a demonstration.
- **Direct.** No filler phrases ("As we can see above...", "It's important to note that..."). Cut them.
- **First-principles.** Always explain *why* before *how*. Why does self-attention use softmax? Why do we divide by √d_k? Why does RLHF need a KL penalty?

### Tone
- Engaged, not dry. It's okay to have a perspective.
- The reader is smart — don't over-explain. But do explain *once*, clearly.
- Surprising facts and counterintuitive results are gold. Lead with them when they exist.

### What Every Section Must Have
1. **Prose explanation** — what it is, why it exists, what problem it solves
2. **The key insight** — the one thing the reader should walk away understanding
3. **At least one concrete number** — a parameter count, a cost, a benchmark score, a token count
4. **An annotated code block** — static Python with `<!-- lm -->` blue comments for plain-English explanations
5. **A runnable Pyodide cell** — pure Python (standard library only, no numpy/torch) the reader can edit and run in their browser
6. **A callout box** — for the most important insight, an analogy, or a warning
7. **A deep-dive section** — links to the primary papers, Jay Alammar's visual posts, Lilian Weng's blog, official docs

---

## HTML Structure Patterns

Every chapter is a single HTML file at `resume/book/chapterN.html`. Copy the CSS and JavaScript from chapter1.html exactly — never modify them. Only write the `<body>` content.

### Section template
```html
<section class="section" id="sN">
  <div class="section-label">N.M</div>
  <h2>Section Title</h2>

  <p>Opening paragraph — what this section is about and why it matters.</p>
  <p>Second paragraph — the key concept, explained precisely.</p>
  <p>Third paragraph — the nuance, the counterintuitive part, the implication.</p>

  <div class="callout"><strong>Key insight:</strong> The most important takeaway in one sentence.</div>

  <!-- Static annotated code block -->
  <div class="overview">
    <div class="overview-label">What this code does</div>
    <p>Plain English description of the code below.</p>
    <ol>
      <li>Step 1 explanation</li>
      <li>Step 2 explanation</li>
    </ol>
  </div>
  <div class="code-block">
    <div class="code-header">
      <span class="code-filename">filename.py</span>
      <button class="code-copy" onclick="copyCode(this)">Copy</button>
    </div>
<pre><code><!-- annotated code with .kw .fn .str .cm .lm .num .cls .pr .bi spans --></code></pre>
  </div>
  <div class="deepdive">
    <div class="deepdive-label">Deep Dive — Learn More</div>
    <div class="deepdive-links">
      <a href="URL" target="_blank"><span class="link-icon">↗</span> <span><span class="link-label">Title</span> — Description</span></a>
    </div>
  </div>

  <!-- Runnable Pyodide cell (pure Python, standard library only) -->
  <div class="run-block" id="run-N">
    <div class="run-header">
      <span class="run-label">▶ Try it — description</span>
      <button class="run-btn" onclick="runCode('run-N')">▶ Run</button>
    </div>
    <textarea class="run-editor" spellcheck="false">
# Pure Python code here — no numpy, no torch, no external imports
# Standard library (math, random, collections, itertools) is fine
    </textarea>
    <div class="run-output"></div>
  </div>

  <div class="divider"></div>
</section>
```

### Syntax highlighting classes (use in `<span>` tags inside `<pre><code>`):
| Class | Colour | Use for |
|-------|--------|---------|
| `.kw` | purple | Python keywords: `import`, `def`, `class`, `for`, `if`, `return`, `with` |
| `.fn` | yellow | Function names when called: `softmax(`, `model.forward(` |
| `.str` | orange | String literals: `"hello"`, `'text'` |
| `.cm` | green italic | Standard comments: `# this is a section header` |
| `.lm` | blue | **Layman comments** — the most important class. Plain-English explanation of the preceding line. Always starts with `# ↑` |
| `.num` | light green | Numeric literals: `0.01`, `175e9`, `128_000` |
| `.cls` | teal | Class names: `TinyLM`, `SelfAttention` |
| `.pr` | light blue | `self` parameter |
| `.bi` | cyan | Python builtins: `print`, `len`, `range`, `enumerate`, `sum` |

### Layman comment convention
```python
self.embedding = nn.Embedding(vocab_size, embed_dim)
# ↑ A lookup table: word ID → 16-number "personality" vector.
#   Starts random, gets refined during training so similar words
#   end up with similar vectors.
```
Every non-trivial line gets a `# ↑` layman comment. These blue comments are the book's most valuable feature — they make the code readable to non-experts without dumbing it down.

---

## Research Process (Before Writing)

Before writing any section, search:
1. **Jay Alammar's blog** (jalammar.github.io) — best visual explanations of attention, transformers, BERT, GPT
2. **Lilian Weng's blog** (lilianweng.github.io) — comprehensive survey posts on attention, alignment, LLM agents
3. **Sebastian Raschka** (magazine.sebastianraschka.com) — hands-on LLM articles with code
4. **Karpathy** (karpathy.ai, karpathy.github.io) — nanoGPT, char-RNN, micrograd — excellent minimal implementations
5. **Hugging Face blog** — practical fine-tuning and deployment posts
6. **The primary papers** — link to arxiv.org for every claim that comes from research
7. **Anthropic research** (transformer-circuits.pub) — mechanistic interpretability

For every major concept, the deep-dive section should contain:
- The original paper
- The best visual explanation (Alammar or 3Blue1Brown when available)
- A practical implementation resource (PyTorch docs, HuggingFace docs, or GitHub repo)

---

## Runnable Cell Guidelines

Pyodide loads Python's standard library but not numpy, torch, or any third-party packages (unless explicitly installed with `micropip`). Every runnable cell must work with only:
- `math` — for sqrt, exp, log
- `random` — for sampling, masking
- `collections` — Counter, defaultdict
- `itertools` — for sequence operations

**Good runnable cell topics:**
- Softmax probability distribution
- BPE merge algorithm
- Cosine similarity between vectors
- Attention weight matrix (pure Python matmul)
- Token pair generation (BERT masking, GPT left-to-right)
- LoRA parameter count comparison
- Perplexity calculation
- Temperature and top-k sampling
- Gradient descent by hand on a toy example

**Cells should be editable and interesting** — the reader should want to change a number and re-run. Always end with a comment like "Try changing X above and re-running."

---

## Quality Checklist (per chapter)

Before finalising:
- [ ] Every section has a prose explanation, annotated code, deepdive, and runnable cell
- [ ] All `# ↑` layman comments are present on every non-trivial code line
- [ ] Every deep-dive link is to a real, specific URL (not a homepage)
- [ ] The chapter summary table covers all key takeaways
- [ ] The TOC links match the actual section IDs
- [ ] Chapter nav link at bottom points to the next chapter
- [ ] All runnable cells use only Python standard library
- [ ] The chapter book/index.html card is updated to "Complete — Read →"

---

## Chapter Index File (`resume/book/index.html`)

The book's table of contents. Each chapter card becomes a link once the chapter is complete:
```html
<a href="chapter2.html" class="ch">
  <div class="ch-num">02</div>
  <div class="ch-title">Working with Text Data</div>
  <div class="ch-status">Complete — Read →</div>
</a>
```
Incomplete chapters:
```html
<div class="ch">
  <div class="ch-num">03</div>
  <div class="ch-title">Attention Mechanisms</div>
  <div class="ch-status">Coming soon</div>
</div>
```

---

## Authoring Workflow

1. **You read** the corresponding section in the source textbook (Raschka's *Build a Large Language Model from Scratch*, or any resource you're learning from)
2. **You note** what you learned, what confused you, what you want explained differently
3. **You give Claude:** "Write chapter 2 using BOOK_INSTRUCTIONS.md as spec. The chapter should cover [topic list]. I want extra depth on [X] because I found it confusing. Include a good analogy for [Y]."
4. **Claude writes** — deep research first, then full chapter draft
5. **You edit** — correct anything wrong, add your own perspective, change analogies to ones that resonate with you
6. **The chapter goes live** on your resume book page

This is your book. Claude is your research assistant and first-draft writer. The final voice, choices, and examples should reflect your understanding.

---

## Notes on Voice from Chapter 1

The established voice from chapter 1:
- Opens with a precise definition, not a hook or anecdote
- Uses bold for the most important terms on first introduction
- Callout boxes for the single most important insight per section
- Numbers are specific: not "billions of parameters" but "175 billion"
- Code blocks have two types of comments: standard green `# section header` and blue `# ↑ layman explanation`
- Deep-dive links include the author and year so readers can assess credibility
- The summary is a bullet list of one-sentence takeaways, all with **bold** key term

---

*Last updated: 2026-03-01. Update this file after each chapter is written with notes on voice, style decisions, and what worked.*
