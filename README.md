# SpiralEntropy Runner (sr_entropy_runner)

A lightweight, provider-agnostic post-hoc analysis runner for LLM generations.

**SpiralEntropy** measures *structural* signals from the **generated text itself** (always),
and optionally overlays **token surprisal / logprobs** when available.  
It’s designed as an *observational instrument* for “phase changes” in generation
(e.g., drift, collapse, bloom-like transitions) rather than a safety or truth detector.

---

## What this is

- A single-file CLI that:
  1) generates text with OpenAI / Mistral / Gemini / local HF models  
  2) segments the output into meaningful blocks (punctuation, newlines, bullets, quotes, bracket closures)
  3) computes structural metrics over **the full output**
  4) optionally overlays **H2 (surprisal)** when logprobs exist
  5) outputs a JSON report + matplotlib plots

---

## Key ideas

### Structural-first (always works)
Even when you **cannot** access model internals or per-token logprobs, you can still measure:
- **H1**: unigram Shannon entropy of the token stream (global + per-segment)
- **Repetition**: n-gram recurrence (degeneracy / loops)
- **Drift**: segment-to-segment changes in normalized entropy

### Optional overlay (when logprobs exist)
If you have per-token logprobs:
- **H2**: average surprisal (bits/token)
- **H2 spikes**: sudden increases in rolling surprisal
- mapping spikes to segments (when alignment is reliable)

> Important: Shannon entropy / surprisal **does not measure truth**.  
> These are *distributional / structural* observables.

---

## Features

- **Rich segmentation**
  - sentence end punctuation
  - newline / blank line
  - bullet list starts
  - quote line starts (`>`)
  - bracket/paren closures (soft boundary)
- **Dual token streams**
  - `struct_tokens`: stable, human-friendly token stream for metrics/segmentation
  - `align_tokens`: alignment-oriented stream for mapping overlays
- **Bloom / phase labeling (heuristic)**
  - `Stable`, `Drift`, `Collapse`, `Bloom`
- **Plots**
  - segment curves with Bloom shading, phase boundaries, log(n_tok) bars
  - phase strip
  - phase space scatter (H1_norm vs repetition; marker=phase)
  - token-level surprisal + spike markers (if logprobs exist)
- **Dynamic sampling (optional)**
  - “diversity push” triggered by spike detectors
  - supports multiple signal sources:
    - `h2` (logprobs)
    - `teacher` (separate scorer)
    - `compressor` (compression proxy)
    - `text` (structural proxy)

---

## Installation

### Minimal
```bash
pip install httpx matplotlib
