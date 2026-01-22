#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sr_entropy_runner.py — SpiralEntropy Runner
(OpenAI / Mistral / Gemini / Local HF)
+ Rich segmentation (punct/newline/bullets/quotes/paren-close)
+ Structural metrics computed from FULL generated text (always)
+ H2 (logprobs) handled as OPTIONAL overlay:
    - token-level H2 spikes always computed if logprobs exist
    - segment-level H2 only if we can align provider tokens to text tokens (suffix-align)
+ Bloom detection:
    - drift spikes (segment H1_norm jumps)
    - H2 spikes mapped onto segments only when alignment exists
+ Visualization:
    - segments plot: Bloom shading + phase boundaries + log(n_tok) bars + curves
    - phase strip: boundaries + optional Bloom shading
    - phase space: H1_norm vs rep_ngram, marker=phase (no colors)
    - token H2: rolling surprisal + spikes (if logprobs exist)

Deps:
  pip install httpx matplotlib
Optional:
  pip install tiktoken
Local:
  pip install torch transformers

Env vars:
  OPENAI_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY

Examples:
  # OpenAI + logprobs → full overlay
  python sr_entropy_runner.py --provider openai --model gpt-4o \
    --prompt "途中で視点が変わる散文を書いて。箇条書きと引用も混ぜて。" \
    --want_logprobs --segment_mode rich --plot_dir ./plots --out entropy.json

  # Mistral: structural-only (H2 overlay usually unavailable)
  python sr_entropy_runner.py --provider mistral --model mistral-large-latest \
    --prompt "日記の終盤がだんだん雑になる現象について" \
    --segment_mode rich --plot_dir ./plots --out entropy.json

  # Local HF:
  python sr_entropy_runner.py --provider local --local_model /path/to/model \
    --prompt "Write a surreal poem." --want_logprobs --device mps \
    --segment_mode rich --plot_dir ./plots --out entropy.json
"""

from __future__ import annotations
import argparse, json, math, os, re, sys, time, statistics
import bisect
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except Exception as e:
    print("Missing dependency: httpx. pip install httpx", file=sys.stderr)
    raise

EPS = 1e-12
DEFAULT_ENDCHARS = "。！？.!?"


# =========================================================
# Tokenization (tiktoken optional; else regex)
# =========================================================
REGEX_TOKEN_PATTERN = r"\w+|[^\w\s]"
REGEX_STRUCT_PATTERN = r"\n\n|\n|\w+|[^\w\s]"

def try_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except Exception:
        return None

def tokenize_text(text: str, mode: str = "auto") -> Tuple[List[str], str]:
    """
    Alignment-oriented tokenization.

    Returns token_strs, tokenizer_name.
    If tiktoken is available, uses `decode_with_offsets()` to safely decode UTF-8 and return
    a 1ID=1 display token stream (may include "" tokens when byte-fallback crosses UTF-8 boundaries).
    """
    tiktoken = try_tiktoken()
    if mode in ("auto", "tiktoken") and tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        try:
            decoded, offsets = enc.decode_with_offsets(ids)
        except Exception:
            return [enc.decode(ids)], "tiktoken:cl100k_base"

        if decoded != text:
            toks = re.findall(REGEX_TOKEN_PATTERN, text, flags=re.UNICODE)
            return toks, "regex"

        toks: List[str] = []
        for i, s in enumerate(offsets):
            e = offsets[i + 1] if (i + 1) < len(offsets) else len(decoded)
            toks.append(decoded[int(s):int(e)])

        if len(toks) != len(ids):
            return [enc.decode(ids)], "tiktoken:cl100k_base"

        return toks, "tiktoken:cl100k_base"
    toks = re.findall(REGEX_TOKEN_PATTERN, text, flags=re.UNICODE)
    return toks, "regex"

def tokenize_text_struct(text: str, mode: str = "auto") -> Tuple[List[str], str]:
    """
    Structural tokenization for SR metrics/segmentation (human-friendly, stable counts).

    - If tiktoken is available, groups consecutive tokens that share the same char offset
      (eliminates "" tokens from UTF-8 byte fallback) while preserving whitespace/newlines.
    - Else falls back to a regex that preserves newline tokens.
    """
    tiktoken = try_tiktoken()
    if mode in ("auto", "tiktoken_grouped") and tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        try:
            decoded, offsets = enc.decode_with_offsets(ids)
        except Exception:
            return [enc.decode(ids)], "tiktoken_grouped:cl100k_base"
        if decoded != text:
            toks = re.findall(REGEX_STRUCT_PATTERN, text, flags=re.UNICODE)
            return toks, "regex_struct"

        toks: List[str] = []
        i = 0
        while i < len(offsets):
            s = int(offsets[i])
            j = i + 1
            while j < len(offsets) and int(offsets[j]) == s:
                j += 1
            e = int(offsets[j]) if j < len(offsets) else len(decoded)
            toks.append(decoded[s:e])
            i = j

        if "".join(toks) != text:
            # last-resort fallback
            toks = re.findall(REGEX_STRUCT_PATTERN, text, flags=re.UNICODE)
            return toks, "regex_struct"
        return toks, "tiktoken_grouped:cl100k_base"

    # regex fallback (keeps newline tokens for rich segmentation)
    toks = re.findall(REGEX_STRUCT_PATTERN, text, flags=re.UNICODE)
    return toks, "regex_struct"

# =========================================================
# Metrics
# =========================================================
def safe_log2(x: float) -> float:
    return math.log(max(x, EPS), 2.0)

def shannon_entropy_from_counts(counts: Dict[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * safe_log2(p)
    return H

def unigram_counts(tokens: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for t in tokens:
        d[t] = d.get(t, 0) + 1
    return d

def norm_entropy(H_bits: float, vocab_size: int) -> float:
    if vocab_size <= 1:
        return 0.0
    return H_bits / max(safe_log2(float(vocab_size)), EPS)

def ngram_recurrence(tokens: List[str], n: int = 3) -> float:
    """
    recurrence rate = (repeat occurrences beyond first) / total ngrams
    """
    if n <= 0:
        return 0.0
    total = max(len(tokens) - n + 1, 0)
    if total <= 0:
        return 0.0
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(total):
        g = tuple(tokens[i:i+n])
        counts[g] = counts.get(g, 0) + 1
    repeats = sum((c - 1) for c in counts.values() if c > 1)
    return repeats / max(total, 1)

def distinct_ngram_ratio(tokens: List[str], n: int) -> float:
    if n <= 0:
        return 0.0
    total = max(len(tokens) - n + 1, 0)
    if total <= 0:
        return 1.0
    seen: set = set()
    for i in range(total):
        seen.add(tuple(tokens[i:i+n]))
    return len(seen) / max(total, 1)

def repeat_ratio(items: List[str]) -> float:
    total = len(items)
    if total <= 0:
        return 0.0
    counts: Dict[str, int] = {}
    for s in items:
        counts[s] = counts.get(s, 0) + 1
    repeats = sum((c - 1) for c in counts.values() if c > 1)
    return repeats / max(total, 1)

def normalize_repeat_key(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[0-9０-９]+", "0", s)
    s = s.rstrip("。！？.!?、，,;:：；")
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def rolling_mean(x: List[float], w: int) -> List[float]:
    if w <= 1:
        return x[:]
    out = []
    s = 0.0
    q = []
    for v in x:
        q.append(v)
        s += v
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def mad_zscores(values: List[float]) -> List[float]:
    """
    Robust z using MAD:
      z = 0.6745 * (x - median) / MAD
    """
    if not values:
        return []
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev) if abs_dev else 0.0
    if mad <= EPS:
        return [0.0 for _ in values]
    return [0.6745 * (v - med) / mad for v in values]

def ln_to_bits_surprisal(lnp: float) -> float:
    return -lnp / math.log(2.0)


# =========================================================
# Segmentation
# =========================================================
_BULLETS = {"-", "*", "•", "・", "‣", "◦", "▶", "→", "—", "–", "▪", "◆", "◇", "□", "■"}
_CLOSE_BRACKETS = set(list(")]}）］】」』〉》〕〗〙〛"))

def _strip_invis(s: str) -> str:
    return s.replace("\u200b", "").replace("\ufeff", "")

def _looks_like_ordered_list_piece(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if re.match(r"^\d{1,3}[\.\)]$", s):
        return True
    if re.match(r"^\d{1,3}）$", s):
        return True
    if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫]$", s):
        return True
    if re.match(r"^[IVXLC]{1,6}\.$", s, flags=re.IGNORECASE):
        return True
    return False

def segment_indices_by_window(n_tokens: int, window_tokens: int = 64) -> List[Tuple[int, int]]:
    segs = []
    s = 0
    while s < n_tokens:
        e = min(n_tokens, s + max(1, window_tokens))
        segs.append((s, e))
        s = e
    return segs

def segment_indices_by_punct(token_strs: List[str], endchars: str = DEFAULT_ENDCHARS,
                             min_tokens: int = 4, max_tokens: Optional[int] = None) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    start = 0
    endset = set(endchars)

    def is_end(tok: str) -> bool:
        return any(ch in endset for ch in tok)

    for i, tok in enumerate(token_strs):
        seg_len = (i + 1) - start
        force_max = (max_tokens is not None and seg_len >= max_tokens)
        if is_end(tok) or force_max:
            if seg_len >= min_tokens:
                segs.append((start, i + 1))
                start = i + 1

    n = len(token_strs)
    if start < n:
        if (n - start) >= min_tokens:
            segs.append((start, n))
        elif segs:
            a, _b = segs[-1]
            segs[-1] = (a, n)
        else:
            segs = [(0, n)]
    return segs

def segment_indices_rich(token_strs: List[str],
                         endchars: str = DEFAULT_ENDCHARS,
                         min_tokens: int = 4,
                         max_tokens: Optional[int] = None,
                         newline_hard: bool = True,
                         bullet_hard: bool = True,
                         quote_hard: bool = True,
                         parenclose_soft: bool = True) -> List[Tuple[int, int]]:
    """
    Rich token-space segmentation:
      - sentence end chars
      - newline / blank line
      - bullet list starts at line-begin
      - quote line starts (">")
      - optional paren-close soft boundary
    """
    segs: List[Tuple[int, int]] = []
    start = 0
    n = len(token_strs)
    endset = set(endchars)

    line_start = True

    def token_has_end(tok: str) -> bool:
        return any(ch in endset for ch in tok)

    def token_has_newline(tok: str) -> bool:
        return "\n" in tok

    def token_is_blankline(tok: str) -> bool:
        return "\n\n" in tok

    def token_is_bullet(tok: str) -> bool:
        t = _strip_invis(tok).strip()
        if not t:
            return False
        if t in _BULLETS:
            return True
        if len(t) <= 3 and any(t.startswith(b) for b in _BULLETS):
            return True
        return False

    def token_is_quote(tok: str) -> bool:
        t = _strip_invis(tok).lstrip()
        return t.startswith(">") or t.startswith("＞")

    def closes_paren(tok: str) -> bool:
        return any(ch in _CLOSE_BRACKETS for ch in tok)

    i = 0
    while i < n:
        tok = token_strs[i]
        seg_len_if_include = (i + 1) - start
        force_max = (max_tokens is not None and seg_len_if_include >= max_tokens)

        saw_newline = token_has_newline(tok)
        blankline = token_is_blankline(tok)

        # cut BEFORE at line start (new block begins)
        cut_before = False
        if line_start and (i > start):
            if bullet_hard and token_is_bullet(tok):
                cut_before = True
            elif quote_hard and token_is_quote(tok):
                cut_before = True
            else:
                # ordered list (can be split)
                t0 = _strip_invis(tok).strip()
                if bullet_hard and _looks_like_ordered_list_piece(t0):
                    cut_before = True
                elif bullet_hard and re.match(r"^\d{1,3}$", t0) and (i + 1) < n:
                    t1 = _strip_invis(token_strs[i + 1]).strip()
                    if t1 in (".", ")", "）"):
                        cut_before = True

        if cut_before and (i - start) >= min_tokens:
            segs.append((start, i))
            start = i

        # cut AFTER events
        cut_after = False
        if token_has_end(tok):
            cut_after = True
        if newline_hard and blankline:
            cut_after = True
        if newline_hard and saw_newline and (i + 1 - start) >= min_tokens:
            cut_after = True
        if parenclose_soft and closes_paren(tok):
            nxt = token_strs[i + 1] if (i + 1) < n else ""
            if (i + 1) == n or token_has_newline(nxt) or token_has_end(nxt):
                cut_after = True
        if force_max:
            cut_after = True

        if cut_after:
            if (i + 1 - start) >= min_tokens:
                segs.append((start, i + 1))
                start = i + 1

        # update line_start
        if saw_newline:
            line_start = True
        else:
            if _strip_invis(tok).strip() != "":
                line_start = False

        i += 1

    # tail
    if start < n:
        if (n - start) >= min_tokens:
            segs.append((start, n))
        elif segs:
            a, _b = segs[-1]
            segs[-1] = (a, n)
        else:
            segs = [(0, n)]

    cleaned = [(s, e) for (s, e) in segs if e > s]
    return cleaned if cleaned else [(0, n)]


# =========================================================
# Providers
# =========================================================
@dataclass
class GenResult:
    text: str
    token_strs: Optional[List[str]] = None
    token_logprobs_ln: Optional[List[float]] = None
    meta: Dict[str, Any] = None

class Provider:
    name: str
    def generate(self, prompt: str, system: Optional[str], args: argparse.Namespace, *, strip_output: bool = True) -> GenResult:
        raise NotImplementedError

class OpenAIProvider(Provider):
    name = "openai"
    def generate(self, prompt: str, system: Optional[str], args: argparse.Namespace, *, strip_output: bool = True) -> GenResult:
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing (or pass --openai_api_key).")

        url = (args.openai_base_url.rstrip("/") if args.openai_base_url else "https://api.openai.com").rstrip("/") + "/v1/responses"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": args.model,
            "input": prompt,
            "max_output_tokens": args.max_out,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if system:
            payload["instructions"] = system
        if args.want_logprobs:
            payload["include"] = ["message.output_text.logprobs"]
            if args.top_logprobs and args.top_logprobs > 0:
                payload["top_logprobs"] = int(args.top_logprobs)

        with httpx.Client(timeout=args.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
            data = r.json()

        out_text_parts: List[str] = []
        toks: List[str] = []
        lps: List[float] = []

        for item in data.get("output", []) or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        out_text_parts.append(part.get("text", "") or "")
                        lp_list = part.get("logprobs", None)
                        if isinstance(lp_list, list) and lp_list:
                            for lp in lp_list:
                                if isinstance(lp, dict):
                                    tok = lp.get("token") or lp.get("text") or ""
                                    logp = lp.get("logprob")
                                    if tok is not None and logp is not None:
                                        toks.append(str(tok))
                                        lps.append(float(logp))

        text = "".join(out_text_parts)
        if strip_output:
            text = text.strip()
        return GenResult(text=text, token_strs=(toks or None), token_logprobs_ln=(lps or None), meta={"raw": data, "provider": "openai"})

class MistralProvider(Provider):
    name = "mistral"
    def generate(self, prompt: str, system: Optional[str], args: argparse.Namespace, *, strip_output: bool = True) -> GenResult:
        api_key = args.mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY missing (or pass --mistral_api_key).")

        url = (args.mistral_base_url.rstrip("/") if args.mistral_base_url else "https://api.mistral.ai").rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_out,
            "stream": False,
        }

        with httpx.Client(timeout=args.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"Mistral error {r.status_code}: {r.text}")
            data = r.json()

        text = ""
        try:
            ch0 = (data.get("choices") or [])[0]
            msg = ch0.get("message") or {}
            text = msg.get("content") or ""
        except Exception:
            text = ""

        if strip_output:
            text = text.strip()
        return GenResult(text=text, token_strs=None, token_logprobs_ln=None, meta={"raw": data, "provider": "mistral"})

class GeminiProvider(Provider):
    name = "gemini"
    def generate(self, prompt: str, system: Optional[str], args: argparse.Namespace, *, strip_output: bool = True) -> GenResult:
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing (or pass --gemini_api_key).")

        model = args.model
        base = args.gemini_base_url.rstrip("/") if args.gemini_base_url else "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"

        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": args.temperature,
                "topP": args.top_p,
                "maxOutputTokens": args.max_out,
            }
        }
        if getattr(args, "top_k", 0) and int(args.top_k) > 0:
            body["generationConfig"]["topK"] = int(args.top_k)
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        if args.want_logprobs:
            body["generationConfig"]["responseLogprobs"] = True
            if args.top_logprobs and args.top_logprobs > 0:
                body["generationConfig"]["logprobs"] = int(args.top_logprobs)

        with httpx.Client(timeout=args.timeout) as client:
            r = client.post(url, headers={"Content-Type": "application/json"}, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"Gemini error {r.status_code}: {r.text}")
            data = r.json()

        text_parts: List[str] = []
        toks: List[str] = []
        lps_ln: List[float] = []

        cands = data.get("candidates") or []
        if cands:
            cand0 = cands[0] or {}
            content = cand0.get("content") or {}
            for p in (content.get("parts") or []):
                if isinstance(p, dict) and "text" in p:
                    text_parts.append(p.get("text") or "")
            lpr = cand0.get("logprobsResult") or cand0.get("logprobs_result") or None
            if isinstance(lpr, dict):
                chosen = lpr.get("chosenCandidates") or lpr.get("chosen_candidates") or []
                for it in chosen:
                    if isinstance(it, dict):
                        tok = it.get("token")
                        lp = it.get("logProbability") or it.get("log_probability")
                        if tok is not None and lp is not None:
                            toks.append(str(tok))
                            lps_ln.append(float(lp))

        text = "".join(text_parts)
        if strip_output:
            text = text.strip()
        return GenResult(text=text, token_strs=(toks or None), token_logprobs_ln=(lps_ln or None), meta={"raw": data, "provider": "gemini"})

class LocalHFProvider(Provider):
    name = "local"
    def generate(self, prompt: str, system: Optional[str], args: argparse.Namespace, *, strip_output: bool = True) -> GenResult:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("Local provider requires torch + transformers. pip install torch transformers") from e

        model_path = args.local_model
        if not model_path:
            raise RuntimeError("For --provider local, set --local_model /path/or/hf_id")

        import torch
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        dtype = args.dtype.lower()
        torch_dtype = torch.float32
        if dtype in ("fp16", "float16"):
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype in ("fp32", "float32"):
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        model.eval()

        if args.device == "cuda" and torch.cuda.is_available():
            model.to("cuda")
        elif args.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model.to("mps")
        else:
            model.to("cpu")

        if system and hasattr(tok, "apply_chat_template"):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            in_ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        else:
            in_ids = tok(prompt, return_tensors="pt").input_ids

        in_ids = in_ids.to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_out,
            do_sample=(args.temperature > 0),
            temperature=max(args.temperature, 1e-6),
            top_k=(int(args.top_k) if getattr(args, "top_k", 0) and int(args.top_k) > 0 else 0),
            top_p=args.top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=args.want_logprobs,
        )

        with torch.no_grad():
            out = model.generate(in_ids, **gen_kwargs)

        seq = out.sequences[0].tolist()
        gen_ids = seq[len(in_ids[0]):]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        if strip_output:
            text = text.strip()
        token_strs = [tok.decode([tid]) for tid in gen_ids]

        lps_ln: Optional[List[float]] = None
        if args.want_logprobs and hasattr(out, "scores") and out.scores is not None:
            lps = []
            for step_logits, chosen_id in zip(out.scores, gen_ids):
                logits = step_logits[0]
                logp = torch.log_softmax(logits, dim=-1)[int(chosen_id)].item()
                lps.append(float(logp))
            lps_ln = lps

        return GenResult(text=text, token_strs=(token_strs or None), token_logprobs_ln=lps_ln, meta={"provider": "local", "model_path": model_path})


# =========================================================
# Dynamic sampling: H2-spike-triggered "diversity push"
# =========================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    d = vars(args).copy()
    d.update(overrides)
    return argparse.Namespace(**d)

def _robust_z_against_recent(value: float, history: List[float], window: int) -> float:
    """
    Robust z-score (MAD) of `value` against a trailing window of `history` (excluding the current value).
    """
    if window <= 1:
        return 0.0
    if len(history) < 4:
        return 0.0
    start = max(0, len(history) - window - 1)
    ref = history[start:-1]  # exclude current
    if len(ref) < 3:
        return 0.0
    med = statistics.median(ref)
    mad = statistics.median([abs(v - med) for v in ref]) if ref else 0.0
    if mad <= EPS:
        return 0.0
    return 0.6745 * (value - med) / mad

class H2SpikeDetector:
    def __init__(self, *, h2_win: int, z_window: int, spike_z: float, slope_window: int, slope_z: float, min_tokens: int):
        self.h2_win = max(1, int(h2_win))
        self.z_window = max(1, int(z_window))
        self.spike_z = float(spike_z)
        self.slope_window = max(1, int(slope_window))
        self.slope_z = float(slope_z)
        self.min_tokens = max(0, int(min_tokens))

        self._surp = deque()
        self._surp_sum = 0.0
        self._rm_prev: Optional[float] = None

        self.pos_jumps: List[float] = []
        self.rm: List[float] = []
        self.jump_z: List[float] = []
        self.spikes: List[int] = []
        self.pos_slopes: List[float] = []
        self.slope_zs: List[float] = []
        self.slope_events: List[int] = []

    def update(self, logprob_ln: float) -> Dict[str, Any]:
        i = len(self.pos_jumps)
        bits = ln_to_bits_surprisal(float(logprob_ln))

        if len(self._surp) >= self.h2_win:
            self._surp_sum -= float(self._surp.popleft())
        self._surp.append(bits)
        self._surp_sum += bits

        rm = self._surp_sum / max(len(self._surp), 1)
        prev = rm if self._rm_prev is None else self._rm_prev
        self._rm_prev = rm
        self.rm.append(rm)

        jump = rm - prev
        pos = max(0.0, float(jump))
        self.pos_jumps.append(pos)
        z = _robust_z_against_recent(pos, self.pos_jumps, window=self.z_window)
        self.jump_z.append(z)

        is_spike = (i >= self.min_tokens) and (pos > 0.0) and (z >= self.spike_z)
        if is_spike:
            self.spikes.append(i)

        slope = 0.0
        if len(self.rm) > self.slope_window:
            slope = (self.rm[-1] - self.rm[-1 - self.slope_window]) / float(self.slope_window)
        pos_slope = max(0.0, float(slope))
        self.pos_slopes.append(pos_slope)
        slope_z = _robust_z_against_recent(pos_slope, self.pos_slopes, window=self.z_window)
        self.slope_zs.append(slope_z)

        is_slope = (i >= self.min_tokens) and (pos_slope > 0.0) and (slope_z >= self.slope_z)
        if is_slope:
            self.slope_events.append(i)

        return {
            "i": i,
            "bits": bits,
            "rm": rm,
            "pos_jump": pos,
            "z": z,
            "is_spike": is_spike,
            "slope": float(slope),
            "pos_slope": float(pos_slope),
            "slope_z": float(slope_z),
            "is_slope": is_slope,
        }

def _dynamic_strength(z: float, z0: float, z1: float) -> float:
    if z1 <= z0 + EPS:
        return 1.0 if z >= z0 else 0.0
    return _clamp((z - z0) / (z1 - z0), 0.0, 1.0)

def _compute_dynamic_controls(*,
                              base_temperature: float,
                              base_top_p: float,
                              base_top_k: int,
                              z: float,
                              args: argparse.Namespace) -> Tuple[Dict[str, Any], float]:
    """
    Returns (controls_dict, strength) where strength ∈ [0,1].
    """
    z0 = float(args.dyn_spike_z)
    z1 = float(args.dyn_full_z)
    strength = _dynamic_strength(float(z), z0, z1)

    temperature = float(base_temperature) + strength * float(args.dyn_temp_boost)
    top_p = float(base_top_p) + strength * float(args.dyn_top_p_boost)
    top_p = _clamp(top_p, 0.0, float(args.dyn_top_p_cap))

    top_k = int(base_top_k)
    if top_k > 0 and int(args.dyn_top_k_cap) > top_k:
        widened = int(round(top_k + strength * (int(args.dyn_top_k_cap) - top_k)))
        top_k = max(0, widened)
    if top_k > 0 and float(z) >= float(args.dyn_top_k_disable_z):
        top_k = 0

    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}, strength

def _build_continuation_prompt(prompt: str, generated_so_far: str, *, tail_chars: int) -> str:
    tail = generated_so_far if tail_chars <= 0 else generated_so_far[-tail_chars:]
    return (
        f"{prompt}\n\n"
        "[Instruction]\n"
        "Continue the assistant response from exactly where it ended. "
        "Do NOT repeat or paraphrase earlier text. Output only the continuation.\n\n"
        "[Previous output (tail)]\n"
        f"{tail}"
    )

def _sample_local_next_token(*,
                             logits,
                             temperature: float,
                             top_p: float,
                             top_k: int):
    """
    Returns (token_id:int, logprob_ln:float) where logprob_ln is under the sampling distribution.
    """
    import torch

    vocab = int(logits.shape[-1])
    if temperature <= 0:
        # greedy
        token_id = int(torch.argmax(logits).item())
        logp = float(torch.log_softmax(logits, dim=-1)[token_id].item())
        return token_id, logp

    t = max(float(temperature), 1e-6)
    scores = logits / t

    tk = int(top_k)
    if tk > 0:
        tk = min(tk, vocab)
        v, _ = torch.topk(scores, k=tk)
        kth = v[-1]
        scores = torch.where(scores < kth, torch.full_like(scores, float("-inf")), scores)

    tp = float(top_p)
    if 0.0 < tp < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > tp
        if mask.numel() > 0:
            mask[0] = False  # keep at least 1
        sorted_scores = sorted_scores.masked_fill(mask, float("-inf"))
        # scatter back
        scores2 = torch.full_like(scores, float("-inf"))
        scores2[sorted_idx] = sorted_scores
        scores = scores2

    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        probs = torch.softmax(logits / t, dim=-1)

    token_id = int(torch.multinomial(probs, num_samples=1).item())
    logp = float(torch.log(probs[token_id]).item())
    return token_id, logp

def _generate_local_dynamic(prompt: str, system: Optional[str], args: argparse.Namespace) -> GenResult:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("Local provider requires torch + transformers. pip install torch transformers") from e

    model_path = args.local_model
    if not model_path:
        raise RuntimeError("For --provider local, set --local_model /path/or/hf_id")

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    dtype = args.dtype.lower()
    torch_dtype = torch.float32
    if dtype in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif dtype in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    elif dtype in ("fp32", "float32"):
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model.eval()

    if args.device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    elif args.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")

    if system and hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        in_ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    else:
        in_ids = tok(prompt, return_tensors="pt").input_ids

    in_ids = in_ids.to(model.device)
    eos_id = tok.eos_token_id

    base_temperature = float(args.temperature)
    base_top_p = float(args.top_p)
    base_top_k = int(args.top_k)

    target_budget = int(args.max_out)
    max_budget_on_spike = int(args.dyn_max_out_on_spike) if int(args.dyn_max_out_on_spike) > 0 else 0

    detector = H2SpikeDetector(
        h2_win=int(args.h2_win),
        z_window=int(args.dyn_z_window),
        spike_z=float(args.dyn_spike_z),
        slope_window=int(args.dyn_slope_window),
        slope_z=float(args.dyn_slope_z),
        min_tokens=max(int(args.h2_win), int(args.dyn_slope_window), 8),
    )

    boost_left = 0
    boost_z = 0.0
    events: List[Dict[str, Any]] = []

    gen_ids: List[int] = []
    token_strs: List[str] = []
    logprobs_ln: List[float] = []

    with torch.no_grad():
        out0 = model(input_ids=in_ids, use_cache=True)
    past = out0.past_key_values
    next_logits = out0.logits[0, -1, :]

    step = 0
    while step < target_budget:
        # controls for THIS token
        if boost_left > 0:
            controls, _s = _compute_dynamic_controls(
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                base_top_k=base_top_k,
                z=boost_z,
                args=args,
            )
        else:
            controls = {"temperature": base_temperature, "top_p": base_top_p, "top_k": base_top_k}

        tid, lp_ln = _sample_local_next_token(
            logits=next_logits,
            temperature=float(controls["temperature"]),
            top_p=float(controls["top_p"]),
            top_k=int(controls["top_k"]),
        )
        gen_ids.append(int(tid))
        token_strs.append(tok.decode([int(tid)]))
        logprobs_ln.append(float(lp_ln))

        if boost_left > 0:
            boost_left -= 1

        info = detector.update(float(lp_ln))
        is_spike = bool(info.get("is_spike"))
        is_slope = (not bool(args.dyn_no_slope)) and bool(info.get("is_slope"))
        if is_spike or is_slope:
            z = max(float(info.get("z", 0.0)) if is_spike else 0.0,
                    float(info.get("slope_z", 0.0)) if is_slope else 0.0)
            kind = "spike" if is_spike else "slope"
            if is_spike and is_slope:
                kind = "spike+slope"
            applied, strength = _compute_dynamic_controls(
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                base_top_k=base_top_k,
                z=z,
                args=args,
            )
            boost_left = max(boost_left, int(args.dyn_boost_tokens))
            boost_z = max(boost_z, z)

            if max_budget_on_spike and max_budget_on_spike > target_budget:
                target_budget = max_budget_on_spike

            ev = {
                "kind": kind,
                "token_i": int(info["i"]),
                "z": z,
                "strength": float(strength),
                "pos_jump": float(info["pos_jump"]),
                "slope": float(info.get("slope", 0.0)),
                "pos_slope": float(info.get("pos_slope", 0.0)),
                "slope_z": float(info.get("slope_z", 0.0)),
                "rm": float(info["rm"]),
                "applied_next": applied,
                "new_target_max_out": int(target_budget),
            }
            events.append(ev)
            if getattr(args, "dyn_debug", False):
                print(f"[dyn] {ev['kind']}@{ev['token_i']} z={ev['z']:.2f} -> next={ev['applied_next']} max_out={target_budget}", file=sys.stderr)

        if eos_id is not None and int(tid) == int(eos_id):
            break

        with torch.no_grad():
            out = model(input_ids=torch.tensor([[int(tid)]], device=model.device), use_cache=True, past_key_values=past)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]
        step += 1

    text = tok.decode(gen_ids, skip_special_tokens=True)
    if getattr(args, "strip_generated", True):
        text = text.strip()

    meta = {
        "provider": "local",
        "model_path": model_path,
        "dynamic_sampling": {
            "enabled": True,
            "mode": "h2",
            "step_unit": "gen_token",
            "measure_unit": "token",
            "budget_unit": "gen_token",
            "events": events,
            "detector": {
                "h2_win": int(args.h2_win),
                "dyn_z_window": int(args.dyn_z_window),
                "dyn_spike_z": float(args.dyn_spike_z),
                "dyn_full_z": float(args.dyn_full_z),
                "dyn_no_slope": bool(args.dyn_no_slope),
                "dyn_slope_window": int(args.dyn_slope_window),
                "dyn_slope_z": float(args.dyn_slope_z),
            },
        },
    }
    return GenResult(text=text, token_strs=(token_strs or None), token_logprobs_ln=(logprobs_ln or None), meta=meta)


def _bits_to_pseudo_logprob_ln(bits: float) -> float:
    return -float(bits) * math.log(2.0)

def _build_score_context(prompt: str, system: Optional[str], generated_so_far: str, *, tail_chars: int) -> str:
    tail = generated_so_far if tail_chars <= 0 else generated_so_far[-tail_chars:]
    if system:
        return f"{system}\n\n{prompt}\n\n{tail}"
    return f"{prompt}\n\n{tail}"

class Scorer:
    name: str
    def score(self, context: str, continuation: str) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

class LocalHFScorer(Scorer):
    name = "local"
    def __init__(self, *, model_path: str, device: str, dtype: str, stride_tokens: int):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("Teacher scoring (--score_provider local) requires torch + transformers.") from e

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        dt = dtype.lower()
        torch_dtype = torch.float32
        if dt in ("fp16", "float16"):
            torch_dtype = torch.float16
        elif dt in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dt in ("fp32", "float32"):
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        model.eval()

        if device == "cuda" and torch.cuda.is_available():
            model.to("cuda")
        elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model.to("mps")
        else:
            model.to("cpu")

        self._model = model
        self._model_path = model_path
        self._stride_tokens = max(1, int(stride_tokens))

    @property
    def model_path(self) -> str:
        return self._model_path

    def token_char_offsets(self, text: str) -> Optional[List[Tuple[int, int]]]:
        tok = self._tok
        try:
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return None
        offs = enc.get("offset_mapping", None)
        if offs is None:
            return None
        out: List[Tuple[int, int]] = []
        for o in offs:
            if o is None or len(o) != 2:
                return None
            out.append((int(o[0]), int(o[1])))
        return out

    def score(self, context: str, continuation: str) -> Tuple[List[str], List[float]]:
        if not continuation:
            return [], []

        tok = self._tok
        torch = self._torch
        model = self._model

        ctx_ids = tok(context, add_special_tokens=False, return_tensors="pt").input_ids[0]
        cont_ids_full = tok(continuation, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if int(cont_ids_full.shape[0]) == 0:
            return [], []

        ctx_len = int(ctx_ids.shape[0])
        cont_len = int(cont_ids_full.shape[0])

        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if isinstance(max_len, int) and max_len > 0:
            overflow = (ctx_len + cont_len) - int(max_len)
            if overflow > 0:
                if overflow < ctx_len:
                    ctx_ids = ctx_ids[overflow:]
                else:
                    ctx_ids = ctx_ids[:0]
                    drop = overflow - ctx_len
                    if drop >= cont_len:
                        return [], []
                    cont_ids_full = cont_ids_full[drop:]

        device = model.device
        past = None
        next_logits = None
        stride = int(self._stride_tokens)

        if int(ctx_ids.shape[0]) > 0:
            for s in range(0, int(ctx_ids.shape[0]), stride):
                block = ctx_ids[s:s + stride].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(input_ids=block, use_cache=True, past_key_values=past)
                past = out.past_key_values
                next_logits = out.logits[0, -1, :]

        if next_logits is None:
            bos = tok.bos_token_id
            if bos is None:
                bos = getattr(getattr(model, "config", None), "bos_token_id", None)
            if bos is None:
                bos = tok.eos_token_id
            if bos is None:
                bos = getattr(getattr(model, "config", None), "eos_token_id", None)
            if bos is None:
                return [], []
            with torch.no_grad():
                out0 = model(input_ids=torch.tensor([[int(bos)]], device=device), use_cache=True)
            past = out0.past_key_values
            next_logits = out0.logits[0, -1, :]

        out_tokens: List[str] = []
        out_lps: List[float] = []

        import torch.nn.functional as F

        cont_ids = cont_ids_full.to(device)
        prev_next_logits = next_logits
        for s in range(0, int(cont_ids.shape[0]), stride):
            block = cont_ids[s:s + stride]
            if int(block.shape[0]) == 0:
                continue

            with torch.no_grad():
                out = model(input_ids=block.unsqueeze(0), use_cache=True, past_key_values=past)
            past = out.past_key_values
            logits = out.logits[0]  # [L, V]

            token_ids = [int(x) for x in block.tolist()]
            token_strs = [tok.decode([tid]) for tid in token_ids]

            lps_block: List[float] = []
            lp0 = (F.log_softmax(prev_next_logits.float(), dim=-1)[token_ids[0]].item())
            lps_block.append(float(lp0))

            if len(token_ids) > 1:
                pred = logits[:-1, :]
                pred_f = pred.float()
                logZ = torch.logsumexp(pred_f, dim=-1)  # [L-1]
                idx = torch.tensor(token_ids[1:], device=device, dtype=torch.long).unsqueeze(-1)  # [L-1, 1]
                sel = torch.gather(pred_f, dim=-1, index=idx).squeeze(-1)  # [L-1]
                lps_rest = (sel - logZ).tolist()
                lps_block.extend([float(x) for x in lps_rest])

            out_tokens.extend(token_strs)
            out_lps.extend(lps_block)
            prev_next_logits = logits[-1, :]

        return out_tokens, out_lps

class OpenAICompletionsScorer(Scorer):
    """
    Teacher scoring using OpenAI /v1/completions with echo=True.
    Note: Not all chat models support this endpoint; pick an instruct/completions-capable model.
    """
    name = "openai"
    def __init__(self, *, api_key: str, base_url: Optional[str], model: str, timeout: float, retries: int, retry_backoff: float):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing (or pass --openai_api_key).")
        if not model:
            raise RuntimeError("--score_model is required for --score_provider openai.")
        self._api_key = api_key
        self._url = (base_url.rstrip("/") if base_url else "https://api.openai.com").rstrip("/") + "/v1/completions"
        self._model = model
        self._timeout = float(timeout)
        self._retries = max(0, int(retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._last_offsets: Optional[List[Tuple[int, int]]] = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_offsets(self) -> Optional[List[Tuple[int, int]]]:
        return self._last_offsets

    def score(self, context: str, continuation: str) -> Tuple[List[str], List[float]]:
        if not continuation:
            return [], []

        prompt = context + continuation
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        data = None
        last_err = None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    r = client.post(self._url, headers=headers, json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"OpenAI scoring transient error {r.status_code}: {r.text}")
                elif r.status_code >= 400:
                    msg = r.text
                    hint = ""
                    if r.status_code == 400:
                        hint = "\nHint: /v1/completions + echo/logprobs requires a completions-capable model. Try e.g. davinci-002 / babbage-002 / gpt-3.5-turbo-instruct."
                    raise RuntimeError(f"OpenAI scoring error {r.status_code}: {msg}{hint}")
                else:
                    data = r.json()
                    last_err = None
                    break
            except Exception as e:
                last_err = e

            if attempt < self._retries and self._retry_backoff > 0:
                time.sleep(self._retry_backoff * (2 ** attempt))

        if data is None:
            raise RuntimeError(f"OpenAI scoring failed after {self._retries + 1} attempt(s): {last_err}") from last_err

        choices = data.get("choices") or []
        if not choices:
            return [], []
        lp = (choices[0] or {}).get("logprobs") or {}
        toks = lp.get("tokens") or []
        lps = lp.get("token_logprobs") or []
        offs = lp.get("text_offset") or []
        if not (isinstance(toks, list) and isinstance(lps, list) and isinstance(offs, list)):
            return [], []

        out_tokens: List[str] = []
        out_lps: List[float] = []
        out_offs: List[Tuple[int, int]] = []
        cut = len(context)
        for tok, logp, off in zip(toks, lps, offs):
            if off is None:
                continue
            if int(off) < cut:
                continue
            if logp is None:
                continue
            out_tokens.append(str(tok))
            out_lps.append(float(logp))
            s = max(0, int(off) - cut)
            out_offs.append((s, s + len(str(tok))))

        self._last_offsets = out_offs
        return out_tokens, out_lps

class GeminiScorer(Scorer):
    """
    Teacher scoring using Gemini generateContent.

    This relies on the API returning logprobs for the *prompt tokens* when maxOutputTokens=0.
    If the API only returns logprobs for generated output tokens, this scorer cannot work
    for true scoring and will raise an error.
    """
    name = "gemini"
    def __init__(self, *, api_key: str, base_url: Optional[str], model: str, timeout: float, retries: int, retry_backoff: float):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing (or pass --gemini_api_key / --score_gemini_api_key).")
        if not model:
            raise RuntimeError("--score_model is required for --score_provider gemini.")
        self._api_key = api_key
        base = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com"
        self._url = f"{base}/v1beta/models/{model}:generateContent?key={api_key}"
        self._model = model
        self._timeout = float(timeout)
        self._retries = max(0, int(retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._last_offsets: Optional[List[Tuple[int, int]]] = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_offsets(self) -> Optional[List[Tuple[int, int]]]:
        return self._last_offsets

    def score(self, context: str, continuation: str) -> Tuple[List[str], List[float]]:
        if not continuation:
            return [], []

        prompt = context + continuation
        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": 0,
                "responseLogprobs": True,
            },
        }

        data = None
        last_err = None
        for attempt in range(self._retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    r = client.post(self._url, headers={"Content-Type": "application/json"}, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"Gemini scoring transient error {r.status_code}: {r.text}")
                elif r.status_code >= 400:
                    raise RuntimeError(f"Gemini scoring error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    last_err = None
                    break
            except Exception as e:
                last_err = e
            if attempt < self._retries and self._retry_backoff > 0:
                time.sleep(self._retry_backoff * (2 ** attempt))

        if data is None:
            raise RuntimeError(f"Gemini scoring failed after {self._retries + 1} attempt(s): {last_err}") from last_err

        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError("Gemini scoring returned no candidates.")
        cand0 = cands[0] or {}
        lpr = cand0.get("logprobsResult") or cand0.get("logprobs_result") or None
        if not isinstance(lpr, dict):
            raise RuntimeError("Gemini scoring missing logprobsResult; cannot score.")

        chosen = lpr.get("chosenCandidates") or lpr.get("chosen_candidates") or []
        if not isinstance(chosen, list) or not chosen:
            raise RuntimeError("Gemini scoring returned empty chosenCandidates; likely does not provide prompt logprobs with maxOutputTokens=0.")

        toks: List[str] = []
        lps_ln: List[float] = []
        for it in chosen:
            if not isinstance(it, dict):
                continue
            tok = it.get("token")
            lp = it.get("logProbability") or it.get("log_probability")
            if tok is None or lp is None:
                continue
            toks.append(str(tok))
            lps_ln.append(float(lp))

        # Align to full prompt, then cut to continuation by char offset.
        ca = align_measure_series_to_text(prompt, toks, lps_ln, [])
        if ca is None:
            raise RuntimeError("Gemini scoring: could not align token series to prompt.")

        cut = len(context)
        out_tokens: List[str] = []
        out_lps: List[float] = []
        out_offs: List[Tuple[int, int]] = []
        for tok, lp, (a, b) in zip(ca.tokens, ca.logprobs_ln, ca.char_offsets):
            if int(a) < cut:
                continue
            out_tokens.append(str(tok))
            out_lps.append(float(lp))
            out_offs.append((int(a) - cut, int(b) - cut))

        self._last_offsets = out_offs if len(out_offs) == len(out_tokens) else None
        return out_tokens, out_lps

def get_scorer(args: argparse.Namespace) -> Scorer:
    prov = (getattr(args, "score_provider", None) or "").lower()
    if not prov:
        raise RuntimeError("--score_provider is required for --dyn_mode teacher.")
    if prov in ("local", "hf"):
        model_path = getattr(args, "score_local_model", None) or getattr(args, "local_model", None)
        if not model_path:
            raise RuntimeError("--score_local_model is required for --score_provider local (teacher).")
        return LocalHFScorer(
            model_path=str(model_path),
            device=str(getattr(args, "score_device", "cpu")),
            dtype=str(getattr(args, "score_dtype", "fp16")),
            stride_tokens=int(getattr(args, "score_stride_tokens", 64)),
        )
    if prov == "openai":
        api_key = getattr(args, "score_openai_api_key", None) or getattr(args, "openai_api_key", None) or os.getenv("OPENAI_API_KEY", "")
        base_url = getattr(args, "score_openai_base_url", None) or getattr(args, "openai_base_url", None)
        return OpenAICompletionsScorer(
            api_key=str(api_key or ""),
            base_url=base_url,
            model=str(getattr(args, "score_model", None) or ""),
            timeout=float(getattr(args, "timeout", 60.0)),
            retries=int(getattr(args, "score_retries", 6)),
            retry_backoff=float(getattr(args, "score_retry_backoff", 0.6)),
        )
    if prov == "gemini":
        api_key = getattr(args, "score_gemini_api_key", None) or getattr(args, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY", "")
        base_url = getattr(args, "score_gemini_base_url", None) or getattr(args, "gemini_base_url", None)
        return GeminiScorer(
            api_key=str(api_key or ""),
            base_url=base_url,
            model=str(getattr(args, "score_model", None) or ""),
            timeout=float(getattr(args, "timeout", 60.0)),
            retries=int(getattr(args, "score_retries", 6)),
            retry_backoff=float(getattr(args, "score_retry_backoff", 0.6)),
        )
    raise RuntimeError(f"Unsupported score_provider: {prov}")

def _compressor_proxy_bits(text: str, *, algo: str, level: int, window_chars: int, invert: bool) -> float:
    window = text if window_chars <= 0 else text[-window_chars:]
    raw = window.encode("utf-8", errors="replace")
    if not raw:
        return 0.0

    a = algo.lower()
    lvl = int(level)
    if a == "zlib":
        import zlib
        lvl = max(0, min(lvl, 9))
        comp = zlib.compress(raw, level=lvl)
    elif a == "lzma":
        import lzma
        lvl = max(0, min(lvl, 9))
        comp = lzma.compress(raw, preset=lvl)
    else:
        raise ValueError(f"unknown comp_algo: {algo}")

    bits_per_byte = 8.0 * (len(comp) / max(len(raw), 1))
    if invert:
        return max(0.0, 8.0 - bits_per_byte)
    return float(bits_per_byte)

def _text_collapse_proxy_bits(tokens: List[str],
                              args: argparse.Namespace,
                              *,
                              text_tail: str,
                              punct_density: float,
                              newline_density: float,
                              density_jump: float) -> Dict[str, float]:
    win = int(getattr(args, "dyn_text_window_tokens", 256))
    t = tokens[-win:] if win > 0 else tokens

    c = unigram_counts(t) if t else {}
    V = float(len(c))
    H1 = shannon_entropy_from_counts(c) if c else 0.0
    H1n = norm_entropy(H1, max(len(c), 2)) if c else 0.0

    rep_ns = (3, 4, 5)
    rep_vals = [ngram_recurrence(t, n=n) for n in rep_ns] if t else [0.0, 0.0, 0.0]
    dist_vals = [distinct_ngram_ratio(t, n=n) for n in rep_ns] if t else [1.0, 1.0, 1.0]
    rep_mix = float(sum(rep_vals) / max(len(rep_vals), 1))
    dist_mix = float(sum(dist_vals) / max(len(dist_vals), 1))

    lines_raw = [ln.strip() for ln in (text_tail or "").splitlines() if ln.strip()]
    lines = [normalize_repeat_key(ln) for ln in lines_raw]
    lines = [ln for ln in lines if ln]
    line_rep = repeat_ratio(lines) if lines else 0.0
    sents_raw = [s.strip() for s in re.split(r"[。！？.!?]+", (text_tail or "")) if s.strip()]
    sents = [normalize_repeat_key(s) for s in sents_raw]
    sents = [s for s in sents if s]
    sent_rep = repeat_ratio(sents) if sents else 0.0
    loop_rep = float(max(line_rep, sent_rep))

    rep_hi = float(getattr(args, "dyn_text_rep_hi", None) if getattr(args, "dyn_text_rep_hi", None) is not None else args.rep_hi)
    h1_low = float(getattr(args, "dyn_text_h1_low", None) if getattr(args, "dyn_text_h1_low", None) is not None else args.h1_low)
    distinct_low = float(getattr(args, "dyn_text_distinct_low", 0.75))
    line_rep_hi = float(getattr(args, "dyn_text_line_rep_hi", 0.20))
    density_jump_hi = float(getattr(args, "dyn_text_density_jump_hi", 0.02))

    w_rep = float(getattr(args, "dyn_text_w_rep", 1.0))
    w_h1 = float(getattr(args, "dyn_text_w_h1", 1.0))
    w_dist = float(getattr(args, "dyn_text_w_distinct", 1.0))
    w_line = float(getattr(args, "dyn_text_w_line_rep", 1.0))
    w_jump = float(getattr(args, "dyn_text_w_density_jump", 0.5))

    rep_excess = max(0.0, rep_mix - rep_hi)
    h1_def = max(0.0, h1_low - float(H1n))
    distinct_def = max(0.0, distinct_low - dist_mix)
    line_rep_ex = max(0.0, loop_rep - line_rep_hi)
    density_ex = max(0.0, float(density_jump) - density_jump_hi)

    bits = (w_rep * rep_excess) + (w_h1 * h1_def) + (w_dist * distinct_def) + (w_line * line_rep_ex) + (w_jump * density_ex)
    return {
        "bits": float(bits),
        "H1_norm": float(H1n),
        "V": float(V),
        "rep_mix": float(rep_mix),
        "distinct_mix": float(dist_mix),
        "rep_3": float(rep_vals[0]),
        "rep_4": float(rep_vals[1]),
        "rep_5": float(rep_vals[2]),
        "distinct_3": float(dist_vals[0]),
        "distinct_4": float(dist_vals[1]),
        "distinct_5": float(dist_vals[2]),
        "line_rep": float(line_rep),
        "sent_rep": float(sent_rep),
        "loop_rep": float(loop_rep),
        "rep_excess": float(rep_excess),
        "h1_deficit": float(h1_def),
        "distinct_deficit": float(distinct_def),
        "line_rep_excess": float(line_rep_ex),
        "punct_density": float(punct_density),
        "newline_density": float(newline_density),
        "density_jump": float(density_jump),
        "density_excess": float(density_ex),
    }

def _generate_remote_dynamic(prov: Provider, prompt: str, system: Optional[str], args: argparse.Namespace) -> GenResult:
    mode = str(getattr(args, "dyn_mode", "h2")).lower()
    if mode not in ("h2", "teacher", "compressor", "text"):
        raise RuntimeError(f"unknown dyn_mode: {mode}")

    base_temperature = float(args.temperature)
    base_top_p = float(args.top_p)
    base_top_k = int(args.top_k)

    target_budget = int(args.max_out)
    max_budget_on_event = int(args.dyn_max_out_on_spike) if int(args.dyn_max_out_on_spike) > 0 else 0

    detector = H2SpikeDetector(
        h2_win=int(args.h2_win),
        z_window=int(args.dyn_z_window),
        spike_z=float(args.dyn_spike_z),
        slope_window=int(args.dyn_slope_window),
        slope_z=float(args.dyn_slope_z),
        min_tokens=max(int(args.h2_win), int(args.dyn_slope_window), 8),
    )

    scorer: Optional[Scorer] = None
    if mode == "teacher":
        scorer = get_scorer(args)

    teacher_cov_num = 0
    teacher_cov_den = 0
    teacher_budget_tokens = 0
    teacher_full_i = 0
    score_sample_every = int(getattr(args, "score_sample_every", 1))
    if score_sample_every <= 0:
        score_sample_every = 1
    proxy_ema_bits: Optional[float] = None
    proxy_ema_alpha = float(getattr(args, "dyn_proxy_ema_alpha", 0.35))

    def smooth_proxy_bits(bits_raw: float) -> Tuple[float, float]:
        nonlocal proxy_ema_bits
        a = float(proxy_ema_alpha)
        x = float(bits_raw)
        if a <= 0.0:
            return x, x
        if proxy_ema_bits is None:
            proxy_ema_bits = x
        else:
            proxy_ema_bits = a * x + (1.0 - a) * float(proxy_ema_bits)
        return float(proxy_ema_bits), x

    comp_tail = ""
    dyn_text_tokens: List[str] = []
    dyn_text_tail = ""
    prev_punct_density: Optional[float] = None
    prev_newline_density: Optional[float] = None
    was_in_collapse = False

    boost_left = 0  # measured in (rough) text tokens for remote chunk mode
    boost_z = 0.0
    events: List[Dict[str, Any]] = []

    out_text_parts: List[str] = []
    meas_tokens: List[str] = []
    meas_logprobs: List[float] = []
    meas_char_offsets: List[Tuple[int, int]] = []
    raw_calls: List[Any] = []

    gen_text_tokens = 0  # regex token proxy

    def progress_tokens() -> int:
        if mode == "h2":
            return len(meas_tokens)
        if mode == "teacher":
            return int(teacher_budget_tokens)
        return gen_text_tokens

    if mode == "h2":
        budget_unit = "gen_token"
        measure_unit = "provider_token"
    elif mode == "teacher":
        budget_unit = "teacher_token"
        measure_unit = "teacher_token"
    else:
        budget_unit = "gen_regex_token"
        measure_unit = "proxy_step"
    step_unit = measure_unit

    chunk_max = int(args.dyn_chunk_tokens)
    if chunk_max <= 0:
        raise RuntimeError("--dyn_chunk_tokens must be > 0")

    # Keep generator chunk budget fixed; stop condition uses progress_tokens() to avoid unit mismatches.
    while progress_tokens() < target_budget:

        if boost_left > 0:
            controls, _s = _compute_dynamic_controls(
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                base_top_k=base_top_k,
                z=boost_z,
                args=args,
            )
        else:
            controls = {"temperature": base_temperature, "top_p": base_top_p, "top_k": base_top_k}

        prev_text = "".join(out_text_parts)
        cont = _build_continuation_prompt(prompt, prev_text, tail_chars=int(args.dyn_tail_chars)) if prev_text else prompt

        want_lp = bool(args.want_logprobs) if mode == "h2" else False
        call_args = _clone_args(
            args,
            max_out=int(chunk_max),
            temperature=float(controls["temperature"]),
            top_p=float(controls["top_p"]),
            top_k=int(controls["top_k"]),
            want_logprobs=want_lp,
            top_logprobs=(int(args.top_logprobs) if want_lp else 0),
        )
        gr = prov.generate(prompt=cont, system=system, args=call_args, strip_output=False)
        if isinstance(gr.meta, dict) and "raw" in gr.meta:
            raw_calls.append(gr.meta.get("raw"))
        if not gr.text:
            break

        out_text_parts.append(gr.text)

        chunk_tokens, _ = tokenize_text(gr.text, mode="regex")
        chunk_tok_incr = max(1, len(chunk_tokens))
        gen_text_tokens += chunk_tok_incr
        if boost_left > 0:
            boost_left = max(0, boost_left - chunk_tok_incr)

        chunk_i = len(out_text_parts) - 1

        max_event_z = None
        max_event_token_i = None
        max_event_kind = None
        max_event_extra: Dict[str, Any] = {}

        def consider_event(info: Dict[str, Any], *, prefix: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
            nonlocal max_event_z, max_event_token_i, max_event_kind, max_event_extra
            is_spike = bool(info.get("is_spike"))
            is_slope = (not bool(args.dyn_no_slope)) and bool(info.get("is_slope"))
            if not (is_spike or is_slope):
                return
            z = max(float(info.get("z", 0.0)) if is_spike else 0.0,
                    float(info.get("slope_z", 0.0)) if is_slope else 0.0)
            kind = "spike" if is_spike else "slope"
            if is_spike and is_slope:
                kind = "spike+slope"
            if prefix:
                kind = f"{prefix}:{kind}"
            if (max_event_z is None) or (z > max_event_z):
                max_event_z = float(z)
                max_event_token_i = int(info.get("i")) if info.get("i") is not None else None
                max_event_kind = kind
                max_event_extra = dict(extra or {})

        if mode == "h2":
            if not (gr.token_strs and gr.token_logprobs_ln) or len(gr.token_strs) != len(gr.token_logprobs_ln):
                raise RuntimeError("dyn_mode=h2 requires provider logprobs (token_strs + token_logprobs_ln).")
            new_tokens = list(gr.token_strs)
            new_lps = [float(x) for x in gr.token_logprobs_ln]
            meas_tokens.extend(new_tokens)
            meas_logprobs.extend(new_lps)
            for lp in new_lps:
                info = detector.update(lp)
                consider_event(info)
            if not new_tokens:
                break

        elif mode == "teacher":
            assert scorer is not None
            score_ctx = _build_score_context(prompt, system, prev_text, tail_chars=int(getattr(args, "score_tail_chars", 1600)))
            try:
                t_toks, t_lps = scorer.score(score_ctx, gr.text)
            except Exception as e:
                if str(getattr(args, "score_error_mode", "error")) == "skip":
                    if getattr(args, "dyn_debug", False):
                        print(f"[dyn] teacher scoring failed (skipping chunk): {e}", file=sys.stderr)
                    teacher_cov_den += int(chunk_tok_incr)
                    teacher_budget_tokens += int(chunk_tok_incr)
                    continue
                raise
            if t_toks and t_lps and len(t_toks) == len(t_lps):
                teacher_cov_num += len(t_toks)
                teacher_cov_den += int(chunk_tok_incr)
                teacher_budget_tokens += len(t_toks)
                base_i = int(teacher_full_i)
                teacher_full_i += len(t_toks)

                toks_use = list(t_toks)
                lps_use = list(t_lps)
                offs_full: Optional[List[Tuple[int, int]]] = None
                if isinstance(scorer, LocalHFScorer):
                    offs_full = scorer.token_char_offsets(gr.text)
                    if offs_full is not None and len(offs_full) >= len(t_toks):
                        drop = len(offs_full) - len(t_toks)
                        if drop > 0:
                            offs_full = offs_full[drop:]
                    elif offs_full is not None and len(offs_full) != len(t_toks):
                        offs_full = None
                elif isinstance(scorer, OpenAICompletionsScorer):
                    offs_full = scorer.last_offsets
                    if offs_full is not None and len(offs_full) != len(t_toks):
                        offs_full = None
                elif isinstance(scorer, GeminiScorer):
                    offs_full = scorer.last_offsets
                    if offs_full is not None and len(offs_full) != len(t_toks):
                        offs_full = None

                offs_use: Optional[List[Tuple[int, int]]] = offs_full
                if score_sample_every > 1:
                    toks_use = []
                    lps_use = []
                    offs_use = ([] if offs_full is not None else None)
                    for j, (tok_str, lp) in enumerate(zip(t_toks, t_lps)):
                        if ((base_i + j) % int(score_sample_every)) == 0:
                            toks_use.append(tok_str)
                            lps_use.append(lp)
                            if offs_use is not None and offs_full is not None and j < len(offs_full):
                                offs_use.append(offs_full[j])

                meas_tokens.extend(toks_use)
                meas_logprobs.extend([float(x) for x in lps_use])
                if offs_use is not None and len(offs_use) == len(toks_use):
                    base_char = len(prev_text)
                    meas_char_offsets.extend([(base_char + int(a), base_char + int(b)) for (a, b) in offs_use])
                for lp in t_lps:
                    info = detector.update(float(lp))
                    consider_event(info, prefix="teacher")
            else:
                teacher_cov_den += int(chunk_tok_incr)
                teacher_budget_tokens += int(chunk_tok_incr)

        elif mode == "compressor":
            comp_tail = (comp_tail + gr.text)
            w = int(getattr(args, "comp_window_chars", 2400))
            if w > 0 and len(comp_tail) > w:
                comp_tail = comp_tail[-w:]
            bits_raw = _compressor_proxy_bits(
                comp_tail,
                algo=str(getattr(args, "comp_algo", "zlib")),
                level=int(getattr(args, "comp_level", 6)),
                window_chars=0,
                invert=bool(getattr(args, "comp_invert", False)),
            )
            bits, _ = smooth_proxy_bits(bits_raw)
            info = detector.update(_bits_to_pseudo_logprob_ln(bits))
            consider_event(info, prefix="compressor", extra={"proxy_bits": float(bits), "proxy_bits_raw": float(bits_raw), "proxy_ema_alpha": float(proxy_ema_alpha)})

        elif mode == "text":
            dyn_text_tokens.extend(chunk_tokens)
            dyn_text_tail = (dyn_text_tail + gr.text)
            wchars = int(getattr(args, "dyn_text_window_chars", 2400))
            if wchars > 0 and len(dyn_text_tail) > wchars:
                dyn_text_tail = dyn_text_tail[-wchars:]

            txt = gr.text or ""
            n_chars = max(len(txt), 1)
            punct_set = "。！？.!?、，,;:：；"
            punct_density = sum(1 for ch in txt if ch in punct_set) / float(n_chars)
            newline_density = txt.count("\n") / float(n_chars)
            density_jump = 0.0
            if prev_punct_density is not None:
                density_jump += abs(float(punct_density) - float(prev_punct_density))
            if prev_newline_density is not None:
                density_jump += abs(float(newline_density) - float(prev_newline_density))
            prev_punct_density = float(punct_density)
            prev_newline_density = float(newline_density)

            metrics = _text_collapse_proxy_bits(
                dyn_text_tokens,
                args,
                text_tail=dyn_text_tail,
                punct_density=float(punct_density),
                newline_density=float(newline_density),
                density_jump=float(density_jump),
            )
            bits_raw = float(metrics.get("bits", 0.0))
            bits, _ = smooth_proxy_bits(bits_raw)
            info = detector.update(_bits_to_pseudo_logprob_ln(bits))
            consider_event(info, prefix="text", extra={"proxy_bits": float(bits), "proxy_bits_raw": float(bits_raw), "proxy_ema_alpha": float(proxy_ema_alpha), **metrics})

            in_collapse = bits > 0.0
            if in_collapse and not was_in_collapse:
                rep_hi = float(getattr(args, "dyn_text_rep_hi", None) if getattr(args, "dyn_text_rep_hi", None) is not None else args.rep_hi)
                h1_low = float(getattr(args, "dyn_text_h1_low", None) if getattr(args, "dyn_text_h1_low", None) is not None else args.h1_low)
                distinct_low = float(getattr(args, "dyn_text_distinct_low", 0.75))
                line_rep_hi = float(getattr(args, "dyn_text_line_rep_hi", 0.20))
                density_jump_hi = float(getattr(args, "dyn_text_density_jump_hi", 0.02))
                rep_ex = float(metrics.get("rep_excess", 0.0))
                h1_def = float(metrics.get("h1_deficit", 0.0))
                dist_def = float(metrics.get("distinct_deficit", 0.0))
                line_ex = float(metrics.get("line_rep_excess", 0.0))
                jump_ex = float(metrics.get("density_excess", 0.0))
                parts = [
                    rep_ex / max(rep_hi, EPS),
                    h1_def / max(h1_low, EPS),
                    dist_def / max(distinct_low, EPS),
                    line_ex / max(line_rep_hi, EPS),
                    jump_ex / max(density_jump_hi, EPS),
                ]
                sev = _clamp(float(sum(parts) / max(len(parts), 1)), 0.0, 1.0)
                z_edge = float(args.dyn_spike_z) + sev * (float(args.dyn_full_z) - float(args.dyn_spike_z))
                if (max_event_z is None) or (z_edge > max_event_z):
                    max_event_z = float(z_edge)
                    max_event_token_i = int(info.get("i")) if info.get("i") is not None else None
                    max_event_kind = "text:collapse"
                    max_event_extra = {"proxy_bits": float(bits), **metrics}

            was_in_collapse = in_collapse

        if max_event_z is not None:
            applied, strength = _compute_dynamic_controls(
                base_temperature=base_temperature,
                base_top_p=base_top_p,
                base_top_k=base_top_k,
                z=float(max_event_z),
                args=args,
            )
            boost_left = max(boost_left, int(args.dyn_boost_tokens))
            boost_z = max(boost_z, float(max_event_z))

            if max_budget_on_event and max_budget_on_event > target_budget:
                target_budget = max_budget_on_event

            ev = {
                "mode": mode,
                "step_unit": step_unit,
                "measure_unit": measure_unit,
                "budget_unit": budget_unit,
                "chunk_i": int(chunk_i),
                "kind": str(max_event_kind) if max_event_kind is not None else "event",
                "token_i": int(max_event_token_i) if max_event_token_i is not None else None,
                "z": float(max_event_z),
                "strength": float(strength),
                "applied_next": applied,
                "new_target_max_out": int(target_budget),
                **max_event_extra,
            }
            events.append(ev)
            if getattr(args, "dyn_debug", False):
                print(f"[dyn] {ev['kind']} z={ev['z']:.2f} -> next={ev['applied_next']} max_out={target_budget}", file=sys.stderr)

    text_raw = "".join(out_text_parts)
    if getattr(args, "strip_generated", True):
        w0, w1 = _strip_window(text_raw)
        text = text_raw[w0:w1]
        if mode == "teacher" and meas_char_offsets and meas_tokens and meas_logprobs and len(meas_char_offsets) == len(meas_tokens) == len(meas_logprobs):
            ktoks: List[str] = []
            klps: List[float] = []
            koff: List[Tuple[int, int]] = []
            for tok, lp, (a, b) in zip(meas_tokens, meas_logprobs, meas_char_offsets):
                if int(b) <= int(w0) or int(a) >= int(w1):
                    continue
                if int(a) < int(w0) or int(b) > int(w1):
                    continue
                ktoks.append(tok)
                klps.append(float(lp))
                koff.append((int(a) - int(w0), int(b) - int(w0)))
            meas_tokens = ktoks
            meas_logprobs = klps
            meas_char_offsets = koff
    else:
        text = text_raw

    ds_meta: Dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "step_unit": step_unit,
        "measure_unit": measure_unit,
        "budget_unit": budget_unit,
        "events": events,
        "detector": {
            "h2_win": int(args.h2_win),
            "dyn_z_window": int(args.dyn_z_window),
            "dyn_spike_z": float(args.dyn_spike_z),
            "dyn_full_z": float(args.dyn_full_z),
            "dyn_no_slope": bool(args.dyn_no_slope),
            "dyn_slope_window": int(args.dyn_slope_window),
            "dyn_slope_z": float(args.dyn_slope_z),
        },
    }
    if mode == "teacher" and scorer is not None:
        ds_meta["teacher"] = {
            "score_provider": str(getattr(args, "score_provider", "")),
            "score_model": str(getattr(args, "score_model", "")) if hasattr(args, "score_model") else "",
            "score_local_model": (scorer.model_path if isinstance(scorer, LocalHFScorer) else None),
            "score_stride_tokens": int(getattr(args, "score_stride_tokens", 64)),
            "score_sample_every": int(getattr(args, "score_sample_every", 1)),
            "score_tail_chars": int(getattr(args, "score_tail_chars", 1600)),
        }
    if mode == "compressor":
        ds_meta["compressor"] = {
            "comp_algo": str(getattr(args, "comp_algo", "zlib")),
            "comp_level": int(getattr(args, "comp_level", 6)),
            "comp_window_chars": int(getattr(args, "comp_window_chars", 2400)),
            "comp_invert": bool(getattr(args, "comp_invert", False)),
            "dyn_proxy_ema_alpha": float(proxy_ema_alpha),
        }
    if mode == "text":
        rep_hi = getattr(args, "dyn_text_rep_hi", None)
        if rep_hi is None:
            rep_hi = float(args.rep_hi)
        h1_low = getattr(args, "dyn_text_h1_low", None)
        if h1_low is None:
            h1_low = float(args.h1_low)
        ds_meta["text_proxy"] = {
            "dyn_text_window_tokens": int(getattr(args, "dyn_text_window_tokens", 256)),
            "dyn_text_window_chars": int(getattr(args, "dyn_text_window_chars", 2400)),
            "dyn_text_rep_hi": float(rep_hi),
            "dyn_text_h1_low": float(h1_low),
            "dyn_text_w_rep": float(getattr(args, "dyn_text_w_rep", 1.0)),
            "dyn_text_w_h1": float(getattr(args, "dyn_text_w_h1", 1.0)),
            "dyn_text_distinct_low": float(getattr(args, "dyn_text_distinct_low", 0.75)),
            "dyn_text_line_rep_hi": float(getattr(args, "dyn_text_line_rep_hi", 0.20)),
            "dyn_text_density_jump_hi": float(getattr(args, "dyn_text_density_jump_hi", 0.02)),
            "dyn_text_w_distinct": float(getattr(args, "dyn_text_w_distinct", 1.0)),
            "dyn_text_w_line_rep": float(getattr(args, "dyn_text_w_line_rep", 1.0)),
            "dyn_text_w_density_jump": float(getattr(args, "dyn_text_w_density_jump", 0.5)),
            "ngram_ns": [3, 4, 5],
            "dyn_proxy_ema_alpha": float(proxy_ema_alpha),
        }

    meta = {
        "provider": getattr(prov, "name", "remote"),
        "dynamic_sampling": ds_meta,
        "raw": raw_calls,
    }

    if mode == "teacher":
        cov = (float(teacher_cov_num) / float(teacher_cov_den)) if teacher_cov_den > 0 else None
        meta["measure"] = {
            "mode": "teacher",
            "tokens": meas_tokens,
            "logprobs_ln": meas_logprobs,
            "coverage": cov,
            "coverage_is_approx": True,
            "coverage_basis": "teacher_token / gen_regex_token",
            "score_sample_every": int(score_sample_every),
        }
        if meas_char_offsets and len(meas_char_offsets) == len(meas_tokens):
            meta["measure"]["char_offsets"] = meas_char_offsets
            meta["measure"]["char_offsets_unit"] = "generated_text_char"
        if scorer is not None:
            meta["measure"]["score_provider"] = str(getattr(args, "score_provider", "") or scorer.name)
            if isinstance(scorer, OpenAICompletionsScorer):
                meta["measure"]["score_model"] = scorer.model
            if isinstance(scorer, LocalHFScorer):
                meta["measure"]["score_local_model"] = scorer.model_path
                meta["measure"]["score_stride_tokens"] = int(getattr(args, "score_stride_tokens", 64))

    token_strs_out = (meas_tokens or None) if mode in ("h2", "teacher") else None
    lps_out = (meas_logprobs or None) if mode in ("h2", "teacher") else None
    return GenResult(text=text, token_strs=token_strs_out, token_logprobs_ln=lps_out, meta=meta)


# =========================================================
# Alignment: provider logprobs tokens ↔ text tokens (suffix match)
# =========================================================
@dataclass
class AlignResult:
    aligned_logprobs_ln: List[Optional[float]]   # len = len(text_tokens), None where unknown
    mapped_h2_spikes_text_idx: List[int]         # token indices in text space
    suffix_match_len: int
    coverage: Optional[float]

@dataclass
class MeasureCharAlign:
    tokens: List[str]
    logprobs_ln: List[float]
    char_offsets: List[Tuple[int, int]]  # in `text` char space
    spikes_idx: List[int]               # indices in this aligned series
    info: Dict[str, Any]

def align_by_longest_common_suffix(text_tokens: List[str],
                                  lp_tokens: List[str],
                                  lp_logprobs_ln: List[float],
                                  h2_spikes_lp_idx: List[int],
                                  min_suffix_tokens: int = 24) -> AlignResult:
    """
    Find L = longest common suffix such that
      text_tokens[-L:] == lp_tokens[-L:]
    Then place lp_logprobs into aligned array at the last L positions.
    Map H2 spikes from lp token indices to text token indices only when spike falls inside suffix window.
    """
    N = len(text_tokens)
    M = len(lp_tokens)
    L = 0
    while L < min(N, M):
        if text_tokens[N - 1 - L] == lp_tokens[M - 1 - L]:
            L += 1
        else:
            break

    aligned: List[Optional[float]] = [None] * N
    mapped_spikes: List[int] = []

    coverage = (M / N) if N > 0 else None

    if L >= min_suffix_tokens:
        # suffix window:
        text_start = N - L
        lp_start = M - L
        for j in range(L):
            aligned[text_start + j] = float(lp_logprobs_ln[lp_start + j])

        # map spikes
        for si in h2_spikes_lp_idx:
            if lp_start <= si < M:
                mapped = text_start + (si - lp_start)
                mapped_spikes.append(mapped)

    return AlignResult(aligned_logprobs_ln=aligned,
                       mapped_h2_spikes_text_idx=sorted(set(mapped_spikes)),
                       suffix_match_len=L,
                       coverage=coverage)

def _strip_window(s: str) -> Tuple[int, int]:
    """
    Returns (start, end) such that s[start:end] == s.strip().
    """
    start = len(s) - len(s.lstrip())
    end = len(s.rstrip())
    return int(start), int(end)

def align_measure_series_to_text(text: str,
                                 tokens: List[str],
                                 logprobs_ln: List[float],
                                 spikes_idx: List[int]) -> Optional[MeasureCharAlign]:
    """
    Attempts to derive per-token char offsets in `text` for a (token_str, logprob) series.
    Supports:
      - exact concat: ''.join(tokens) == text
      - strip-window: ''.join(tokens).strip() == text  (common when generated text was .strip()'d)
      - greedy find: sequential substring find (allows whitespace gaps)
    """
    if not tokens or not logprobs_ln or len(tokens) != len(logprobs_ln):
        return None

    joined = "".join(tokens)

    # Fast path: exact concatenation
    if joined == text:
        offs: List[Tuple[int, int]] = []
        pos = 0
        for t in tokens:
            start = pos
            pos += len(t)
            offs.append((start, pos))
        return MeasureCharAlign(tokens=list(tokens),
                                logprobs_ln=[float(x) for x in logprobs_ln],
                                char_offsets=offs,
                                spikes_idx=sorted(set(int(i) for i in spikes_idx if 0 <= int(i) < len(tokens))),
                                info={"method": "concat_exact", "approx": False, "dropped": 0})

    # Common case: provider tokens reflect unstripped text but `text` is stripped.
    if joined.strip() == text:
        w0, w1 = _strip_window(joined)
        kept_tokens: List[str] = []
        kept_lps: List[float] = []
        kept_offs: List[Tuple[int, int]] = []
        kept_map: Dict[int, int] = {}

        pos = 0
        for i, t in enumerate(tokens):
            start = pos
            pos += len(t)
            end = pos
            if end <= w0 or start >= w1:
                continue
            # drop partial boundary tokens to keep offsets well-defined in stripped text
            if start < w0 or end > w1:
                continue
            kept_map[int(i)] = len(kept_tokens)
            kept_tokens.append(t)
            kept_lps.append(float(logprobs_ln[i]))
            kept_offs.append((start - w0, end - w0))

        kept_spikes: List[int] = []
        for si in spikes_idx:
            if int(si) in kept_map:
                kept_spikes.append(int(kept_map[int(si)]))

        dropped = int(len(tokens) - len(kept_tokens))
        return MeasureCharAlign(tokens=kept_tokens,
                                logprobs_ln=kept_lps,
                                char_offsets=kept_offs,
                                spikes_idx=sorted(set(kept_spikes)),
                                info={"method": "concat_strip_window", "approx": True, "dropped": dropped, "strip_start": w0, "strip_end": w1})

    # Fallback: greedy substring alignment in text
    kept_tokens = []
    kept_lps = []
    kept_offs = []
    kept_map: Dict[int, int] = {}
    pos = 0
    for i, t in enumerate(tokens):
        if t == "":
            continue
        idx = pos if text.startswith(t, pos) else text.find(t, pos)
        if idx < 0:
            return None
        if idx > pos:
            gap = text[pos:idx]
            if gap.strip() != "":
                return None
        start = idx
        end = idx + len(t)
        kept_map[int(i)] = len(kept_tokens)
        kept_tokens.append(t)
        kept_lps.append(float(logprobs_ln[i]))
        kept_offs.append((int(start), int(end)))
        pos = end

    kept_spikes = []
    for si in spikes_idx:
        if int(si) in kept_map:
            kept_spikes.append(int(kept_map[int(si)]))

    dropped = int(len(tokens) - len(kept_tokens))
    return MeasureCharAlign(tokens=kept_tokens,
                            logprobs_ln=kept_lps,
                            char_offsets=kept_offs,
                            spikes_idx=sorted(set(kept_spikes)),
                            info={"method": "greedy_find", "approx": True, "dropped": dropped})

def map_spike_chars_to_text_tokens(spike_char_positions: List[int],
                                  text_token_spans: List[Tuple[int, int]]) -> List[int]:
    if not spike_char_positions or not text_token_spans:
        return []
    starts = [int(s) for (s, _e) in text_token_spans]
    out: List[int] = []
    for p in spike_char_positions:
        i = bisect.bisect_right(starts, int(p)) - 1
        if i < 0 or i >= len(text_token_spans):
            continue
        s, e = text_token_spans[i]
        if int(s) <= int(p) < int(e):
            out.append(int(i))
    return sorted(set(out))

def token_spans_in_text(text: str, tokens: List[str], tokenizer_name: str) -> Optional[List[Tuple[int, int]]]:
    if tokenizer_name.startswith("regex"):
        pat = REGEX_STRUCT_PATTERN if tokenizer_name.startswith("regex_struct") else REGEX_TOKEN_PATTERN
        spans = [(m.start(), m.end()) for m in re.finditer(pat, text, flags=re.UNICODE)]
        if len(spans) != len(tokens):
            return None
        return spans
    if "".join(tokens) == text:
        spans: List[Tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            start = pos
            pos += len(tok)
            spans.append((start, pos))
        if pos != len(text):
            return None
        return spans
    return None

def segment_h2_from_char_offsets(*,
                                 text: str,
                                 seg_ix: List[Tuple[int, int]],
                                 text_token_spans: List[Tuple[int, int]],
                                 measure_char_offsets: List[Tuple[int, int]],
                                 measure_logprobs_ln: List[float],
                                 h2_spikes_meas_idx: List[int]) -> Tuple[List[Optional[float]], List[int]]:
    nseg = len(seg_ix)
    seg_char_spans: List[Tuple[int, int]] = []
    for i, (s, _e) in enumerate(seg_ix):
        start = text_token_spans[s][0] if 0 <= s < len(text_token_spans) else 0
        if i + 1 < nseg:
            ns = seg_ix[i + 1][0]
            end = text_token_spans[ns][0] if 0 <= ns < len(text_token_spans) else len(text)
        else:
            end = len(text)
        seg_char_spans.append((int(start), int(end)))

    seg_sum = [0.0] * nseg
    seg_cnt = [0] * nseg
    seg_i = 0
    for (off, lp) in zip(measure_char_offsets, measure_logprobs_ln):
        if seg_i >= nseg:
            break
        ts = int(off[0])
        while seg_i < nseg and ts >= seg_char_spans[seg_i][1]:
            seg_i += 1
        if seg_i >= nseg:
            break
        if ts < seg_char_spans[seg_i][0]:
            continue
        seg_sum[seg_i] += ln_to_bits_surprisal(float(lp))
        seg_cnt[seg_i] += 1

    seg_h2 = [(seg_sum[i] / seg_cnt[i]) if seg_cnt[i] > 0 else None for i in range(nseg)]

    seg_ends = [b for (_a, b) in seg_char_spans]
    spike_segs: set = set()
    for si in h2_spikes_meas_idx:
        if not (0 <= int(si) < len(measure_char_offsets)):
            continue
        ts = int(measure_char_offsets[int(si)][0])
        j = bisect.bisect_right(seg_ends, ts)
        if 0 <= j < nseg and ts >= seg_char_spans[j][0]:
            spike_segs.add(int(j))

    return seg_h2, sorted(spike_segs)


# =========================================================
# Analysis
# =========================================================
@dataclass
class SegmentReport:
    i: int
    start: int
    end: int
    n_tok: int
    preview: str
    H1_bits: float
    H1_norm: float
    rep_ngram: float
    drift: Optional[float]
    drift_z: Optional[float]
    H2_avg_bits: Optional[float]
    bloom: bool
    phase: str

@dataclass
class GlobalReport:
    tokenizer: str
    n_tokens: int
    vocab_unique: int
    H1_bits: float
    H1_norm: float
    rep_ngram: float
    H2_bits_per_tok: Optional[float]
    drift_mean: float
    drift_max: float
    drift_spikes: List[int]     # segment idx
    h2_spikes_lp: List[int]     # lp token idx
    h2_spikes_text: List[int]   # text token idx (aligned only)
    bloom_segments: List[int]   # segment idx
    alignment: Dict[str, Any]   # coverage, suffix_match_len, aligned

def analyze_structural(text_tokens: List[str], args: argparse.Namespace) -> Tuple[float, float, float, int]:
    c = unigram_counts(text_tokens)
    V = len(c)
    H1 = shannon_entropy_from_counts(c)
    H1n = norm_entropy(H1, max(V, 2))
    rep = ngram_recurrence(text_tokens, n=args.ngram_n)
    return H1, H1n, rep, V

def segment_indices(tokens: List[str], args: argparse.Namespace) -> Tuple[List[Tuple[int, int]], str]:
    if args.segment_mode == "window":
        return segment_indices_by_window(len(tokens), window_tokens=args.window_tokens), "window"
    if args.segment_mode == "punct":
        return segment_indices_by_punct(tokens, endchars=args.endchars,
                                        min_tokens=args.min_seg_tokens,
                                        max_tokens=(args.max_seg_tokens if args.max_seg_tokens > 0 else None)), "punct"
    return segment_indices_rich(tokens, endchars=args.endchars,
                               min_tokens=args.min_seg_tokens,
                               max_tokens=(args.max_seg_tokens if args.max_seg_tokens > 0 else None),
                               newline_hard=not args.no_newline_split,
                               bullet_hard=not args.no_bullet_split,
                               quote_hard=not args.no_quote_split,
                               parenclose_soft=not args.no_parenclose_split), "rich"

def compute_drift_spikes(seg_H1n: List[float], spike_z: float) -> Tuple[List[float], List[float], List[int]]:
    drift = []
    idx = []
    for i in range(1, len(seg_H1n)):
        drift.append(abs(seg_H1n[i] - seg_H1n[i - 1]))
        idx.append(i)
    z = mad_zscores(drift) if drift else []
    spikes = [idx[i] for i, zz in enumerate(z) if zz >= spike_z]
    return drift, z, spikes

def compute_h2_token_series(lp_logprobs_ln: List[float], h2_win: int, spike_z: float) -> Tuple[List[float], List[float], List[int], float]:
    """
    Returns:
      surprisal_bits (per token),
      rolling_mean,
      jump_z (MAD z on positive rolling jumps),
      spike_indices (in lp token space),
      H2_global (avg bits/token)
    """
    s_bits = [ln_to_bits_surprisal(x) for x in lp_logprobs_ln]
    H2_global = sum(s_bits) / max(len(s_bits), 1)
    rm = rolling_mean(s_bits, w=h2_win)
    jumps = [0.0] + [rm[i] - rm[i - 1] for i in range(1, len(rm))]
    pos = [max(0.0, j) for j in jumps]
    z = mad_zscores(pos)
    spikes = [i for i, zz in enumerate(z) if zz >= spike_z and pos[i] > 0]
    return s_bits, rm, z, spikes, H2_global

def map_token_spikes_to_segments(token_spikes_text_idx: List[int], seg_ix: List[Tuple[int, int]]) -> List[int]:
    hit = set()
    for ti in token_spikes_text_idx:
        for si, (s, e) in enumerate(seg_ix):
            if s <= ti < e:
                hit.add(si)
                break
    return sorted(hit)

def build_reports(text: str,
                  text_tokens: List[str],
                  tokenizer_name: str,
                  seg_ix: List[Tuple[int, int]],
                  drift_spikes_seg: List[int],
                  drift_vals: List[float],
                  drift_z: List[float],
                  aligned_logprobs_ln: Optional[List[Optional[float]]],
                  mapped_h2_spikes_text: List[int],
                  H2_global_bits: Optional[float],
                  args: argparse.Namespace,
                  *,
                  seg_h2avg_override_bits: Optional[List[Optional[float]]] = None,
                  h2_spike_segs_override: Optional[List[int]] = None) -> Tuple[GlobalReport, List[SegmentReport], Dict[str, Any]]:

    # global structural
    H1, H1n, rep, V = analyze_structural(text_tokens, args)
    drift_mean = float(sum(drift_vals) / max(len(drift_vals), 1)) if drift_vals else 0.0
    drift_max = float(max(drift_vals)) if drift_vals else 0.0

    # segment structural + optional segment-H2
    seg_reports: List[SegmentReport] = []
    seg_H1n: List[float] = []
    seg_rep: List[float] = []
    seg_preview: List[str] = []
    seg_H1bits: List[float] = []
    seg_H2avg: List[Optional[float]] = []

    for i, (s, e) in enumerate(seg_ix):
        stoks = text_tokens[s:e]
        sc = unigram_counts(stoks)
        VH = len(sc)
        H1s = shannon_entropy_from_counts(sc)
        H1sn = norm_entropy(H1s, max(VH, 2))
        seg_H1bits.append(H1s)
        seg_H1n.append(H1sn)
        seg_rep.append(ngram_recurrence(stoks, n=args.ngram_n))

        preview = "".join(stoks).replace("\n", " ")
        if len(preview) > args.preview_chars:
            preview = preview[:args.preview_chars] + "…"
        seg_preview.append(preview)

        if seg_h2avg_override_bits is not None:
            seg_H2avg.append(seg_h2avg_override_bits[i] if i < len(seg_h2avg_override_bits) else None)
        elif aligned_logprobs_ln is not None:
            lps = [lp for lp in aligned_logprobs_ln[s:e] if lp is not None]
            if lps:
                bits = [ln_to_bits_surprisal(x) for x in lps]
                seg_H2avg.append(sum(bits) / max(len(bits), 1))
            else:
                seg_H2avg.append(None)
        else:
            seg_H2avg.append(None)

    # bloom segments: drift spikes OR (H2 spikes passing structural gate)
    bloom_seg_flags = [False] * len(seg_ix)
    for si in drift_spikes_seg:
        if 0 <= si < len(bloom_seg_flags):
            bloom_seg_flags[si] = True

    if h2_spike_segs_override is not None:
        h2_spike_segs = sorted(set(int(x) for x in h2_spike_segs_override if isinstance(x, int) or (isinstance(x, float) and x == int(x))))
    else:
        h2_spike_segs = map_token_spikes_to_segments(mapped_h2_spikes_text, seg_ix) if mapped_h2_spikes_text else []
    for si in h2_spike_segs:
        if 0 <= si < len(bloom_seg_flags):
            rep_ok = float(seg_rep[si]) <= float(args.bloom_h2_rep_max)
            h1_ok = float(seg_H1n[si]) >= float(args.bloom_h2_h1_min)
            if rep_ok and h1_ok:
                bloom_seg_flags[si] = True

    bloom_segments = [i for i, f in enumerate(bloom_seg_flags) if f]

    # phase labels
    for i, (s, e) in enumerate(seg_ix):
        H1sn = seg_H1n[i]
        repn = seg_rep[i]
        d = None
        dz = None
        if i >= 1 and drift_vals:
            d = drift_vals[i - 1]
            dz = drift_z[i - 1] if (i - 1) < len(drift_z) else None

        bloom = bloom_seg_flags[i]

        if (repn >= args.rep_hi) and (H1sn <= args.h1_low) and (d is None or d <= args.drift_low):
            phase = "Collapse"
        elif bloom:
            phase = "Bloom"
        elif d is not None and d >= args.drift_mid:
            phase = "Drift"
        else:
            phase = "Stable"

        seg_reports.append(SegmentReport(
            i=i, start=s, end=e, n_tok=(e - s),
            preview=seg_preview[i],
            H1_bits=float(seg_H1bits[i]),
            H1_norm=float(H1sn),
            rep_ngram=float(repn),
            drift=(float(d) if d is not None else None),
            drift_z=(float(dz) if dz is not None else None),
            H2_avg_bits=(float(seg_H2avg[i]) if seg_H2avg[i] is not None else None),
            bloom=bool(bloom),
            phase=phase
        ))

    alignment_info = {
        "aligned_segment_h2": bool((aligned_logprobs_ln is not None) or (seg_h2avg_override_bits is not None)),
        "mapped_h2_spikes_text_count": len(mapped_h2_spikes_text),
        "mapped_h2_spikes_seg_count": len(h2_spike_segs),
    }

    g = GlobalReport(
        tokenizer=tokenizer_name,
        n_tokens=len(text_tokens),
        vocab_unique=V,
        H1_bits=float(H1),
        H1_norm=float(H1n),
        rep_ngram=float(rep),
        H2_bits_per_tok=(float(H2_global_bits) if H2_global_bits is not None else None),
        drift_mean=drift_mean,
        drift_max=drift_max,
        drift_spikes=sorted(set(drift_spikes_seg)),
        h2_spikes_lp=[],        # filled by caller
        h2_spikes_text=sorted(set(mapped_h2_spikes_text)),
        bloom_segments=bloom_segments,
        alignment=alignment_info
    )

    extra = {
        "thresholds": {
            "rep_hi": args.rep_hi,
            "h1_low": args.h1_low,
            "drift_low": args.drift_low,
            "drift_mid": args.drift_mid,
            "bloom_h2_rep_max": float(args.bloom_h2_rep_max),
            "bloom_h2_h1_min": float(args.bloom_h2_h1_min),
        }
    }
    return g, seg_reports, extra


# =========================================================
# Plotting
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_segments(segs: List[SegmentReport], outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.8))
    ax = plt.gca()

    x = [s.i for s in segs]
    H1 = [s.H1_norm for s in segs]
    rep = [s.rep_ngram for s in segs]
    drift = [(s.drift if s.drift is not None else 0.0) for s in segs]
    H2 = [(s.H2_avg_bits if s.H2_avg_bits is not None else float("nan")) for s in segs]
    n_tok = [max(1, int(s.n_tok)) for s in segs]

    phases = [s.phase for s in segs]
    blooms = [bool(s.bloom) for s in segs]

    # Bloom shading
    for i, b in enumerate(blooms):
        if b:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08)

    # Phase boundary vertical lines (any phase change)
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(i - 0.5, alpha=0.20)

    # Segment length bars on log-scale secondary axis
    ax2 = ax.twinx()
    ax2.bar(x, n_tok, alpha=0.12, label="seg_len (n_tok)")
    ax2.set_yscale("log")
    ax2.set_ylabel("n_tok (log)")

    # Main curves
    ax.plot(x, H1, label="H1_norm (seg)")
    ax.plot(x, rep, label="rep_ngram (seg)")
    ax.plot(x, drift, label="drift (seg)")
    if any(not (isinstance(v, float) and math.isnan(v)) for v in H2):
        ax.plot(x, H2, label="H2_avg_bits (seg)")

    ax.set_title("Segment Signals + Bloom shading + Phase boundaries + log(n_tok) bars")
    ax.set_xlabel("segment index")
    ax.set_ylabel("value")

    # Merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2h, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2h, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)

def plot_phase_strip(segs: List[SegmentReport], outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase_to_y = {"Collapse": 0, "Stable": 1, "Drift": 2, "Bloom": 3}
    x = [s.i for s in segs]
    y = [phase_to_y.get(s.phase, 1) for s in segs]
    phases = [s.phase for s in segs]
    blooms = [bool(s.bloom) for s in segs]

    fig = plt.figure(figsize=(11, 2.2))
    ax = plt.gca()

    # Bloom shading
    for i, b in enumerate(blooms):
        if b:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08)

    # boundaries
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(i - 0.5, alpha=0.20)

    ax.scatter(x, y, s=20)
    ax.set_yticks([0, 1, 2, 3], ["Collapse", "Stable", "Drift", "Bloom"])
    ax.set_title("Phase Strip + boundaries (Bloom shaded)")
    ax.set_xlabel("segment index")

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)

def plot_phase_space(segs: List[SegmentReport], outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_map = {
        "Collapse": "s",  # square
        "Stable":   "o",  # circle
        "Drift":    "x",  # x
        "Bloom":    "^",  # triangle up
    }

    groups: Dict[str, Tuple[List[float], List[float]]] = {}
    for s in segs:
        ph = s.phase
        groups.setdefault(ph, ([], []))
        groups[ph][0].append(s.H1_norm)
        groups[ph][1].append(s.rep_ngram)

    fig = plt.figure(figsize=(6.8, 6.0))
    ax = plt.gca()

    order = ["Collapse", "Stable", "Drift", "Bloom"]
    for ph in order:
        if ph not in groups:
            continue
        xs, ys = groups[ph]
        ax.scatter(xs, ys,
                   s=42 if ph in ("Bloom", "Collapse") else 30,
                   marker=marker_map.get(ph, "o"),
                   label=ph)

    ax.set_title("Phase Space: H1_norm vs rep_ngram (marker=phase)")
    ax.set_xlabel("H1_norm (seg)")
    ax.set_ylabel("rep_ngram (seg)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)

def plot_token_surprisal(token_rm: List[float], spikes: List[int], outpath: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(token_rm)))
    fig = plt.figure(figsize=(11, 3.8))
    ax = plt.gca()

    ax.plot(x, token_rm, label="rolling mean surprisal (bits)")
    if spikes:
        xs = [i for i in spikes if 0 <= i < len(token_rm)]
        ys = [token_rm[i] for i in xs]
        ax.scatter(xs, ys, s=22, label="H2 spikes")

    ax.set_title("Token-level H2 (rolling surprisal) + spikes")
    ax.set_xlabel("lp token index")
    ax.set_ylabel("bits")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


# =========================================================
# CLI / main
# =========================================================
def get_provider(name: str) -> Provider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "mistral":
        return MistralProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "local":
        return LocalHFProvider()
    raise ValueError(f"unknown provider: {name}")

def read_text_arg(inline: Optional[str], path: Optional[str]) -> Optional[str]:
    if inline is not None:
        return inline
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", type=str, required=True, choices=["openai", "mistral", "gemini", "local"])
    ap.add_argument("--model", type=str, default=None, help="remote model name (openai/mistral/gemini)")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--system_file", type=str, default=None)

    ap.add_argument("--max_out", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40, help="top-k cutoff (0 disables). Used by local + gemini; ignored elsewhere.")
    ap.add_argument("--want_logprobs", action="store_true")
    ap.add_argument("--top_logprobs", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)

    # dynamic sampling (signal -> temporary diversity push)
    ap.add_argument("--dyn_sampling", action="store_true", help="Enable adaptive sampling (local=tokenwise, remote=chunkwise).")
    ap.add_argument("--dyn_mode", type=str, default="h2", choices=["h2", "teacher", "compressor", "text"],
                    help="Signal source: h2=provider logprobs, teacher=separate scorer, compressor=text compression proxy, text=structural proxy.")
    ap.add_argument("--dyn_chunk_tokens", type=int, default=64, help="Remote chunk size when --dyn_sampling (approx).")
    ap.add_argument("--dyn_tail_chars", type=int, default=1200, help="Chars of previous output included in remote continuation prompt (0=full).")
    ap.add_argument("--dyn_z_window", type=int, default=64, help="Window for robust z-score on positive H2 rolling-jumps.")
    ap.add_argument("--dyn_spike_z", type=float, default=None, help="Trigger z (default: --spike_z).")
    ap.add_argument("--dyn_full_z", type=float, default=None, help="Full-strength z (default: dyn_spike_z+2.0).")
    ap.add_argument("--dyn_slope_window", type=int, default=24, help="Slope window (tokens) on rolling surprisal for trend triggers.")
    ap.add_argument("--dyn_slope_z", type=float, default=None, help="Trigger z for slope events (default: dyn_spike_z).")
    ap.add_argument("--dyn_no_slope", action="store_true", help="Disable slope-based triggers (use spikes only).")
    ap.add_argument("--dyn_boost_tokens", type=int, default=32, help="Boost duration in tokens after a spike/slope event.")
    ap.add_argument("--dyn_temp_boost", type=float, default=0.6, help="Max temperature increase at full strength.")
    ap.add_argument("--dyn_top_p_boost", type=float, default=0.05, help="Max top_p increase at full strength.")
    ap.add_argument("--dyn_top_p_cap", type=float, default=1.0, help="Clamp for boosted top_p.")
    ap.add_argument("--dyn_top_k_cap", type=int, default=400, help="Top-k ceiling while widening (before disabling at dyn_top_k_disable_z).")
    ap.add_argument("--dyn_top_k_disable_z", type=float, default=None, help="If z >= this, disable top_k (0). Default: dyn_full_z.")
    ap.add_argument("--dyn_max_out_on_spike", type=int, default=0, help="If >0, raise total max_out to this after first spike.")
    ap.add_argument("--dyn_debug", action="store_true", help="Print spike->control decisions to stderr.")
    ap.add_argument("--dyn_proxy_ema_alpha", type=float, default=0.35, help="EMA alpha for proxy bits smoothing (dyn_mode=text/compressor). 0 disables.")

    # teacher scoring (for dyn_mode=teacher)
    ap.add_argument("--score_provider", type=str, default=None, choices=["openai", "gemini", "local", "hf"], help="Scoring provider for dyn_mode=teacher.")
    ap.add_argument("--score_model", type=str, default=None, help="Scoring model name (teacher).")
    ap.add_argument("--score_openai_api_key", type=str, default=None, help="API key for OpenAI-compatible scoring (teacher). Overrides --openai_api_key/OPENAI_API_KEY.")
    ap.add_argument("--score_openai_base_url", type=str, default=None, help="Base URL for OpenAI-compatible scoring (teacher). Overrides --openai_base_url.")
    ap.add_argument("--score_gemini_api_key", type=str, default=None, help="API key for Gemini scoring (teacher). Overrides --gemini_api_key/GEMINI_API_KEY.")
    ap.add_argument("--score_gemini_base_url", type=str, default=None, help="Base URL for Gemini scoring (teacher). Overrides --gemini_base_url.")
    ap.add_argument("--score_local_model", type=str, default=None, help="Local HF model path/id for scoring (teacher).")
    ap.add_argument("--score_device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device for local scoring model.")
    ap.add_argument("--score_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"], help="Dtype for local scoring model.")
    ap.add_argument("--score_stride_tokens", type=int, default=64, help="Token block size for local teacher scoring (KV-cache stride).")
    ap.add_argument("--score_sample_every", type=int, default=1, help="Subsample teacher tokens/logprobs: keep 1 out of N (>=1).")
    ap.add_argument("--score_tail_chars", type=int, default=1600, help="Chars of previous output included for teacher scoring context (0=full).")
    ap.add_argument("--score_retries", type=int, default=6, help="Retries for teacher scoring API calls on transient errors.")
    ap.add_argument("--score_retry_backoff", type=float, default=0.6, help="Base backoff (seconds) for teacher scoring retries (exponential).")
    ap.add_argument("--score_error_mode", type=str, default="error", choices=["error", "skip"], help="If scoring fails for a chunk: error or skip (continue).")

    # compressor proxy (for dyn_mode=compressor)
    ap.add_argument("--comp_algo", type=str, default="zlib", choices=["zlib", "lzma"])
    ap.add_argument("--comp_level", type=int, default=6)
    ap.add_argument("--comp_window_chars", type=int, default=2400)
    ap.add_argument("--comp_invert", action="store_true", help="Invert compressor signal (react to becoming MORE compressible). Often good for collapse detection.")

    # text proxy (for dyn_mode=text)
    ap.add_argument("--dyn_text_window_tokens", type=int, default=256)
    ap.add_argument("--dyn_text_window_chars", type=int, default=2400, help="Chars kept for line/sentence repetition + density features.")
    ap.add_argument("--dyn_text_rep_hi", type=float, default=None, help="Override rep_hi for text proxy (default: --rep_hi).")
    ap.add_argument("--dyn_text_h1_low", type=float, default=None, help="Override h1_low for text proxy (default: --h1_low).")
    ap.add_argument("--dyn_text_w_rep", type=float, default=1.0)
    ap.add_argument("--dyn_text_w_h1", type=float, default=1.0)
    ap.add_argument("--dyn_text_distinct_low", type=float, default=0.75, help="Trigger when distinct-n falls below this (avg of n=3/4/5).")
    ap.add_argument("--dyn_text_line_rep_hi", type=float, default=0.20, help="Trigger when line/sentence repetition exceeds this.")
    ap.add_argument("--dyn_text_density_jump_hi", type=float, default=0.02, help="Trigger when punct/newline density jump exceeds this (chunk-wise).")
    ap.add_argument("--dyn_text_w_distinct", type=float, default=1.0)
    ap.add_argument("--dyn_text_w_line_rep", type=float, default=1.0)
    ap.add_argument("--dyn_text_w_density_jump", type=float, default=0.5)

    # tokenization (dual streams: struct for SR metrics, align for overlays)
    ap.add_argument("--tok_mode", "--struct_tok_mode", dest="struct_tok_mode",
                    type=str, default="auto", choices=["auto", "tiktoken_grouped", "regex"],
                    help="Structural token stream for SR metrics/segmentation. (Legacy alias: --tok_mode)")
    ap.add_argument("--align_tok_mode", type=str, default="auto", choices=["auto", "tiktoken", "regex"],
                    help="Alignment token stream (1ID=1 when tiktoken). Used for coverage/debug; structural analysis uses --struct_tok_mode.")

    # segmentation
    ap.add_argument("--segment_mode", type=str, default="rich", choices=["rich", "punct", "window"])
    ap.add_argument("--window_tokens", type=int, default=64)
    ap.add_argument("--endchars", type=str, default=DEFAULT_ENDCHARS)
    ap.add_argument("--min_seg_tokens", type=int, default=4)
    ap.add_argument("--max_seg_tokens", type=int, default=0)
    ap.add_argument("--preview_chars", type=int, default=72)

    # rich toggles
    ap.add_argument("--no_newline_split", action="store_true")
    ap.add_argument("--no_bullet_split", action="store_true")
    ap.add_argument("--no_quote_split", action="store_true")
    ap.add_argument("--no_parenclose_split", action="store_true")

    # repetition + spikes
    ap.add_argument("--ngram_n", type=int, default=3)
    ap.add_argument("--h2_win", type=int, default=16)
    ap.add_argument("--spike_z", type=float, default=2.5)

    # phase thresholds
    ap.add_argument("--rep_hi", type=float, default=0.18)
    ap.add_argument("--h1_low", type=float, default=0.35)
    ap.add_argument("--drift_low", type=float, default=0.08)
    ap.add_argument("--drift_mid", type=float, default=0.20)
    ap.add_argument("--bloom_h2_rep_max", type=float, default=None,
                    help="Only treat H2 spike segments as Bloom when rep_ngram <= this. Default: rep_hi*0.8.")
    ap.add_argument("--bloom_h2_h1_min", type=float, default=None,
                    help="Only treat H2 spike segments as Bloom when H1_norm >= this. Default: h1_low.")

    # alignment thresholds
    ap.add_argument("--coverage_min", type=float, default=0.60, help="min lp_tokens/text_tokens to attempt segment H2 mapping")
    ap.add_argument("--align_min_suffix", type=int, default=24, help="min suffix tokens for alignment")

    # outputs
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save_tokens", action="store_true", help="store tokens/logprobs arrays in JSON (can be large)")
    ap.add_argument("--save_raw", action="store_true", help="store raw provider response under report['raw']")
    ap.add_argument("--print_segments", type=int, default=14)

    # plots
    ap.add_argument("--plot_dir", type=str, default=None)
    ap.add_argument("--plot_prefix", type=str, default="sr_entropy")

    # OpenAI config
    ap.add_argument("--openai_api_key", type=str, default=None)
    ap.add_argument("--openai_base_url", type=str, default=None)

    # Mistral config
    ap.add_argument("--mistral_api_key", type=str, default=None)
    ap.add_argument("--mistral_base_url", type=str, default=None)

    # Gemini config
    ap.add_argument("--gemini_api_key", type=str, default=None)
    ap.add_argument("--gemini_base_url", type=str, default=None)

    # Local HF config
    ap.add_argument("--local_model", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])

    args = ap.parse_args()

    if args.dyn_spike_z is None:
        args.dyn_spike_z = float(args.spike_z)
    if args.dyn_full_z is None:
        args.dyn_full_z = float(args.dyn_spike_z) + 2.0
    if float(args.dyn_full_z) < float(args.dyn_spike_z) + EPS:
        args.dyn_full_z = float(args.dyn_spike_z) + 2.0
    if args.dyn_top_k_disable_z is None:
        args.dyn_top_k_disable_z = float(args.dyn_full_z)
    if args.dyn_slope_z is None:
        args.dyn_slope_z = float(args.dyn_spike_z)
    if args.bloom_h2_rep_max is None:
        args.bloom_h2_rep_max = float(args.rep_hi) * 0.8
    if args.bloom_h2_h1_min is None:
        args.bloom_h2_h1_min = float(args.h1_low)

    prompt = read_text_arg(args.prompt, args.prompt_file)
    system = read_text_arg(args.system, args.system_file)

    if prompt is None:
        print("Need --prompt or --prompt_file", file=sys.stderr)
        sys.exit(2)
    if args.provider != "local" and not args.model:
        print("Need --model for remote providers", file=sys.stderr)
        sys.exit(2)

    prov = get_provider(args.provider)

    t0 = time.time()
    if args.dyn_sampling:
        dyn_mode = str(getattr(args, "dyn_mode", "h2")).lower()
        if dyn_mode == "h2" and args.provider != "local":
            if not args.want_logprobs:
                print("dyn_mode=h2 requires --want_logprobs (provider logprobs).", file=sys.stderr)
                sys.exit(2)
            if args.provider == "mistral":
                print("dyn_mode=h2 is not supported for --provider mistral (logprobs unavailable in this runner).", file=sys.stderr)
                sys.exit(2)
        if dyn_mode == "teacher" and not getattr(args, "score_provider", None):
            print("dyn_mode=teacher requires --score_provider (and model settings).", file=sys.stderr)
            sys.exit(2)

        if args.provider == "local" and dyn_mode == "h2":
            gr = _generate_local_dynamic(prompt=prompt, system=system, args=args)
        else:
            gr = _generate_remote_dynamic(prov=prov, prompt=prompt, system=system, args=args)
    else:
        gr = prov.generate(prompt=prompt, system=system, args=args)
    gen_dt = time.time() - t0

    # --- Structural + alignment token streams
    struct_tokens, struct_tok_name = tokenize_text_struct(gr.text, mode=args.struct_tok_mode)
    align_tokens, align_tok_name = tokenize_text(gr.text, mode=args.align_tok_mode)
    seg_ix, seg_mode_name = segment_indices(struct_tokens, args)

    # --- Provider logprobs (if present)
    lp_tokens = None
    lp_ln = None
    if gr.token_strs and gr.token_logprobs_ln and len(gr.token_strs) == len(gr.token_logprobs_ln):
        lp_tokens = gr.token_strs
        lp_ln = gr.token_logprobs_ln

    coverage_struct = (len(lp_tokens) / len(struct_tokens)) if (lp_tokens is not None and len(struct_tokens) > 0) else None
    coverage_align = (len(lp_tokens) / len(align_tokens)) if (lp_tokens is not None and len(align_tokens) > 0) else None
    token_coverage = coverage_align if (coverage_align is not None and align_tok_name.startswith("tiktoken")) else coverage_struct

    # --- Measurement series for H2 (provider logprobs or teacher)
    measure_mode = None
    measure_tokens = None
    measure_logprobs_ln = None
    measure_coverage = None
    measure_meta = None
    if lp_tokens is not None and lp_ln is not None:
        measure_mode = "provider"
        measure_tokens = lp_tokens
        measure_logprobs_ln = lp_ln
        measure_coverage = token_coverage
        if isinstance(gr.meta, dict) and isinstance(gr.meta.get("measure"), dict):
            m = gr.meta.get("measure") or {}
            measure_meta = m
            mmode = m.get("mode")
            if mmode:
                measure_mode = str(mmode)
            cov = m.get("coverage")
            if cov is not None:
                measure_coverage = float(cov)
    elif isinstance(gr.meta, dict) and isinstance(gr.meta.get("measure"), dict):
        m = gr.meta.get("measure") or {}
        mtoks = m.get("tokens")
        mln = m.get("logprobs_ln")
        if isinstance(mtoks, list) and isinstance(mln, list) and len(mtoks) == len(mln):
            try:
                measure_tokens = [str(x) for x in mtoks]
                measure_logprobs_ln = [float(x) for x in mln]
            except Exception:
                measure_tokens = None
                measure_logprobs_ln = None
            else:
                measure_meta = m
                measure_mode = str(m.get("mode") or "measure")
                cov = m.get("coverage")
                measure_coverage = float(cov) if cov is not None else None

    if measure_mode is not None and measure_mode != "provider":
        token_coverage = None

    token_h2_bits = None
    token_h2_rm = None
    token_h2_jump_z = None
    h2_spikes_lp = []
    H2_global_bits = None

    if measure_logprobs_ln is not None and len(measure_logprobs_ln) > 0:
        token_h2_bits, token_h2_rm, token_h2_jump_z, h2_spikes_lp, H2_global_bits = compute_h2_token_series(
            measure_logprobs_ln, h2_win=args.h2_win, spike_z=args.spike_z
        )

    # --- Segment-level H2 overlay: prefer char-offset alignment; fallback to suffix match
    aligned_logprobs_ln = None
    mapped_h2_spikes_text: List[int] = []
    suffix_match_len = 0
    aligned_ok = False
    align_method = None
    char_align_info: Optional[Dict[str, Any]] = None

    seg_h2avg_override_bits = None
    h2_spike_segs_override = None

    text_spans = token_spans_in_text(gr.text, struct_tokens, struct_tok_name)

    if (measure_mode == "teacher") and isinstance(measure_meta, dict) and (measure_logprobs_ln is not None) and (text_spans is not None):
        offs = measure_meta.get("char_offsets")
        if isinstance(offs, list):
            tmp: List[Tuple[int, int]] = []
            ok = True
            for o in offs:
                if o is None or len(o) != 2:
                    ok = False
                    break
                tmp.append((int(o[0]), int(o[1])))
            if ok and len(tmp) == len(measure_logprobs_ln):
                seg_h2avg_override_bits, h2_spike_segs_override = segment_h2_from_char_offsets(
                    text=gr.text,
                    seg_ix=seg_ix,
                    text_token_spans=text_spans,
                    measure_char_offsets=tmp,
                    measure_logprobs_ln=measure_logprobs_ln,
                    h2_spikes_meas_idx=h2_spikes_lp,
                )
                if h2_spikes_lp:
                    spike_chars = [tmp[i][0] for i in h2_spikes_lp if 0 <= i < len(tmp)]
                    mapped_h2_spikes_text = map_spike_chars_to_text_tokens(spike_chars, text_spans)
                aligned_ok = True
                align_method = "char_offsets:teacher"

    if (seg_h2avg_override_bits is None) and (measure_mode == "provider") and (measure_tokens is not None) and (measure_logprobs_ln is not None) and (text_spans is not None):
        ca = align_measure_series_to_text(gr.text, measure_tokens, measure_logprobs_ln, h2_spikes_lp)
        if ca is not None:
            seg_h2avg_override_bits, h2_spike_segs_override = segment_h2_from_char_offsets(
                text=gr.text,
                seg_ix=seg_ix,
                text_token_spans=text_spans,
                measure_char_offsets=ca.char_offsets,
                measure_logprobs_ln=ca.logprobs_ln,
                h2_spikes_meas_idx=ca.spikes_idx,
            )
            if ca.spikes_idx:
                spike_chars = [ca.char_offsets[i][0] for i in ca.spikes_idx if 0 <= i < len(ca.char_offsets)]
                mapped_h2_spikes_text = map_spike_chars_to_text_tokens(spike_chars, text_spans)
            aligned_ok = True
            align_method = f"char_offsets:{ca.info.get('method', 'provider')}"
            char_align_info = dict(ca.info)

    if (seg_h2avg_override_bits is None) and (measure_mode == "provider") and (lp_tokens is not None) and (lp_ln is not None) and (token_coverage is not None) and (token_coverage >= args.coverage_min):
        ar = align_by_longest_common_suffix(
            text_tokens=struct_tokens,
            lp_tokens=lp_tokens,
            lp_logprobs_ln=lp_ln,
            h2_spikes_lp_idx=h2_spikes_lp,
            min_suffix_tokens=args.align_min_suffix
        )
        suffix_match_len = ar.suffix_match_len
        if suffix_match_len >= args.align_min_suffix:
            aligned_logprobs_ln = ar.aligned_logprobs_ln
            mapped_h2_spikes_text = ar.mapped_h2_spikes_text_idx
            aligned_ok = True
            align_method = "suffix"

    # --- Drift spikes from structural segments
    seg_H1n_for_drift = []
    for (s, e) in seg_ix:
        stoks = struct_tokens[s:e]
        sc = unigram_counts(stoks)
        VH = len(sc)
        H1s = shannon_entropy_from_counts(sc)
        seg_H1n_for_drift.append(norm_entropy(H1s, max(VH, 2)))

    drift_vals, drift_z, drift_spikes_seg = compute_drift_spikes(seg_H1n_for_drift, spike_z=args.spike_z)

    # --- Build reports
    g, segs, extra = build_reports(
        text=gr.text,
        text_tokens=struct_tokens,
        tokenizer_name=struct_tok_name,
        seg_ix=seg_ix,
        drift_spikes_seg=drift_spikes_seg,
        drift_vals=drift_vals,
        drift_z=drift_z,
        aligned_logprobs_ln=aligned_logprobs_ln,
        mapped_h2_spikes_text=mapped_h2_spikes_text,
        H2_global_bits=H2_global_bits,
        seg_h2avg_override_bits=seg_h2avg_override_bits,
        h2_spike_segs_override=h2_spike_segs_override,
        args=args
    )
    g.h2_spikes_lp = h2_spikes_lp
    g.alignment.update({
        "coverage": token_coverage,
        "coverage_struct": coverage_struct,
        "coverage_align": coverage_align,
        "suffix_match_len": suffix_match_len,
        "coverage_min": args.coverage_min,
        "align_min_suffix": args.align_min_suffix,
        "aligned_ok": aligned_ok,
        "method": align_method,
        "char_align": char_align_info,
        "struct_tokenizer": struct_tok_name,
        "align_tokenizer": align_tok_name,
    })

    # --- Console summary
    print("=== SR Entropy Mix Runner v3 ===")
    print(f"provider       : {args.provider}")
    print(f"model          : {args.model if args.provider!='local' else args.local_model}")
    print(f"segment_mode   : {seg_mode_name}")
    print(f"t_gen(s)       : {gen_dt:.3f}")
    if isinstance(gr.meta, dict) and gr.meta.get("dynamic_sampling"):
        ds = gr.meta.get("dynamic_sampling") or {}
        evs = ds.get("events") or []
        mode = ds.get("mode") or "?"
        budget_unit = ds.get("budget_unit") or ds.get("step_unit") or "?"
        measure_unit = ds.get("measure_unit") or "?"
        print(f"dyn_sampling   : on   mode={mode}  events={len(evs)}  budget={budget_unit}  measure={measure_unit}")
    print(f"struct_tokens  : {len(struct_tokens)}  (tokenizer={g.tokenizer})")
    print(f"align_tokens   : {len(align_tokens)}  (tokenizer={align_tok_name})")
    print(f"lp_tokens      : {len(lp_tokens) if lp_tokens is not None else 0}")
    print(f"coverage       : {token_coverage}")
    if coverage_struct is not None and (coverage_align is not None) and coverage_align != coverage_struct:
        print(f"coverage_alt   : struct={coverage_struct:.3f}  align={coverage_align:.3f}")
    if measure_mode and measure_mode != "provider":
        cov_label = "coverage"
        if isinstance(measure_meta, dict) and bool(measure_meta.get("coverage_is_approx")):
            cov_label = "coverage≈"
        print(f"measure        : {measure_mode}  tokens={len(measure_tokens) if measure_tokens is not None else 0}  {cov_label}={measure_coverage}")
    print(f"alignment      : {align_method or 'none'}   suffix_match={suffix_match_len}   (aligned_ok={aligned_ok})")
    print(f"H1_norm        : {g.H1_norm:.4f}   H1_bits={g.H1_bits:.3f}  V={g.vocab_unique}")
    print(f"rep_ngram      : {g.rep_ngram:.4f}   (n={args.ngram_n})")
    if g.H2_bits_per_tok is not None:
        src = measure_mode or "logprobs"
        print(f"H2(bits/tok)   : {g.H2_bits_per_tok:.4f}   (from {src})")
    else:
        print("H2(bits/tok)   : (none)")
    print(f"drift_mean     : {g.drift_mean:.4f}   drift_max={g.drift_max:.4f}")
    print(f"drift_spikes   : {g.drift_spikes}")
    print(f"h2_spikes_lp   : {g.h2_spikes_lp[:30]}{'…' if len(g.h2_spikes_lp)>30 else ''}")
    print(f"h2_spikes_text : {g.h2_spikes_text[:30]}{'…' if len(g.h2_spikes_text)>30 else ''}")
    print(f"bloom_segments : {g.bloom_segments}")
    print()

    # segments
    k = min(args.print_segments, len(segs))
    print(f"--- segments (show {k}/{len(segs)}) ---")
    for s in segs[:k]:
        d = "—" if s.drift is None else f"{s.drift:.3f}"
        dz = "—" if s.drift_z is None else f"{s.drift_z:.2f}"
        h2 = "—" if s.H2_avg_bits is None else f"{s.H2_avg_bits:.3f}"
        print(f"[{s.i:02d}] tok[{s.start}:{s.end}] n={s.n_tok:4d} "
              f"phase={s.phase:7s} bloom={int(s.bloom)} "
              f"H1n={s.H1_norm:.3f} rep={s.rep_ngram:.3f} drift={d:>6} z={dz:>5} H2={h2:>6} | {s.preview}")

    print("\n--- output text (first 1200 chars) ---")
    print((gr.text[:1200] + ("…" if len(gr.text) > 1200 else "")))
    print()

    # plots
    if args.plot_dir:
        ensure_dir(args.plot_dir)

        p1 = os.path.join(args.plot_dir, f"{args.plot_prefix}_segments.png")
        plot_segments(segs, p1)

        p2 = os.path.join(args.plot_dir, f"{args.plot_prefix}_phase.png")
        plot_phase_strip(segs, p2)

        p3 = os.path.join(args.plot_dir, f"{args.plot_prefix}_phase_space.png")
        plot_phase_space(segs, p3)

        if token_h2_rm is not None:
            p4 = os.path.join(args.plot_dir, f"{args.plot_prefix}_token_h2.png")
            plot_token_surprisal(token_h2_rm, h2_spikes_lp, p4)

        print(f"plots written under: {args.plot_dir}")

    measure_summary = None
    if measure_mode is not None:
        measure_summary = {
            "mode": measure_mode,
            "n_tokens": int(len(measure_tokens) if measure_tokens is not None else 0),
            "coverage": measure_coverage,
        }
        if isinstance(measure_meta, dict):
            for k in ("score_provider", "score_model", "score_local_model"):
                v = measure_meta.get(k)
                if v:
                    measure_summary[k] = v
            for k in ("coverage_is_approx", "coverage_basis", "score_stride_tokens", "score_sample_every", "char_offsets_unit"):
                v = measure_meta.get(k)
                if v is not None:
                    measure_summary[k] = v
            if isinstance(measure_meta.get("char_offsets"), list):
                measure_summary["char_offsets_count"] = int(len(measure_meta.get("char_offsets") or []))

    # JSON report
    report: Dict[str, Any] = {
        "generated": {
            "provider": args.provider,
            "model": (args.model if args.provider != "local" else args.local_model),
            "text": gr.text,
            "gen_seconds": gen_dt,
        },
        "global": asdict(g),
        "segments": [asdict(s) for s in segs],
        "extra": {
            "segment_mode": seg_mode_name,
            "segment_count": len(seg_ix),
            "sampling": {
                "max_out": int(args.max_out),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            },
            "segmentation": {
                "endchars": args.endchars,
                "min_seg_tokens": args.min_seg_tokens,
                "max_seg_tokens": args.max_seg_tokens,
                "window_tokens": args.window_tokens,
                "rich_flags": {
                    "newline_split": not args.no_newline_split,
                    "bullet_split": not args.no_bullet_split,
                    "quote_split": not args.no_quote_split,
                    "parenclose_split": not args.no_parenclose_split,
                }
            },
            "spike": {"z": args.spike_z, "h2_win": args.h2_win},
            **extra
        }
    }
    if isinstance(gr.meta, dict) and gr.meta.get("dynamic_sampling") is not None:
        report["extra"]["dynamic_sampling"] = gr.meta.get("dynamic_sampling")
    if measure_summary is not None:
        report["extra"]["measure"] = measure_summary

    if args.save_tokens:
        report["tokens"] = {
            "struct_tokens": struct_tokens,
            "align_tokens": align_tokens,
            "struct_tokenizer": struct_tok_name,
            "align_tokenizer": align_tok_name,
            "lp_tokens": lp_tokens,
            "lp_logprobs_ln": lp_ln,
            "aligned_logprobs_ln": aligned_logprobs_ln,
            "token_h2_bits": token_h2_bits,
            "token_h2_rm": token_h2_rm,
            "token_h2_jump_z": token_h2_jump_z,
        }
        if (lp_tokens is None or lp_ln is None) and measure_mode is not None and measure_mode != "provider" and measure_tokens is not None and measure_logprobs_ln is not None:
            report["tokens"]["measure"] = {
                "mode": measure_mode,
                "tokens": measure_tokens,
                "logprobs_ln": measure_logprobs_ln,
                "coverage": measure_coverage,
            }
            if isinstance(measure_meta, dict):
                for k in ("score_provider", "score_model", "score_local_model"):
                    v = measure_meta.get(k)
                    if v:
                        report["tokens"]["measure"][k] = v

    if args.save_raw:
        report["raw"] = gr.meta.get("raw") if isinstance(gr.meta, dict) else gr.meta

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote: {args.out}")

if __name__ == "__main__":
    main()
