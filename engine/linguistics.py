"""Transcript intelligence: where in the sentence a pause, filler, repeat or skip would actually happen.

Every word boundary k (between word k and k+1) gets a type from the dependency parse:

  sentence         full stop / ? / !                               a real pause belongs here
  clause_punct     comma / semicolon / colon                       a real pause belongs here
  before_coord     next word is and / or / but                     speakers hesitate or drag right here
  before_subord    next word is that / because / when / if ...     clause opener: classic hesitation point
  before_relative  next word is which / who / where ...            clause opener
  after_discourse  previous word is well / so / okay / yes ...     "well... um"
  phrase_tight     inside a phrase (the|people, of|the, to|go)     a SILENT pause here is a delivery flaw (PAUSE_BAD)
  phrase_loose     between phrases (subject|verb, noun|prep-phrase) mild: a hesitation is plausible

Words also get coarse POS, function/content flags and a stress-neutral "reducible" flag (the/a/to/of/and: vowels that
can be drawn out into a drawl).
"""
from __future__ import annotations

import re
from functools import lru_cache

FUNC_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PRON", "PART"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
DISCOURSE = {"well", "so", "okay", "ok", "yes", "now", "look", "right", "anyway", "actually", "basically", "honestly", "listen"}
TIGHT_DEP = {"det", "amod", "poss", "nummod", "compound", "predet", "quantmod", "neg", "aux", "auxpass", "prt", "nmod", "case", "mark"}
REDUCIBLE = {"the", "a", "an", "to", "of", "and", "but", "that", "for", "in", "on", "at", "or", "so", "as", "i", "we", "it"}

# how natural a HESITATION is at each boundary type (weights for sampling; 0 = never)
HESITATION_WEIGHT = {"sentence": 6.0, "before_coord": 6.0, "before_subord": 5.0, "clause_punct": 4.0, "before_relative": 3.0,
                     "after_discourse": 3.0, "phrase_loose": 0.2, "phrase_tight": 0.0}


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def annotate(words: list[dict]) -> dict:
    """words: alignment words with key "w" (text incl. punctuation). Returns {"w": [...], "b": [...]}.
    w[i]: pos, dep, tag, function, content, reducible.  b[k]: boundary type between word k and k+1."""
    text, spans = "", []
    for w in words:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(w["w"])))
        text += w["w"]
    doc = _nlp()(text)
    first, last = {}, {}
    for t in doc:
        if t.is_punct or t.is_space:
            continue
        for i, (a, b) in enumerate(spans):
            if a <= t.idx < b:
                first.setdefault(i, t)
                last[i] = t
                break
    out = []
    for i, w in enumerate(words):
        t = first.get(i)
        clean = re.sub(r"[^a-z']", "", w["w"].lower())
        if t is None:
            out.append({"pos": "X", "dep": "", "tag": "", "function": False, "content": False, "reducible": False, "lemma": clean})
        else:
            out.append({"pos": t.pos_, "dep": t.dep_, "tag": t.tag_, "function": t.pos_ in FUNC_POS, "content": t.pos_ in CONTENT_POS,
                        "reducible": clean in REDUCIBLE, "lemma": t.lemma_.lower()})
    bounds = []
    for k in range(len(words) - 1):
        w = words[k]
        pv, nx = last.get(k), first.get(k + 1)
        if w.get("sent_end"):
            bounds.append("sentence")
        elif w.get("clause_end"):
            bounds.append("clause_punct")
        elif pv is None or nx is None:
            bounds.append("phrase_loose")
        elif nx.dep_ == "cc" or nx.lower_ in ("and", "or", "but"):
            bounds.append("before_coord")
        elif nx.tag_ in ("WDT", "WP", "WRB", "WP$") and nx.lower_ not in ("when",):
            bounds.append("before_relative")
        elif nx.dep_ == "mark" or nx.pos_ == "SCONJ":
            bounds.append("before_subord")
        elif pv.lower_ in DISCOURSE and (pv.i == 0 or pv.dep_ in ("intj", "advmod", "discourse")):
            bounds.append("after_discourse")
        elif (pv.dep_ in TIGHT_DEP and pv.head.i >= nx.i) or (pv.pos_ == "ADP" and nx.i in [d.i for d in pv.subtree]) \
                or (pv.pos_ in ("DET", "AUX", "PART") and pv.i < nx.i):
            bounds.append("phrase_tight")
        else:
            bounds.append("phrase_loose")
    return {"w": out, "b": bounds}


def context(words: list[dict], k0: int, k1: int | None = None, width: int = 3, mark: str = "⟦…⟧") -> str:
    """Readable snippet for labels, e.g. 'act as a ⟦…⟧ prism and'. Insertion between k0 and k0+1 if k1 is None."""
    n = len(words)
    if k1 is None:
        left = " ".join(w["w"] for w in words[max(0, k0 - width + 1): k0 + 1])
        right = " ".join(w["w"] for w in words[k0 + 1: k0 + 1 + width])
        return f"{left} {mark} {right}".strip()
    left = " ".join(w["w"] for w in words[max(0, k0 - width): k0])
    mid = " ".join(w["w"] for w in words[k0: k1 + 1])
    right = " ".join(w["w"] for w in words[k1 + 1: k1 + 1 + width])
    return f"{left} [{mid}] {right}".strip()


def describe(kind: str, btype: str, prev: str, nxt: str) -> str:
    return {
        "sentence": f"after the full stop following '{prev}'",
        "clause_punct": f"at the comma after '{prev}'",
        "before_coord": f"before the conjunction '{nxt}'",
        "before_subord": f"before the clause opener '{nxt}'",
        "before_relative": f"before the relative word '{nxt}'",
        "after_discourse": f"after the discourse marker '{prev}'",
        "phrase_tight": f"inside a tight phrase, between '{prev}' and '{nxt}'",
        "phrase_loose": f"between phrases, '{prev}' | '{nxt}'",
    }.get(btype, btype)
