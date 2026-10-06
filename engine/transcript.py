"""Turn plain text (a pasted transcript, or what the recogniser heard) into the word records the engine works on:
w (with punctuation), clean, sent_end / clause_end flags. No timings: those come from the participant's audio."""
from __future__ import annotations

import re


def words_from_text(text: str) -> list[dict]:
    out = []
    for tok in text.replace("—", " ").replace("–", " ").replace("--", " ").split():
        clean = re.sub(r"[^a-z']", "", tok.lower())
        if not clean:
            continue
        end = tok.rstrip("\"')]}")
        out.append({"i": len(out), "w": tok, "clean": clean, "punct": "", "sent_end": end.endswith((".", "?", "!")),
                    "clause_end": end.endswith((",", ";", ":")), "start_s": 0.0, "end_s": 0.0})
    return out
