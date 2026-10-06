"""Causal explanations: one template per flaw, filled with the measured numbers and the context words (no free text generation)."""
from __future__ import annotations

from .render_facts import syllables


def words_str(ref_words: list[dict], i0: int, i1: int, width: int = 6) -> str:
    ws = [w["w"].strip(",.;:!?") for w in ref_words[i0: i1 + 1]]
    return " ".join(ws[:width]) + ("…" if len(ws) > width else "")


def explain(c, C, ref_words: list[dict], sev: float) -> tuple[str, str]:
    """Returns (explanation sentence, coaching tip)."""
    ctx = f'Words {c.w0}–{c.w1} "{words_str(ref_words, c.w0, c.w1)}"'
    f = c.flaw
    cat = c.category
    tail = f" ({cat}, severity {sev:.1f})"
    if f in ("PACE_FAST", "PACE_SLOW"):
        syl = sum(syllables(ref_words[i]["clean"]) for i in range(c.w0, c.w1 + 1))
        dur_p = max(C.w[c.w1]["pe"] - C.w[c.w0]["ps"], 1e-3)
        dur_r = max(C.w[c.w1]["re"] - C.w[c.w0]["rs"], 1e-3)
        rp, rr = syl / dur_p, syl / dur_r
        ratio = c.facts.get("ratio", rp / rr)
        verb = "Rushed" if f == "PACE_FAST" else "Dragging"
        return (f"{ctx}: {rp:.1f} syllables/s against {rr:.1f} in the reference ({(rp / rr - 1) * 100:+.0f}%). {verb} delivery{tail}.",
                "Let key phrases land: slow down on the words that carry the point." if f == "PACE_FAST" else "Keep momentum through the phrase; trust your preparation.")
    if f == "PAUSE_BAD":
        return (f'{ctx}: {c.d:.2f} s pause inside a phrase ({c.facts.get("boundary", "")}); the reference does not stop here. Misplaced pause{tail}.',
                "Keep phrases together; pause at commas and full stops, not between a word and its phrase.")
    if f == "PAUSE_LOST":
        return (f"{ctx}: a {c.d:.2f} s pause at a clause boundary was removed. Lost rhetorical pause{tail}.", "Let the point land: pause after a key clause.")
    if f in ("FILLER", "REPEAT", "RARE_HESIT"):
        what = {"FILLER": "Unscripted hesitation sound", "REPEAT": "False start / repeated words", "RARE_HESIT": "Hesitation before a difficult word"}[f]
        return (f'{ctx}: {c.d:.2f} s of inserted speech not in the text. {what}{tail}.', "Rehearse the lines flagged and breathe silently at the boundary instead of filling it.")
    if f == "MONOTONE":
        return (f'{ctx}: pitch range {c.facts.get("iqr_part", 0):.1f} st against {c.facts.get("iqr_ref", 0):.1f} st in the reference. Flat, monotone delivery{tail}.',
                "Mark three key words per sentence and lift your pitch on them.")
    if f == "UPTALK":
        return (f'{ctx}: sentence ends on a rise of {c.facts.get("rise_part", 0):+.1f} st (reference {c.facts.get("rise_ref", 0):+.1f} st). Uptalk on a statement{tail}.',
                "End statements with a falling pitch; save the rise for questions.")
    if f == "EMPH_FLAT":
        return (f"{ctx}: words that carry the meaning lost prominence ({c.d:.1f} z-units below the reference: pitch, loudness and length). Buried emphasis{tail}.",
                "Stress the words that carry the meaning; make them longer, louder and higher.")
    if f == "FADE":
        return (f"{ctx}: the end of the sentence is {c.d:.1f} dB quieter than its start, relative to the reference. Trailing off{tail}.", "Breathe at clause boundaries and finish at full energy.")
    if f == "SHOUT":
        return (f"{ctx}: {c.d:.1f} dB louder than the reference over this stretch. Sudden volume jump{tail}.", "Keep volume steady; build emphasis with pitch and timing too.")
    if f == "SLUR":
        return (f"{ctx}: consonant energy (2–7.5 kHz) is {c.d:.1f} dB below the reference. Slurred articulation{tail}.", "Open the mouth and land the consonants, especially at word ends.")
    if f == "WORD_SKIP":
        return (f"{ctx}: {c.facts.get('n_words', 1)} word(s) missing against the text. Skipped words{tail}.", "Read the line slowly once to check every word is there.")
    if f == "WORD_SWAP":
        return (f"{ctx}: this word does not match the text. Misread word{tail}.", "Check the exact wording; mispronounced or swapped words change the meaning.")
    return (f"{ctx}: {f}{tail}.", "")
