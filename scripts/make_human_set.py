"""Seed flawline-dataset/human/clips with neutral-named clips for the human reference set: 3 unedited readings + 5 hidden injected clips (levels 3-5).
The mapping is written to human/hidden_map.json (the labelling page and its API never read it) and the read-aloud text of each clip to human/transcripts.json
(used by eval/human_agreement.py to analyse a clip the way an upload with a pasted transcript is analysed). Add your own recordings by dropping
WAV/M4A/FLAC files into human/clips/ (and, optionally, their text into transcripts.json under the file name).
   python scripts/make_human_set.py"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "flawline-dataset"
H = D / "human"
UNEDITED = ["B02-CHAMP_C0", "B03-CHAMP_C0", "B09-CHAMP_C0"]
HIDDEN = ["B06-CHAMP_C0__PAUSE_BAD_L3_s5270", "B07-CHAMP_C0__FILLER_L3_s7599", "B04-CHAMP_C0__SHOUT_L3_s5399",
          "B10-CHAMP_C0__PACE_SLOW_L4_s1612", "B01-CHAMP_C0__WORD_SKIP_L5_s5306"]


def text_of(take_id: str) -> str:
    w = json.loads((D / "takes" / f"{take_id.split('__')[0]}.align.json").read_text())["words"]
    return " ".join(x["w"] for x in w)


def main():
    (H / "clips").mkdir(parents=True, exist_ok=True)
    items = [(c, D / "takes" / f"{c}.flac", None) for c in UNEDITED] + [(c, D / "variants" / f"{c}.flac", json.loads((D / "variants" / f"{c}.json").read_text())) for c in HIDDEN]
    random.Random(2026).shuffle(items)
    hidden, texts = {}, {}
    for k, (cid, src, lab) in enumerate(items, 1):
        name = f"h{k:02d}.flac"
        shutil.copyfile(src, H / "clips" / name)
        texts[name] = text_of(cid)
        hidden[name] = {"source_clip": cid, "injected": [{"flaw": r["flaw"], "level": r.get("level"), "start_s": r["start_s"], "end_s": r["end_s"], "category": r["category"]} for r in lab["what"]] if lab else []}
    (H / "hidden_map.json").write_text(json.dumps(hidden, indent=1) + "\n")
    (H / "transcripts.json").write_text(json.dumps(texts, indent=1) + "\n")
    print("wrote", len(items), "clips;", sum(1 for v in hidden.values() if v["injected"]), "hidden injected")


if __name__ == "__main__":
    main()
