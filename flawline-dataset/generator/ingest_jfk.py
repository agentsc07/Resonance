"""Ingest two excerpts of President Kennedy's Rice University address (12 Sep 1962) as extra baselines B09 and B10.
Source: Rice University's copy on the Internet Archive (item president-john-f.-kennedy-09-12-1962, Public Domain Mark 1.0; the JFK Library lists
the recording JFKWHA-127-002 as public domain). The full 21-minute audio is NOT stored in the repo, only the two ~60 s excerpts under takes/.
   python ingest_jfk.py /tmp/jfk/rice16k.wav        (16 kHz mono extracted from the 720p mp4 with ffmpeg)
Text is the speech as delivered (checked against an ASR transcript, hesitations of the original not transcribed)."""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

from common import GEN, SR
from ingest_corpus import trim, write_take

URL = "https://archive.org/details/president-john-f.-kennedy-09-12-1962"
EXCERPTS = {
    "B09": (88.0, 154.0, "We meet at a college noted for knowledge, in a city noted for progress, in a state noted for strength, and we stand in need of all three. "
            "For we meet in an hour of change and challenge, in a decade of hope and fear, in an age of both knowledge and ignorance. "
            "The greater our knowledge increases, the greater our ignorance unfolds. "
            "Despite the striking fact that most of the scientists that the world has ever known are alive and working today, "
            "despite the fact that this nation's own scientific manpower is doubling every twelve years, in a rate of growth more than three times "
            "that of our population as a whole, despite that, the vast stretches of the unknown and the unanswered and the unfinished still far outstrip "
            "our collective comprehension.", "Rice University address, 12 Sep 1962 (excerpt 1: 'a college noted for knowledge')"),
    "B10": (634.9, 697.0, "In the last twenty four hours we have seen facilities now being created for the greatest and most complex exploration in man's history. "
            "We have felt the ground shake and the air shattered by the testing of a Saturn C1 booster rocket, many times as powerful as the Atlas which "
            "launched John Glenn, generating power equivalent to ten thousand automobiles with their accelerators on the floor. "
            "We have seen the site where five F1 rocket engines, each one as powerful as all eight engines of the Saturn combined, will be clustered "
            "together to make the advanced Saturn missile, assembled in a new building to be built at Cape Canaveral, as tall as a forty eight story "
            "structure, as wide as a city block, and as long as two lengths of this field.", "Rice University address, 12 Sep 1962 (excerpt 2: 'the last twenty four hours')"),
}


def main(wav: str):
    x16, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    entries = []
    for slot, (t0, t1, text, title) in EXCERPTS.items():
        seg = x16[int(t0 * sr): int(t1 * sr)]
        seg = resample_poly(seg, 441, 320).astype(np.float32)            # 16 kHz -> 22.05 kHz
        seg = trim(seg, SR)
        meta = {"id": slot, "title": title, "genre": "persuasive oratory", "gender": "M", "age_band": "26-45", "age": "45",
                "origin": "American (Massachusetts)", "accent": "American (New England)", "register": "formal oratory", "slang_terms": []}
        prov = {"corpus": "JFK Rice University address (1962)", "speaker_id": "JFK", "licence": "Public domain (US Government work; Public Domain Mark 1.0 on the Rice University copy)",
                "url": URL, "citation": "John F. Kennedy, Address at Rice University on the Nation's Space Effort, 12 September 1962 (JFK Library JFKWHA-127-002).",
                "note": f"1962 open-air speech recording with crowd and PA noise, excerpt {t0:.0f}-{t1:.0f} s of the full address, silence trimmed to 0.3 s margins. "
                        "Not studio quality: used to widen the dataset beyond read-aloud studio speech."}
        write_take(slot, seg, text, meta, prov)
        entries.append(meta)
    p = GEN / "baselines.yaml"
    d = yaml.safe_load(open(p))
    have = {b["id"] for b in d["baselines"]}
    d["baselines"] += [m for m in entries if m["id"] not in have]
    yaml.safe_dump(d, open(p, "w"), sort_keys=False, allow_unicode=True)
    print("baselines.yaml now lists", [b["id"] for b in d["baselines"]])


if __name__ == "__main__":
    main(sys.argv[1])
