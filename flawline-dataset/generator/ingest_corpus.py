"""Ingest REAL recorded baselines from VCTK and Svarah (see real_sources.yaml). Usage: python ingest_corpus.py [B01 ...]

Real audio is never altered beyond: first-N-utterance selection, silence trimmed to 0.3 s margins,
resample to 22.05 kHz mono, and a single gain to -3 dBFS peak. No noise floor is added.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from scipy.signal import resample_poly

from build_baselines import align
from common import BASELINES, GEN, SR, TAKES, db, dump_json, frame_rms_db, load_yaml, pink_noise, s2n, undb, write_audio

API = "https://datasets-server.huggingface.co/rows?dataset=sanchit-gandhi/vctk&config=default&split=train"
MARGIN = 0.3


def band(age: int) -> str:
    return "18-25" if age <= 25 else "26-45" if age <= 45 else "46-65" if age <= 65 else "65+"


def get_rows(off: int, n: int):
    for _ in range(5):
        r = requests.get(f"{API}&offset={off}&length={n}", timeout=120)
        if r.ok:
            return r.json()["rows"]
    r.raise_for_status()


def trim(x: np.ndarray, sr: int) -> np.ndarray:
    fr = frame_rms_db(x, hop=int(sr * 0.01), win=int(sr * 0.02))
    act = np.where(fr > fr.max() - 40)[0]
    a = max(0, act[0] * int(sr * 0.01) - int(MARGIN * sr))
    b = min(len(x), (act[-1] + 2) * int(sr * 0.01) + int(MARGIN * sr))
    return x[a:b]


def write_take(slot, full, text, meta, prov):
    full = full / (np.abs(full).max() + 1e-9) * undb(-3.0)
    words = align(text, full)
    take = f"{slot}-CHAMP"
    write_audio(TAKES / f"{take}_C0.flac", full)
    dump_json(TAKES / f"{take}_C0.align.json", {
        "take_id": take, "sr": SR, "duration_s": round(len(full) / SR, 4), "align_source": "whisper-base.en (matched to corpus transcript)",
        "words": [{"i": w["i"], "w": w["text"], "clean": w["clean"], "punct": w["punct"], "sent_end": w["sent_end"],
                   "clause_end": w["clause_end"], "start_s": round(w["start"] / SR, 4), "end_s": round(w["end"] / SR, 4)} for w in words]})
    d = BASELINES / slot
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.txt").write_text(text + "\n")
    dump_json(d / "source.json", {**meta, "status": "real_corpus", "voice": None, "take_id": take, **prov})
    print(f"{slot} {prov['speaker_id']} {meta['gender']} {meta['age']:>6} {meta['accent']:28s} {len(words)} words {len(full)/SR:.1f}s")


def band(age) -> str:
    if isinstance(age, str):                       # Svarah age group
        return {"18-30": "18-25", "30-45": "26-45", "45-60": "46-65", "60+": "65+"}[age]
    return "18-25" if age <= 25 else "26-45" if age <= 45 else "46-65" if age <= 65 else "65+"


def ingest_vctk(slot, speaker, cfg):
    spk = pd.read_csv(GEN / "vctk_speakers.csv", index_col=0, keep_default_na=False).loc[speaker]
    rows = get_rows(int(spk.start), 100)          # mic1/mic2 interleaved
    rows = sorted([r["row"] for r in rows if r["row"]["file"].endswith("_mic1.flac")], key=lambda r: r["text_id"])
    pieces, texts, total = [], [], 0.0
    for r in rows:
        x, sr = sf.read(io.BytesIO(requests.get(r["audio"][0]["src"], timeout=120).content), dtype="float32")
        x = trim(x if x.ndim == 1 else x.mean(axis=1), sr)
        x = resample_poly(x, 147, 320)               # 48 kHz -> 22.05 kHz
        pieces.append(x)
        texts.append(" ".join(r["text"].split()))
        total += len(x) / SR
        if total >= cfg["target_s"][0]:
            break
    age = int(spk.age)
    c = cfg["corpora"]["vctk"]
    meta = {"id": slot, "title": f"VCTK {speaker} · elicitation paragraph + Rainbow Passage", "genre": "interpretive reading",
            "gender": spk.gender, "age_band": band(age), "age": str(age),
            "origin": spk.accent + (f" ({spk.region.strip()})" if spk.region.strip() else ""), "accent": spk.accent,
            "register": "neutral", "slang_terms": []}
    prov = {"corpus": c["name"], "speaker_id": speaker, "licence": c["licence"], "url": c["url"], "citation": c["citation"],
            "note": c["note"] + " Contiguous first utterances, silence trimmed to 0.3 s margins."}
    write_take(slot, np.concatenate(pieces), " ".join(texts), meta, prov)
    return meta


_SV = None


def svarah_table():
    global _SV
    if _SV is None:
        import glob
        import pyarrow.parquet as pq
        fs = sorted(glob.glob(str(GEN / ".cache" / "svarah" / "data" / "*.parquet")))
        if not fs:
            sys.exit("Svarah shards missing: huggingface-cli login, accept ai4bharat/Svarah terms, then "
                     "hf_hub_download the 3 data/*.parquet files into generator/.cache/svarah")
        d = pd.concat([pq.read_table(f).to_pandas() for f in fs], ignore_index=True)
        d["path"] = d.audio_filepath.map(lambda a: a["path"])
        m = d.path.str.extract(r"^(\d+)_(\w+?)_chunk_(\d+)\.wav$")
        d["rec"], d["spk"], d["chunk"] = m[0], m[1], m[2].astype(int)
        _SV = d
    return _SV


def ingest_svarah(slot, speaker, cfg):
    s = svarah_table()
    s = s[s.spk == speaker].sort_values("chunk")
    assert len(s) == s.chunk.max() + 1, "chunks not contiguous"
    first = s.iloc[0]
    rng = np.random.default_rng(int(speaker[1:]))
    pieces, texts, total = [], [], 0.0
    for _, r in s.iterrows():
        x, sr = sf.read(io.BytesIO(r.audio_filepath["bytes"]), dtype="float32")
        x = resample_poly(x if x.ndim == 1 else x.mean(axis=1), 441, 320) if sr == 16000 else x
        if pieces:                                    # chunks were cut at pauses: restore a short pause of the speaker's own floor
            fl = np.sort(frame_rms_db(x))[: max(1, len(x) // 441 // 20)].mean()
            pieces.append(pink_noise(s2n(0.35), rng) * undb(max(fl, -70.0)))
        pieces.append(x)
        texts.append(" ".join(str(r.text).split()))
        total += len(x) / SR
        if total >= cfg["target_s"][0]:
            break
    grp = first["age-group"]
    lang, state = first.primary_language, first.native_place_state
    c = cfg["corpora"]["svarah"]
    meta = {"id": slot, "title": f"Svarah {speaker} · spontaneous monologue", "genre": "extemporaneous",
            "gender": "F" if first.gender == "Female" else "M", "age_band": band(grp), "age": f"{grp} (group)",
            "origin": f"{state}, India", "accent": f"Indian English ({lang})", "register": "colloquial", "slang_terms": []}
    prov = {"corpus": c["name"], "speaker_id": speaker, "licence": c["licence"], "url": c["url"], "citation": c["citation"],
            "native_language": lang, "recording_id": first.rec,
            "note": c["note"] + " Consecutive chunks joined with 0.35 s of the speaker's own noise floor where chunks were cut."}
    write_take(slot, np.concatenate(pieces), " ".join(texts), meta, prov)
    return meta


if __name__ == "__main__":
    cfg = load_yaml(GEN / "real_sources.yaml")
    want = set(sys.argv[1:])
    metas = []
    for slot, v in cfg["slots"].items():
        if not want or slot in want:
            metas.append({"vctk": ingest_vctk, "svarah": ingest_svarah}[v["corpus"]](slot, v["speaker"], cfg))
    import yaml
    if not want:
        (GEN / "baselines.yaml").write_text("# generated by ingest_corpus.py from real_sources.yaml\n" + yaml.safe_dump({"baselines": metas}, sort_keys=False))
