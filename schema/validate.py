"""Validate a label/prediction JSON against schema/label.schema.json.   python schema/validate.py file.json [...]"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).parent / "label.schema.json").read_text())
_V = Draft202012Validator(SCHEMA)


def errors(label: dict) -> list[str]:
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in _V.iter_errors(label)]


def validate(label: dict):
    e = errors(label)
    if e:
        raise ValueError("label violates schema 1.1.0:\n  " + "\n  ".join(e[:8]))


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        e = errors(json.loads(Path(p).read_text()))
        print(("OK   " if not e else "FAIL ") + p)
        for line in e[:5]:
            print("   ", line)
        bad += bool(e)
    sys.exit(1 if bad else 0)
