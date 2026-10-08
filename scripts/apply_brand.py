"""Write the product name, tagline and release links into the README (between the BRAND and LINKS markers) from app/brand.json and app/links.json.
   python scripts/apply_brand.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
b = json.loads((ROOT / "app" / "brand.json").read_text())
L = json.loads((ROOT / "app" / "links.json").read_text())
p = ROOT / "README.md"
s = p.read_text()


def sub(tag: str, body: str) -> None:
    global s
    s = re.sub(rf"<!-- {tag}:start -->.*?<!-- {tag}:end -->", lambda _: f"<!-- {tag}:start -->\n{body}\n<!-- {tag}:end -->", s, flags=re.S)


sub("BRAND", f"# {b['product']}\n\n**{b['tagline']}**")
item = lambda label, url: f"[{label}]({url})" if url else f"{label} (link at release)"
local = lambda u, d: u if u.startswith("http") else d
sub("LINKS", " · ".join([item("Demo video", L.get("video", "")), item("Dataset download", L.get("dataset", "")),
                         f"[Technical report (PDF)]({local(L.get('report', ''), 'docs/technical_report.pdf')})", f"[Datasheet]({local(L.get('datasheet', ''), 'DATASHEET.md')})"]))
p.write_text(s)
print("README updated:", b["product"])
