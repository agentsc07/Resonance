"""Write the product name from app/brand.json into the README headline (between the BRAND markers). Run after changing the name: python scripts/apply_brand.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
b = json.loads((ROOT / "app" / "brand.json").read_text())
p = ROOT / "README.md"
s = p.read_text()
block = f"<!-- BRAND:start -->\n# {b['product']}\n\n**{b['tagline']}**" + (f" The product built on the {b['dataset']} dataset and benchmark." if b["product"] != b["dataset"] else "") + "\n<!-- BRAND:end -->"
if "<!-- BRAND:start -->" in s:
    s = re.sub(r"<!-- BRAND:start -->.*?<!-- BRAND:end -->", block, s, flags=re.S)
else:
    s = s.replace("# Flawline\n", block + "\n", 1)
p.write_text(s)
print("README headline:", b["product"])
