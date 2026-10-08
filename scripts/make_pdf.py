"""docs/technical_report.md -> docs/technical_report.pdf (python-markdown + headless Chrome). Requires Google Chrome; set CHROME to override the path.
   python scripts/make_pdf.py"""
import os
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
body = markdown.markdown((DOCS / "technical_report.md").read_text(), extensions=["tables"])
css = """@page{size:A4;margin:14mm 15mm}body{font:9.8pt/1.42 'Helvetica Neue',Arial,sans-serif;color:#1d1d22}h1{font-size:15.5pt;margin:0 0 3pt}h2{font-size:11.5pt;margin:9pt 0 3pt;border-bottom:1px solid #ccc;padding-bottom:1pt}
p{margin:3pt 0;text-align:justify}table{border-collapse:collapse;width:100%;margin:4pt 0;font-size:8.4pt}th,td{border:1px solid #bbb;padding:2pt 4pt;text-align:left}th{background:#f1ede2}
img{width:100%;max-height:78mm;object-fit:contain;display:block;margin:3pt auto 0}img+em,p>em:only-child{font-size:8pt;color:#555}code{font-size:8.3pt;background:#f3f3f0;padding:0 2pt}"""
html = DOCS / "_report.html"
html.write_text(f"<!doctype html><meta charset='utf-8'><title>Resonance technical report</title><style>{css}</style>{body}")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={DOCS / 'technical_report.pdf'}", html.as_uri()], check=True, capture_output=True)
html.unlink()
print("wrote docs/technical_report.pdf")
