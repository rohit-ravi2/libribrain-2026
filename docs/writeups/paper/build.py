"""Build the leakage paper PDF from docs/writeups/paper_leakage_draft.md (+ repro_appendix.md).
Drops the draft note and the 'To finish before posting' list, appends the reproducibility appendix, runs pandoc+xelatex.
Usage: python build.py  -> docs/writeups/paper/leakage_paper.pdf (and leakage_paper.tex for inspection)"""
import os, re, subprocess
H = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(H)
s = open(os.path.join(W, "paper_leakage_draft.md")).read()
s = re.sub(r"\n\*Draft,.*?\*\n", "\n", s, count=1, flags=re.S)
s = s.split("## To finish before posting")[0].rstrip() + "\n\n" + open(os.path.join(H, "repro_appendix.md")).read()
title = s.splitlines()[0].lstrip("# ").strip(); body = s.split("\n", 1)[1]
author = "Rohit Ravi (New York University)"; body = body.replace(author, "", 1)
meta = f"---\ntitle: \"{title}\"\nauthor: \"Rohit Ravi, New York University\"\ndate: \"November 2026\"\n---\n"
src = os.path.join(H, "leakage_paper.md"); open(src, "w").write(meta + body)
args = ["pandoc", src, "--pdf-engine=xelatex", "-H", os.path.join(H, "header.tex"), "--resource-path", f"{W}:{H}",
        "-V", "geometry:margin=1in", "-V", "fontsize=10pt", "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue"]
for out in ("leakage_paper.tex", "leakage_paper.pdf"):
    subprocess.run(args + ["-s", "-o", os.path.join(H, out)], check=True)
print("built", os.path.join(H, "leakage_paper.pdf"))
