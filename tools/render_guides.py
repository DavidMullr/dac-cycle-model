"""Build the static reports and guides using Pandoc. No model execution."""
from __future__ import annotations

import html
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GUIDES = {
    "PROJECT_MAP.md": "project_map.html",
    "DATA_NEEDED.md": "DATA_NEEDED.html",
    "QUICKSTART.md": "quickstart.html",
    "REVIEW_CHECKS.md": "REVIEW_CHECKS.html",
}


def main():
    pandoc = shutil.which("pandoc")
    if not pandoc:
        quarto = shutil.which("quarto") or shutil.which("quarto.cmd")
        if quarto:
            tools_dir = Path(quarto).resolve().parent / "tools"
            pandoc = next((str(p) for p in tools_dir.rglob("pandoc*") if p.name in {"pandoc", "pandoc.exe"} and p.is_file()), None)
    if not pandoc:
        raise SystemExit("Install Pandoc or Quarto to rebuild the documentation.")
    report = ROOT / "model/report/project"
    for name in ("index", "user_guide", "model_reference", "evidence"):
        source = (report / (name + (".md" if name == "user_guide" else ".qmd"))).read_text(encoding="utf-8")
        if name == "user_guide":
            title, source = source.split("\n", 1)
            source = "---\ntitle: " + title.removeprefix("# ") + "\n---\n" + source
            for filename, rendered in GUIDES.items():
                source = source.replace("/" + filename + ")", "/" + rendered + ")")
            source = source.replace("](model_reference.qmd", "](model_reference.html")
        subprocess.run([pandoc, "--from=markdown", "--to=html5", "--standalone",
                        "--embed-resources", "--mathml", "--toc", ("--toc-depth=3" if name == "model_reference" else "--toc-depth=2"), "--css=report.css",
                        "--include-before-body=_navigation.html", "--citeproc", "--bibliography=../references.bib",
                        "--metadata=lang:en", "--output=" + name + ".html"], cwd=report, input=source, text=True, encoding="utf-8", check=True)
        print(name + ".html")
    for source, target in GUIDES.items():
        text = (ROOT / "docs" / source).read_text(encoding="utf-8")
        title = text.splitlines()[0].removeprefix("# ")
        text = text.split("\n", 1)[1]
        for name, rendered in GUIDES.items():
            text = text.replace(f"]({name})", f"]({rendered})")
        body = subprocess.run(
            [pandoc, "--from=gfm", "--to=html5"],
            input=text, text=True, encoding="utf-8", capture_output=True, check=True,
        ).stdout
        document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} | Carbon Team</title>
<link rel="stylesheet" href="style.css">
<script src="assets/ui/theme-init.js"></script>
<link rel="stylesheet" href="assets/ui/app.css">
<script src="assets/ui/app-shell.js" defer></script>
<style>main{{max-width:1100px}}main table{{width:100%;border-collapse:collapse;margin:1.5rem 0}}th,td{{text-align:left;padding:.65rem;border-bottom:1px solid var(--border,#dce3e8)}}main p,main li{{max-width:85ch;line-height:1.65}}pre{{overflow-x:auto;padding:1rem;border-radius:8px;background:var(--surface-2,#eef3f5)}}@media(max-width:650px){{table{{display:block;overflow-x:auto}}}}@media print{{.ct-sidebar,.ct-topbar,.ct-skip{{display:none!important}}.ct-workspace{{margin:0!important}}}}</style>
</head><body class="ct-app">
<header><div class="eyebrow">Carbon Team &middot; 17 November 2026</div><h1>{html.escape(title)}</h1></header>
<main>{body}<p><a href="../model/report/project/index.html">Model and results</a> &middot; <a href="../model/report/project/model_and_results.pdf">Four-page PDF</a> &middot; <a href="{html.escape(source)}">Markdown source</a></p></main>
</body></html>
'''
        (ROOT / "docs" / target).write_text(document, encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
