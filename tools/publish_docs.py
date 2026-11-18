"""Prepare the already rendered reports and guides for GitHub Pages."""
from __future__ import annotations

import argparse
from html import escape
import os
from pathlib import Path
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from check_reports import Document, PAGES, REPORT, ROOT
from render_guides import GUIDES


def publish(output: Path, repository: str, revision: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    docs = ROOT / "docs"
    guides = {docs / name for name in GUIDES.values()}
    pages = guides | {REPORT / f"{name}.html" for name in PAGES}
    assets = {REPORT / "model_and_results.pdf", docs / "assets/model_call_flow.svg", docs / "style.css"}
    hosted = pages | assets
    for asset in assets:
        shutil.copyfile(asset, output / asset.name)

    for page in pages:
        text = page.read_text(encoding="utf-8")
        if page in guides:
            # The local app injects simulator navigation. The public guides use their own links.
            text = "\n".join(line for line in text.splitlines() if "assets/ui/" not in line)
        for value in set(Document(page).links):
            url = urlsplit(value)
            if url.scheme or url.netloc or not url.path or url.path.startswith("assets/ui/"):
                continue
            target = (page.parent / unquote(url.path)).resolve()
            if not target.is_file() or not target.is_relative_to(ROOT):
                raise ValueError(f"{page.name}: missing project file {value}")
            if target in hosted:
                destination = target.name
            else:
                name = quote(target.relative_to(ROOT).as_posix())
                if target.suffix in {".csv", ".json"}:
                    destination = f"https://raw.githubusercontent.com/{repository}/{revision}/{name}"
                else:
                    destination = f"https://github.com/{repository}/blob/{revision}/{name}"
            link = urlunsplit(("", "", destination, url.query, url.fragment))
            for attr in ("href", "src"):
                text = text.replace(f'{attr}="{escape(value)}"', f'{attr}="{escape(link)}"')
        (output / page.name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the site")
    args = parser.parse_args()
    publish(args.output, os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"])
