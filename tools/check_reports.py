"""Check rendered reports, operating guides, anchors and local assets."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "model/report/project"
PAGES = ("index", "model_reference", "evidence", "user_guide")


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.duplicate_ids = []
        self.links = []
        self.images = []
        self.math_blocks = 0
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            if values["id"] in self.ids:
                self.duplicate_ids.append(values["id"])
            self.ids.add(values["id"])
        if tag in {"a", "link"} and values.get("href"):
            self.links.append(values["href"])
        if tag in {"img", "script"} and values.get("src"):
            self.links.append(values["src"])
        if tag == "img":
            self.images.append(values)
        if tag == "math" and values.get("display") == "block":
            self.math_blocks += 1


def check():
    documents = {REPORT / f"{name}.html": Document(REPORT / f"{name}.html") for name in PAGES}
    guides = [ROOT / 'docs' / name for name in ('project_map.html', 'quickstart.html', 'DATA_NEEDED.html', 'REVIEW_CHECKS.html')]
    documents.update({path: Document(path) for path in guides})
    errors = []
    count = 0
    for path, doc in list(documents.items()):
        if doc.duplicate_ids:
            errors.append(f"{path.name}: duplicate ids {doc.duplicate_ids}")
        for img in doc.images:
            if not img.get("alt", "").strip():
                errors.append(f"{path.name}: missing image description")
        for value in doc.links:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            if url.path.startswith("/"):
                errors.append(f"{path.name}: nonportable local link {value}")
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"{path.name}: missing or outside-project link {value}")
                continue
            count += 1
            if url.fragment and target.suffix == ".html":
                if target not in documents:
                    documents[target] = Document(target)
                if unquote(url.fragment) not in documents[target].ids:
                    errors.append(f"{path.name}: missing anchor {value}")
        text = path.read_text(encoding="utf-8")
        if re.search(r"\[WARNING\]|\?@(?:mcdonald|brandani)|<span class=\"citation\"[^>]*>\[\?", text):
            errors.append(f"{path.name}: unresolved rendering or citation marker")
    methods = documents[REPORT / "model_reference.html"]
    if methods.math_blocks < 12:
        errors.append("technical reference: expected native MathML equations")
    required_sections = {
        REPORT / "model_reference.html": {
            "selecting-a-material-law", "how-humidity-changes-the-reviewed-lewatit-result",
            "balances-actually-integrated-in-a-spatial-cell", "well-mixed-companion",
            "from-bed-outlet-to-pump-and-product", "component-conservation-and-a-worked-export",
        },
        ROOT / "docs/project_map.html": {"follow-one-calculation", "trace-a-reported-number"},
    }
    for page, anchors in required_sections.items():
        missing = anchors - documents[page].ids
        if missing:
            errors.append(f"{page.name}: missing reference sections {sorted(missing)}")
    return {
        "status": "pass" if not errors else "fail",
        "reports": len(PAGES),
        "guides": len(guides),
        "checked_local_references": count,
        "images": sum(len(documents[REPORT / f"{name}.html"].images) for name in PAGES),
        "native_display_equations": methods.math_blocks,
        "errors": errors,
        "no_model_execution": True,
    }


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
