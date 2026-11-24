"""Check links outside the report chapters, including the files behind generated downloads."""
import gzip
import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen

from app.server import build_server
from tools.check_reports import Document


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_page_destinations(tmp_path):
    server = build_server(0, output_root=tmp_path / "jobs")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    pages = {
        "/": ROOT / "docs/workspace/index.html",
        "/results/": ROOT / "docs/workspace/results.html",
        "/engineering/": ROOT / "docs/workspace/engineering.html",
        "/screening/": ROOT / "model/app/mof_screening/static/index.html",
        "/docs/geometry.html": ROOT / "docs/geometry.html",
        "/docs/workspace/bed-aging.html": ROOT / "docs/workspace/bed-aging.html",
    }
    fetched = {}
    try:
        for route, source in pages.items():
            for link in Document(source).links:
                url = urlsplit(urljoin(base + route, link))
                if url.netloc != urlsplit(base).netloc:
                    continue
                target = url._replace(fragment="").geturl()
                if target not in fetched:
                    with urlopen(target, timeout=10) as response:
                        fetched[target] = response.read() if response.headers.get_content_type() == "text/html" else None
                if url.fragment:
                    assert fetched[target] is not None, (route, link)
                    document = tmp_path / "target.html"
                    document.write_bytes(fetched[target])
                    assert unquote(url.fragment) in Document(document).ids, (route, link)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        server.manager.close()


def test_bed_aging_downloads_match_stored_run_digests():
    data = ROOT / "docs/data"
    reference = json.loads((data / "bed-aging-lewatit.json").read_bytes())
    for run in reference["runs"]:
        with gzip.open(data / run["source_export"], "rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == run["source_json_sha256"]
