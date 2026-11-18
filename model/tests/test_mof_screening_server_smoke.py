import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.server import build_server


def test_delivery_server_serves_screening_scope_and_safe_errors(tmp_path):
    server = build_server(
        0,
        output_root=tmp_path / "materials",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/screening/") as response:
            html = response.read().decode("utf-8")
        assert "MOF screening workspace" in html
        assert 'id="resultBasisCaption"' in html
        assert 'id="comparisonDiagnostics"' in html
        assert 'id="transportApplicability"' in html
        assert 'id="guidedMaterialPanel"' in html
        assert 'id="gmFamily"' in html
        assert 'id="gmCo2CurveCsv"' in html
        assert "Display/load built-in material" in html
        with urlopen(f"http://127.0.0.1:{port}/screening/app.js") as response:
            javascript = response.read().decode("utf-8")
        assert "conditioned collected gas, wet basis" in javascript
        assert "source identities differ" in javascript
        assert "Lumped companion: well-mixed transport" in javascript
        assert "unknown is not treated as met" in javascript
        assert "unknown is not treated as pass" in javascript
        assert "not counted as a completed-target miss" in javascript
        with urlopen(
            f"http://127.0.0.1:{port}/screening/material_builder.js"
        ) as response:
            builder_javascript = response.read().decode("utf-8")
        assert "guided material package is incomplete" in builder_javascript
        with urlopen(
            f"http://127.0.0.1:{port}/screening/material_state.js"
        ) as response:
            state_javascript = response.read().decode("utf-8")
        assert "createMaterialGate" in state_javascript
        with urlopen(
            f"http://127.0.0.1:{port}/screening/result_view.js"
        ) as response:
            result_view_javascript = response.read().decode("utf-8")
        assert "residualSummary" in result_view_javascript
        with urlopen(
            f"http://127.0.0.1:{port}/api/mof-screening/scope"
        ) as response:
            scope = json.loads(response.read())
        assert scope["schema"] == "mof-screening-public-scope/1"
        assert isinstance(scope["gate"]["start_enabled"], bool)
        request = Request(
            f"http://127.0.0.1:{port}/api/mof-screening/preview",
            data=json.dumps({"material": [], "case": {}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as error:
            payload = json.loads(error.read())
            assert error.code == 400
            assert "material" in payload["error"]
        else:
            raise AssertionError("array material was unexpectedly accepted")
    finally:
        server.shutdown()
        server.server_close()
        server.manager.close()
