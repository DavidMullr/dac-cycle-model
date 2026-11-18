"""Loopback application for the current DAC model, reviewed results and engineering."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from app import engineering
from app.mof_screening import routes
from screening.mof_screening import MofScreeningManager
from screening.mof_screening.contract import MofScreeningError

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PROJECT_ID = 'carbon-team-dac'
MAX_BODY = 800_000


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    @property
    def mof_screening_manager(self):
        return self.server.manager

    @property
    def execution_gate(self):
        return self.server.execution_gate

    def log_message(self, *args):
        return

    def _allowed_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def _allowed_origin(self):
        origin = self.headers.get('Origin')
        return origin is None or origin in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}

    def send_bytes(self, status, payload, content_type, filename=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(payload)

    def json(self, status, payload):
        self.send_bytes(status, json.dumps(payload, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8')

    def _send_text(self, status, text, content_type, filename=None):
        self.send_bytes(status, text.encode('utf-8'), content_type, filename)

    def file(self, path):
        path = Path(path)
        if not path.is_file():
            self.json(404, {'error':'File not found'})
            return
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_bytes(200, path.read_bytes(), mime)

    def contained_file(self, root, relative):
        base = root.resolve()
        path = (base / unquote(relative)).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            self.json(404, {'error':'File not found'})
        else:
            self.file(path)

    def do_GET(self):
        if not self._allowed_host():
            self.json(403, {'error':'Host rejected'})
            return
        parsed = urlsplit(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        try:
            if path == '/health':
                self.json(200, {'project_id':PROJECT_ID, 'project_root':str(ROOT), 'port':self.server.server_port,
                               'pid':os.getpid(), 'started_at':self.server.started_at,
                               'launch_nonce':self.server.launch_nonce, 'no_simulation_on_start':True})
            elif path == '/api/results':
                self.json(200, engineering.catalogue(self.server.manager))
            elif path == '/api/result':
                result, _ = engineering.load_result((query.get('result') or [''])[0], self.server.manager)
                self.json(200, result)
            elif path == '/api/engineering':
                self.json(200, engineering.calculate((query.get('result') or [''])[0], self.server.manager))
            elif path == '/':
                self.file(PROJECT / 'docs/workspace/index.html')
            elif path in {'/engineering', '/engineering/'}:
                self.file(PROJECT / 'docs/workspace/engineering.html')
            elif path in {'/results', '/results/'}:
                self.file(PROJECT / 'docs/workspace/results.html')
            elif path in {'/project-report', '/project-report/'}:
                self.file(ROOT / 'report/project/index.html')
            elif path.startswith('/project-report/project/'):
                self.contained_file(ROOT / 'report/project', path[len('/project-report/project/'):])
            elif path.startswith('/project-report/'):
                self.contained_file(ROOT / 'report/project', path[len('/project-report/'):])
            elif path.startswith('/docs/'):
                self.contained_file(PROJECT / 'docs', path[len('/docs/'):])
            elif path.startswith('/model/report/project/'):
                self.contained_file(ROOT / 'report/project', path[len('/model/report/project/'):])
            elif not routes.get(self, path, query):
                self.json(404, {'error':'Unknown endpoint'})
        except (MofScreeningError, ValueError, KeyError, OSError) as error:
            self.json(getattr(error, 'status', 400), {'error':str(error)})

    def do_POST(self):
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not lengths[0].isdigit() or not 0 < int(lengths[0]) <= MAX_BODY:
            self.json(413, {'error':'A bounded JSON body is required'})
            return
        raw = self.rfile.read(int(lengths[0]))
        if len(raw) != int(lengths[0]):
            self.json(400, {'error':'Incomplete body'})
            return
        if not self._allowed_host() or not self._allowed_origin():
            self.json(403, {'error':'Host/origin rejected'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self.json(415, {'error':'Content-Type must be application/json'})
            return
        original = self.rfile
        self.rfile = io.BytesIO(raw)
        try:
            parsed = urlsplit(self.path)
            path, query = parsed.path, parse_qs(parsed.query)
            if path.startswith('/api/engineering/'):
                body = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON is refused')))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object')
                selector = body.get('result')
                if path == '/api/engineering/save':
                    report = engineering.save(selector, self.server.manager, body.get('costs'), body.get('expected_revision'))
                    self.json(200, report)
                elif path in {'/api/engineering/calculate', '/api/engineering/export'}:
                    report = engineering.calculate(selector, self.server.manager, body.get('costs'))
                    if path.endswith('/calculate'):
                        self.json(200, report)
                    elif (query.get('format') or ['json'])[0] == 'csv':
                        self._send_text(200, engineering.csv_text(report), 'text/csv; charset=utf-8', 'engineering.csv')
                    elif (query.get('format') or ['json'])[0] == 'json':
                        self._send_text(200, json.dumps(report, indent=2, allow_nan=False), 'application/json', 'engineering.json')
                    else:
                        self.json(400, {'error':'Export format must be json or csv'})
                else:
                    self.json(404, {'error':'Unknown engineering endpoint'})
            elif not routes.post(self, path):
                self.json(404, {'error':'Unknown endpoint'})
        except (MofScreeningError, ValueError, KeyError, OSError) as error:
            self.json(getattr(error, 'status', 400), {'error':str(error)})
        finally:
            self.rfile = original


def build_server(port=8772, *, output_root=None, launch_nonce=None, project_root=None):
    if project_root is not None and Path(project_root).resolve() != ROOT:
        raise ValueError('Project root differs from this checkout')
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.manager = MofScreeningManager(output_root=output_root)
    server.execution_gate = threading.Lock()
    server.started_at = datetime.now(timezone.utc).isoformat()
    server.launch_nonce = launch_nonce or uuid.uuid4().hex
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8772)
    parser.add_argument('--launch-nonce')
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    server = build_server(args.port, output_root=args.output_root, launch_nonce=args.launch_nonce, project_root=args.project_root)
    try:
        print(f'http://127.0.0.1:{server.server_port}/', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.manager.close()
        server.server_close()


if __name__ == '__main__':
    main()
