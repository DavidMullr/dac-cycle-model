"""Verify the shared numerical functions used by the current model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSURE_PATH = ROOT / 'config/mof_screening/dependencies.json'


class SourceClosureError(ValueError):
    """A required numerical dependency is missing or changed."""


def validate_source_closure(path=CLOSURE_PATH):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('schema') != 'dac-current-dependencies/1' or not document.get('files'):
        raise SourceClosureError('Current dependency record is missing or malformed')
    files = {}
    for row in document['files']:
        relative = row['path']
        target = (ROOT / relative).resolve()
        if relative in files or not target.is_relative_to(ROOT) or not target.is_file():
            raise SourceClosureError('Invalid current dependency: ' + relative)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise SourceClosureError('Current dependency changed: ' + relative)
        files[relative] = actual
    return {'path':Path(path).relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'files':files, 'file_count':len(files)}
