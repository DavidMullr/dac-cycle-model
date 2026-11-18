"""Check, manifest and package the current source tree without local run state."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'RELEASE_MANIFEST.json'
ROOT_FILES = {'.gitattributes', '.gitignore', 'README.md', 'CONTRIBUTING.md',
              'LICENSE', 'THIRD_PARTY_NOTICES.md', 'INSTALL.cmd', 'START.cmd',
              'STOP.cmd', 'pytest.ini'}
TREES = ('docs', 'tests', 'tools', '.github', 'model/app', 'model/config',
         'model/equipment', 'model/materials', 'model/report', 'model/screening', 'model/tests')
MODEL_FILES = ('run.py', 'launcher.py', 'launch_identity.py', 'requirements-lock.txt')
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.quarto', '.git', 'outputs', 'qa', 'node_modules'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.log', '.tmp', '.bak', '.key', '.pem', '.pfx', '.zip'}


def public_files(root=ROOT):
    candidates = [root / name for name in ROOT_FILES] + [root / 'model' / name for name in MODEL_FILES]
    for name in TREES:
        candidates.extend((root / name).rglob('*'))
    selected = set()
    for path in candidates:
        relative = path.relative_to(root)
        if relative.parts[:4] == ('model', 'config', 'mof_screening', 'runtime'):
            continue
        if not path.is_file() or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if any(p.startswith('.venv') or p.endswith('_files') for p in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or path.name.startswith(('.env', 'credentials', 'secrets')):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('External file link in release: ' + relative.as_posix())
        selected.add(relative.as_posix())
    return sorted(selected)


def inventory(root=ROOT):
    rows = []
    for name in public_files(root):
        raw = (root / name).read_bytes()
        # Do not package credentials or personal workspace prefixes.
        patterns = (rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                    rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{50,}',
                    rb'[A-Za-z]:[\\/]+Users[\\/]+david[\\/]',
                    rb'[A-Za-z]:[\\/]+MasterEPFL[\\/]+Carbon Team')
        if any(re.search(pattern, raw, re.I) for pattern in patterns):
            raise ValueError('Private data pattern in release file: ' + name)
        rows.append({'path':name, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
    return rows


def checks():
    sys.path.insert(0, str(ROOT / 'model'))
    from screening.mof_screening.preflight import readiness
    from tools.check_submission import check as results_check
    from tools.check_reports import check as reports_check
    values = {'execution_readiness':readiness(), 'results':results_check(), 'reports':reports_check()}
    if not values['execution_readiness']['start_enabled'] or values['results']['errors'] or values['reports']['errors']:
        raise ValueError(json.dumps(values, indent=2))
    return values


def verify(root=ROOT):
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    actual = inventory(root)
    if manifest.get('schema') != 'carbon-team-current-release/1' or manifest.get('files') != actual:
        raise ValueError('Release manifest differs from the selected files; review the edits before regenerating it.')
    return actual


def build(output, root=ROOT):
    rows = verify(root)
    output = Path(output).resolve()
    if output.is_relative_to(root.resolve()):
        raise ValueError('Write the release ZIP outside the project directory.')
    names = [row['path'] for row in rows] + [MANIFEST]
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, name)
    with zipfile.ZipFile(output) as archive:
        if sorted(archive.namelist()) != sorted(names) or archive.testzip() is not None:
            raise ValueError('ZIP membership or CRC verification failed')
        for row in rows:
            if hashlib.sha256(archive.read(row['path'])).hexdigest() != row['sha256']:
                raise ValueError('ZIP member changed: ' + row['path'])
        if archive.read(MANIFEST) != (root / MANIFEST).read_bytes():
            raise ValueError('ZIP manifest changed')
    return {'status':'PASS', 'files':len(names), 'bytes':output.stat().st_size,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('manifest', 'verify', 'build'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks()
    if args.command == 'manifest':
        rows = inventory()
        (ROOT / MANIFEST).write_text(json.dumps({'schema':'carbon-team-current-release/1', 'files':rows}, indent=2) + '\n', encoding='utf-8')
        result = {'status':'PASS', 'files':len(rows), 'manifest':MANIFEST}
    elif args.command == 'verify':
        result = {'status':'PASS', 'files':len(verify())}
    elif args.output is None:
        parser.error('build requires --output')
    else:
        result = build(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
