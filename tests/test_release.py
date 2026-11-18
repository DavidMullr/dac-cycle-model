import hashlib
import json
import zipfile

import pytest

from tools.release import MANIFEST, build, inventory, verify


def test_release_excludes_private_runtime_and_verifies_exact_bytes(tmp_path):
    root = tmp_path / 'project'
    for name in ('README.md', 'model/run.py', 'model/outputs/engineering/quote.json',
                 'model/.venv/secret.txt', 'model/config/mof_screening/runtime/screening_example.json', 'docs/.env', 'docs/test.pem', 'docs/__pycache__/cache.pyc'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('private' if name != 'README.md' else 'Review me')
    rows = inventory(root)
    assert [row['path'] for row in rows] == ['README.md', 'model/run.py']
    (root / MANIFEST).write_text(json.dumps({'schema':'carbon-team-current-release/1', 'files':rows}))
    output = tmp_path / 'review.zip'
    assert build(output, root)['files'] == 3
    with zipfile.ZipFile(output) as archive:
        assert archive.read('README.md') == b'Review me'
    with pytest.raises(FileExistsError):
        build(output, root)
    (root / 'README.md').write_text('Changed')
    with pytest.raises(ValueError, match='differs'):
        verify(root)


def test_release_rejects_private_key_in_selected_source(tmp_path):
    (tmp_path / 'README.md').write_text('-----BEGIN ' + 'PRIVATE KEY-----')
    with pytest.raises(ValueError, match='Private data'):
        inventory(tmp_path)
