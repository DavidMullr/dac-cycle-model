"""Run the retained JavaScript checks with Node."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / 'tests').glob('*.js')) + sorted((ROOT / 'model/tests').glob('*.js'))
for path in files:
    subprocess.run(['node', str(path)], cwd=ROOT, check=True)
print(f'PASS: {len(files)} JavaScript test files')
