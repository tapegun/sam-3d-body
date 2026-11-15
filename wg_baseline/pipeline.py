"""Run validation smoke inference then test baseline after the SDF job exits."""
import json
from pathlib import Path
import subprocess
import sys

status=Path('results/pipeline_status.json')
def write(state): status.write_text(json.dumps({'status':state}))
try:
    write('waiting_for_sdf_then_validation_smoke')
    subprocess.run([sys.executable,'-m','wg_baseline.run','--wait-for-sdf','--split','val','--bodies','2','--out','results/validation_smoke'],check=True)
    report=json.loads(Path('results/validation_smoke/report.json').read_text())
    if any('sam_failure' in row for row in report['rows']): raise RuntimeError('Validation smoke failed')
    write('test_baseline')
    subprocess.run([sys.executable,'-m','wg_baseline.run'],check=True)
    write('complete')
except Exception:
    write('failed')
    raise
