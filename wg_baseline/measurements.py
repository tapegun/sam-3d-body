"""Fixed-height geometric section proxies, not anatomical tape measurements."""
import numpy as np

FRACTIONS = {'hip_section_mm': .46, 'waist_section_mm': .56, 'chest_section_mm': .70}

def measure(mesh):
    height = float(mesh.extents[1])
    if not np.isfinite(mesh.vertices).all() or height <= 0:
        raise ValueError('Nonfinite or degenerate mesh')
    result = {'height_mm': height * 1000}
    for name, fraction in FRACTIONS.items():
        y = mesh.bounds[0, 1] + fraction * height
        section = mesh.section(plane_normal=[0, 1, 0], plane_origin=[0, y, 0])
        loops = [] if section is None else [loop for loop in section.discrete
            if len(loop) > 3 and np.linalg.norm(loop[0] - loop[-1]) < 1e-5]
        result[name] = (float(max(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()
                                for loop in loops) * 1000) if loops else None)
    return result

def errors(pred, truth):
    return {name: None if pred.get(name) is None else pred[name] - value
            for name, value in truth.items() if value is not None}

def calibrated(mesh, height_m):
    result = mesh.copy()
    result.vertices *= height_m / mesh.extents[1]
    return result

def summarize(rows, key):
    result = {}
    for metric in ['height_mm', *FRACTIONS]:
        values = [row.get(key, {}).get(metric) for row in rows]
        values = np.array([v for v in values if v is not None], dtype=float)
        result[metric] = {'valid': len(values), 'total': len(rows),
            'mae_mm': float(np.abs(values).mean()) if len(values) else None,
            'bias_mm': float(values.mean()) if len(values) else None,
            'p95_abs_mm': float(np.percentile(np.abs(values),95)) if len(values) else None}
    return result
