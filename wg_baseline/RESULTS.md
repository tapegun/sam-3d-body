# Completed synthetic measurement baseline

All 128 test bodies completed, with 128 matched SDF meshes and no SAM inference failures. Two validation bodies passed full-inference smoke testing first. No parameter fitting or fine-tuning was performed.

Mean absolute error (mm). Calibrated columns use the true height as extra information for BOTH methods.

| Measurement | SAM native | SAM known-height | SDF native | SDF known-height |
| --- | ---: | ---: | ---: | ---: |
| Height | 155.03 | fixed by calibration | 1.99 | fixed by calibration |
| 56% height section | 90.56 | 35.06 | 9.58 | 8.95 |
| 70% height section | 70.84 | 28.04 | 9.03 | 8.96 |
| 46% height section (invalid hip proxy) | 311.35 | 321.51 | 13.89 | 13.76 |

The SDF model performs better on these held-out synthetic bodies, which share its training distribution and calibrated rendering setup. SAM uses its general-purpose pretrained model with default perspective FOV on orthographic input. This does not establish superiority on real photos.

Known-height calibration lowers SAM section error substantially, but residual shape/pose/section-placement errors remain. Native height MAE is 155 mm. No claim of anatomical tape-measure accuracy is supported.

## Hip proxy audit

The 46% height slice crosses two separate contours in 117/128 GT meshes. In 56 of those cases SAM has one contour at that level. There are 61 two/two, 56 two/one, 9 one/one and 2 three/one GT/SAM contour-count pairs. The longest-loop perimeter is therefore often a thigh section, not a hip circumference. Preserve these numbers as protocol diagnostics; do not use them as hip accuracy or tune slice fractions on this test set. An anatomical landmark protocol should be designed on validation data before a new held-out assessment.

## Artifacts

- `baseline_summary.json`: aggregate metrics, p95/bias/coverage and hashes.
- Server `results/full_pretrained_test/report.json`: per-body results.
- Server `results/full_pretrained_test/hip_section_audit.json`: contour-count audit.
- Server `results/full_pretrained_test/test_*.ply` and `*_raw.npz`: full predictions.
- Server `results/pipeline_status.json`: complete.
- Source inference revision: `20103f03583779e33c921a056af660b329597219`.

The existing SDF experiment completed independently before SAM inference started. Its repository and outputs were not modified.
