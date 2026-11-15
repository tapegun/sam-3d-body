# WG full-pretrained SAM measurement baseline

This fork preserves the official SAM 3D Body model and adds a separate,
inference-only measurement experiment. No fine-tuning is performed.

## Isolation and execution

Linux checkout: `/home/tapegun/Projects/sam-3d-body-wg-baseline`.
Branch: `codex/measurement-baseline`. Upstream: facebookresearch/sam-3d-body.
Separate virtual environment, MHR asset, logs and outputs. Existing Anny data,
SAM checkpoint and SDF results are read-only inputs. Nothing is written to the
cleanup or training checkouts, and no packages are installed into their envs.

The environment uses system site packages (including existing CUDA PyTorch),
with additional dependencies installed only into `.venv`. Local additions
are recorded in `requirements-local.txt`. Upstream INSTALL.md also applies.

Run from the fork root:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m wg_baseline.pipeline
```

The pipeline waits for the existing SDF experiment's `stage.txt` to become
`complete`. It exits if that experiment reports failure. It then runs two
validation smoke bodies followed by 128 test bodies. Output directories must
not already exist: accidental overwrites are rejected. It is not resumable;
use `wg_baseline.run --out <new-directory>` for a deliberate separate rerun.
Inspect `results/pipeline_status.json` and `results/pipeline.log`.

## Fixed evaluation protocol

- Same 128 held-out Anny test bodies, same cached 256px front-view RGB images
  used for SDF mesh extraction. Verify dataset and image hashes before inference.
- Official pretrained DINOv3-H+ checkpoint, complete body and hand decoders,
  packaged MHR TorchScript asset. No detector, mask prompts, or FOV estimator:
  one centered synthetic person, official full-image crop and default FOV.
- No ground-truth mesh, height, pose or crop is passed into SAM inference.
- Preserve raw outputs and camera-space predictions. Convert camera axes to
  x-right/y-up/z-back for measurement; coordinates are meters in upstream code.
- Measure vertical height and closed horizontal-section perimeters at 46%,
  56% and 70% of each mesh's height (hip/waist/chest section proxies). Use the
  longest closed loop, matching the existing Anny diversity summary convention.
- Report signed error, MAE and absolute-error p95 in millimeters, with coverage
  and failures. Compare identical measurements from the SDF meshes and GT.
- Native-scale results are primary. Also report both methods after explicit
  known-height calibration using GT height; those are conditional/oracle-height
  diagnostics, not an independently predicted scale result. Calibrated height
  error is zero by construction and must not be claimed as model accuracy.
- Paired native comparison includes only bodies with both predictions; metric
  coverage is reported. Failed predictions are retained in the full report.

These are geometric section proxies, NOT anatomical waist/chest/hip tape
measurements. Pose errors, arm intersections and fixed section placement can
change them. Inputs are synthetic orthographic renders; SAM uses a perspective
camera model here. This is an in-domain comparison favoring the SDF model's
training distribution, not a claim about general-purpose human reconstruction.
No real-person measurement accuracy or millimeter guarantee follows.

## Minimal upstream patch

Pin DINOv3 to revision 6876159a11b4df116f30f667f8c9888617df0751, also used by
the SDF experiment. Initialize its unused mask token to zero because released
SAM weights omit it. Unmasked DINO uses `0 * mask_token`. All substantive
pretrained inference paths remain upstream. The upstream loader also reports
missing MHR fields because they are supplied by the separately loaded official
TorchScript asset, plus the derived `hand_pose_comps_ori` buffers.

The report records checkpoint/MHR/dataset/image hashes, source commit, inputs,
per-body measurements and failures. SAM's LICENSE continues to apply.
