# WoundPatch
Woundpatch detects a likely wound region on skin, estimates the arm direction, then overlays a band-aid (with scaling + rotation). Kindly refer to [approach notes](./APPROACH.md) on my writeup on the different methods I tried and considered.

## Setup
Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python woundpatch.py path/to/arm1.jpg path/to/arm2.jpg --show
```

Outputs are saved to `outputs/` as a bandaged image plus a side-by-side comparison.

## Notes
- `approach_note.md` contains the solution explanation.
- Wound detection defaults to a lightweight GMM clustering mode. You can switch to the older heuristic mode with `--wound-mode heuristic`.
- A segmentation-based option is available with `--wound-mode fastsam` (expects `fastsam-s.pt` in the repo root or downloads via Ultralytics).
- A wound-segmentation model option is available with `--wound-mode woundseg` and uses weights from `wound-segmentation/training_history/2019-12-19 01%3A53%3A15.480800.hdf5`.
- `woundseg` requires TensorFlow/Keras (installed via `requirements.txt`).
