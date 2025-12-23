# WoundPatch
Woundpatch detects a likely wound region on skin, estimates the arm direction, then overlays a band-aid (with scaling + rotation).

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
- `approach_note.md` contains the 150–300 word solution explanation.
- You can provide your own band-aid PNG (with alpha) via `--bandage path/to/bandage.png`.
