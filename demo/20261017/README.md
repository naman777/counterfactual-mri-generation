# Demo assets, 17 October 2026

- `screenshots/`: actual browser captures of the updated local dashboard.
- `frames/`: recorded viewport views for the offline walkthrough.
- `make_backup.py`: assembles captured views using ffmpeg into the captioned,
  silent `mrone-backup-walkthrough.mp4` (three minutes). `captions.srt` and
  `frames.concat` are its inputs. It does not show live GPU inference.
- `mrone-demo.zip`: created by `python -m api.package_demo --output
  demo/20261017/mrone-demo.zip`. Install dependencies before offline use.

The 20-slice result bundle is `api/demo_bundle/`; its manifest records selection
and all 80 exact requests. Research inputs remain intact. Same-machine HTTP
verification: `reports/v4_dashboard_demo_20261017_01/`. No second laptop was
available, so no second-machine result is claimed.
