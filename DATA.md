# Data, private configuration and recovery

The music folder `podcast001/`, archives and derived CSVs stay local. `install_set.py` contains the collection URL and downloads accessible tracks with scdl; run it only for audio you are entitled to download. Alternatively restore your licensed collection into `podcast001/`. Run `detect_bpm.py`, `detect_key.py` and `compute_features.py` to recreate `features.csv`; run `shortest_path.py` for the ordering CSV and the remaining scripts for plots. Consult each script for its expected working directory. No GitHub copy of the audio is provided.

GitHub contains source, not a backup of excluded files. Keep irreplaceable private data in a separate protected backup. Exclusions do not delete local files. Rebuild dependencies and disposable artifacts using the project setup instructions.
