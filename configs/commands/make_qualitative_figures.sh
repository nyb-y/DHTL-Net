#!/usr/bin/env bash
set -euo pipefail

python tools/visualization/make_picture1.py
python tools/visualization/make_picture2.py
python tools/visualization/make_picture3.py
python tools/visualization/make_picture3_paper_layout.py
python tools/visualization/make_picture4.py
