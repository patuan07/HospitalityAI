#!/usr/bin/env bash
set -e
cd webapp_streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
