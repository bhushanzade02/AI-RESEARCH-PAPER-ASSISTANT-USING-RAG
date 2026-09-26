#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
else
  source .venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
streamlit run app.py
