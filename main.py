"""
Web Scraper AI - Main Entry Point Shim

Forwards execution to `app.main:app` to support:
- `uv run uvicorn main:app --reload`
- `uv run uvicorn app.main:app --reload`
- `uv run python main.py`
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
