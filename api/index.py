"""Vercel Python entrypoint shim.

Vercel's Python builder only auto-detects an entrypoint at a fixed set of
conventional paths (see the build error this file fixes), and webapp/app.py
isn't one of them. This file just re-exports the real Flask app from there
so Vercel has a top-level `app` to load.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.app import app  # noqa: E402,F401
