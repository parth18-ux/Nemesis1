"""Vercel / uvicorn entrypoint: `uvicorn index:app`."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "service"))
from server import app  # noqa: E402,F401
