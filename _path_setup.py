"""
_path_setup.py — Ensures the project root is on sys.path.

Imported at the top of every module that does cross-package imports.
This is needed when running `python main.py` from inside the project
directory on systems where the package root is not automatically added
to sys.path (e.g. macOS with conda environments).
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
