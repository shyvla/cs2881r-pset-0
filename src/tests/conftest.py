"""Make the core modules importable when pytest is run from src/ or the repo
root. The test files import `hooks`, `scoring`, `analysis` as top-level names
-- same as every script under src/ -- so src/ itself must be on sys.path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
