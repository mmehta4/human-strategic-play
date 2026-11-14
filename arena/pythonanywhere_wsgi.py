"""WSGI entry point for PythonAnywhere's free plan (see DEPLOY.md, "Free: PythonAnywhere").
Copy this file's contents into the WSGI configuration file PythonAnywhere creates, and
replace YOUR_USERNAME. Secrets are read from ~/.arena_env, never written here."""
import os
import sys
from pathlib import Path

HOME = Path("/home/YOUR_USERNAME")
REPO = HOME / "repo"                       # where you cloned the repository (the stage1 folder's contents)
sys.path.insert(0, str(REPO))

for line in (HOME / ".arena_env").read_text().splitlines():   # lines like ARENA_SECRET=...
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("ARENA_DB", str(HOME / "arena.db"))
os.environ.setdefault("ARENA_HTTPS", "1")

from arena.app import app as application  # noqa: E402
