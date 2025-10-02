from dataclasses import dataclass
from typing import Any, Dict, Optional
import os, json, subprocess, shlex, pathlib

class ToolError(Exception): pass

ALLOW_DIR = os.getenv("WORKDIR", os.getcwd())
def _safe_path(p): return str(pathlib.Path(ALLOW_DIR, p).resolve())

def fs_read(path:str)->str:
    full=_safe_path(path)
    if not full.startswith(ALLOW_DIR): raise ToolError("Path escape blocked")
    with open(full,"r",encoding="utf-8",errors="ignore") as f: return f.read()

def fs_write(path:str, content:str)->str:
    full=_safe_path(path)
    if not full.startswith(ALLOW_DIR): raise ToolError("Path escape blocked")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full,"w",encoding="utf-8") as f: f.write(content)
    return "ok"

def fs_diff(path:str, new:str)->str:
    full=_safe_path(path)
    before = ""
    if os.path.exists(full):
        with open(full,"r",encoding="utf-8",errors="ignore") as f: before=f.read()
    return json.dumps({"path":path, "before":before, "after":new})

def sh(cmd:str, consent:bool=False)->str:
    if not consent: raise ToolError("Consent required for shell")
    cp = subprocess.run(shlex.split(cmd), capture_output=True, text=True, cwd=ALLOW_DIR)
    if cp.returncode: raise ToolError(cp.stderr)
    return cp.stdout
