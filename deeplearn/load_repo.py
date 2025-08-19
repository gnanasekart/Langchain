"""Utilities to load files from a local path or clone a repo URL.

Usage:
    from deeplearn.load_repo import gather_files
    files = gather_files('path_or_git_url')

Returns a list of (relative_path, text) tuples.
"""
import os
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Tuple

TEXT_EXTENSIONS = {'.py', '.md', '.txt', '.js', '.java', '.json', '.yaml', '.yml', '.rs', '.go', '.c', '.cpp', '.cs'}


def _read_ipynb(path: Path) -> str:
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
        parts = []
        for cell in doc.get('cells', []):
            if cell.get('cell_type') == 'code':
                parts.append('\n'.join(cell.get('source', [])))
            else:
                parts.append('\n'.join(cell.get('source', [])))
        return "\n\n".join(parts)
    except Exception:
        return ''


def _read_file(path: Path) -> str:
    if path.suffix == '.ipynb':
        return _read_ipynb(path)
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def gather_files(path_or_url: str, include_extensions: List[str] = None) -> List[Tuple[str, str]]:
    """Return list of (relpath, text) for files under a local path or a git URL.

    If a git URL is provided (starts with http or git@), the repo is cloned to a tmp folder and scanned.
    """
    include_extensions = include_extensions or list(TEXT_EXTENSIONS) + ['.ipynb']

    def scan(root: Path) -> List[Tuple[str, str]]:
        out = []
        for p in root.rglob('*'):
            if p.is_file() and (p.suffix in include_extensions or p.suffix == '.ipynb'):
                rel = str(p.relative_to(root))
                txt = _read_file(p)
                if txt.strip():
                    out.append((rel, txt))
        return out

    # Quick heuristic: treat strings that look like URLs as clone targets
    if path_or_url.startswith('http://') or path_or_url.startswith('https://') or path_or_url.endswith('.git') or path_or_url.startswith('git@'):
        tmp = Path(tempfile.mkdtemp(prefix='deeplearn_repo_'))
        try:
            subprocess.check_call(['git', 'clone', path_or_url, str(tmp)])
            return scan(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        root = Path(path_or_url).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {root}")
        return scan(root)
