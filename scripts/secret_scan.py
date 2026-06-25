from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".gocache",
    ".gomodcache",
}
SKIP_SUFFIXES = {
    ".bin",
    ".onnx",
    ".data",
    ".tflite",
    ".joblib",
    ".task",
    ".npy",
    ".npz",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
}

PATTERNS = [
    ("real Cloudflare tunnel URL", re.compile(r"https://(?!<)[A-Za-z0-9-]+\.trycloudflare\.com")),
    ("Windows user path", re.compile(r"C:\\Users\\")),
    ("Postgres URL with credentials", re.compile(r"postgres://(?!<user>:<password>)[^\s:@]+:[^\s@]+@")),
    ("long hex token", re.compile(r"\b[a-fA-F0-9]{48,}\b")),
    ("hardcoded bearer header", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}")),
    ("assigned secret", re.compile(r"(?i)\b(api_key|auth_token|secret|password)\s*=\s*['\"](?!\s*['\"]|<|test|example|placeholder)[^'\"]+['\"]")),
]


def should_skip(path: Path) -> bool:
    if path.name == "secret_scan.py":
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return any(part in SKIP_DIRS for part in path.parts)


def iter_text_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and not should_skip(path)]


def main() -> int:
    findings: list[str] = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT)
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{rel}:{line_no}: {label}")

    if findings:
        print("Potential secret/public-safety findings:")
        for finding in findings:
            print(finding)
        return 1

    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
