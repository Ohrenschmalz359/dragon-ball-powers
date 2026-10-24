"""Tests for the Windows setup/launch tooling, the model manifest and the VS Code config.

* ``setup.bat`` / ``run.bat`` are checked statically on every platform (CRLF,
  ASCII, structure, every ``goto``/``call`` target exists, pinned URLs and hashes)
  and the Python snippets embedded in them are executed with this interpreter.
* Where ``cmd.exe`` is reachable (Windows, or WSL with interop) the *safe* paths
  of the batch files run for real in a throw-away copy of the project whose path
  contains spaces, ``&``, parentheses and ``!``: ``run.bat`` without ``.venv`` or
  with a broken one, ``setup.bat --help``, an invalid option and ``--dry-run``
  (Python search order and classification with batch-file stand-ins for ``py`` /
  ``python`` on a private PATH, broken ``.venv``, junction safety, pauses).
  Nothing is downloaded or installed, and the real project folder is never touched.
* ``scripts/download_models.py`` is tested against the real model files and
  against a local HTTP server (no internet needed).
"""

from __future__ import annotations

import ast
import collections
import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_BAT = ROOT / "setup.bat"
RUN_BAT = ROOT / "run.bat"
VCAM_BAT = ROOT / "install_virtual_camera.bat"
WIN11_BAT = ROOT / "install_win11_camera.bat"
WIN11_DRIVERS = ROOT / "drivers" / "win11"
SETUP_SH = ROOT / "setup.sh"
RUN_SH = ROOT / "run.sh"
SCRIPT = ROOT / "scripts" / "download_models.py"
MANIFEST = ROOT / "assets" / "models" / "models.json"
MODELS_DIR = ROOT / "assets" / "models"
VSCODE = ROOT / ".vscode"

PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261013/"
           "cpython-3.12.15%2B20261013-x86_64-pc-windows-msvc-install_only.tar.gz")
PBS_SHA256 = "bbc902d4d187ac4914915bf538196a064b49109a33a104965b5164ced5b80a1e"

# file -> (url, sha256, size): the pinned MediaPipe models (see assets/models/README.md)
EXPECTED_MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1", 7819105),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a", 5777746),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1", 9398198),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b", 30664242),
}

BATCH_FILES = [SETUP_BAT, RUN_BAT, VCAM_BAT, WIN11_BAT]


def _text(path: Path) -> str:
    return path.read_bytes().decode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _present_models() -> list[str]:
    return [name for name in EXPECTED_MODELS if (MODELS_DIR / name).is_file()]


# ===================================================================== batch files


@pytest.mark.parametrize("path", BATCH_FILES, ids=lambda p: p.name)
def test_batch_file_is_crlf_ascii(path: Path):
    data = path.read_bytes()
    data.decode("ascii")  # cmd.exe reads batch files in the OEM code page: keep ASCII
    assert re.search(rb"(?<!\r)\n", data) is None, "LF-only line ending (cmd.exe needs CRLF)"
    assert re.search(rb"\r(?!\n)", data) is None, "stray CR"
    assert data.endswith(b"\r\n")
    assert len(data.split(b"\r\n")) > 20


@pytest.mark.parametrize("path", BATCH_FILES, ids=lambda p: p.name)
def test_batch_file_header(path: Path):
    lines = _text(path).split("\r\n")
    assert lines[0].strip().lower() == "@echo off"
    text = "\n".join(lines).lower()
    assert "setlocal enableextensions" in text
    assert 'cd /d "%~dp0"' in text
    # Delayed expansion would corrupt paths containing "!".
    assert "enabledelayedexpansion" not in text
    assert "dbp_no_pause" in text
    assert "pause" in text


def _labels_and_jumps(text: str) -> tuple[set[str], list[str]]:
    labels = {m.group(1).lower() for m in re.finditer(r"(?m)^\s*:([A-Za-z0-9_]+)\s*$", text)}
    code = "\n".join(line for line in text.splitlines() if not line.strip().lower().startswith("rem"))
    jumps = [m.group(1).lower() for m in re.finditer(r"(?i)\bgoto\s+:?([A-Za-z0-9_]+)", code)]
    jumps += [m.group(1).lower() for m in re.finditer(r"(?i)\bcall\s+:([A-Za-z0-9_]+)", code)]
    return labels, jumps


@pytest.mark.parametrize("path", BATCH_FILES, ids=lambda p: p.name)
def test_batch_jump_targets_exist(path: Path):
    labels, jumps = _labels_and_jumps(_text(path))
    assert jumps, "expected goto/call statements"
    missing = sorted({j for j in jumps if j != "eof" and j not in labels})
    assert not missing, f"goto/call targets without a label: {missing}"


@pytest.mark.parametrize("path", BATCH_FILES, ids=lambda p: p.name)
def test_batch_comments_have_no_variable_references(path: Path):
    # "%~" in a REM line can abort cmd.exe and %VAR% in comments is expanded anyway.
    for line in _text(path).split("\r\n"):
        if line.strip().lower().startswith("rem"):
            assert re.search(r"%[^%\s]+%|%~", line.replace("%%", "")) is None, line


def test_setup_bat_references_every_step():
    text = _text(SETUP_BAT)
    for needle in (r".venv\Scripts\python.exe", "-m dragon_ball_powers --check", "requirements.txt",
                   r"scripts\download_models.py", "-m venv", "-m pip install --upgrade pip",
                   '-m pip install -r "%REQ_TO_INSTALL%"', "py -3.12", "py -3.11", "struct.calcsize('P')",
                   "sys.version_info", r"%SYS32%\curl.exe", r"%SYS32%\certutil.exe", "-hashfile",
                   r"%SYS32%\tar.exe", "--strip-components=1", "rmdir /s /q", "Invoke-WebRequest",
                   "Start the app with run.bat", "VS Code", 'set "PYTHONPATH=%ROOT%src"'):
        assert needle in text, needle
    for option in ("--help", "--dry-run", "--force", "--no-pause", "--dev"):
        assert f'"{option}"' in text, option
    for step in range(1, 7):
        assert f"[{step}/6]" in text
    # Microsoft Store placeholder ("python" opening the Store) and App Control blocks.
    assert "9009" in text and "Microsoft Store" in text
    assert "1260" in text and "group policy" in text


def test_setup_bat_pins_python_build_standalone():
    text = _text(SETUP_BAT)
    url = re.search(r'(?m)^set "PBS_URL=(.*)"\r?$', text).group(1)
    sha = re.search(r'(?m)^set "PBS_SHA256=(.*)"\r?$', text).group(1)
    # A single "%" would start a variable reference in a batch file: "+" -> "%%2B".
    assert "%%2B" in url
    assert url.replace("%%", "%") == PBS_URL
    assert sha == PBS_SHA256


VCAM_COMMIT = "3ed54c325e0ad71afcf4f246c07e5e17b3d7f2d2"
VCAM_SHA = {"64": "72812f5363d8ecb45632253f8c8c888844b1b62e27616f3c8cc21064ccde25e5",
            "32": "aa3ebdf03dea7f3aab3dd7b724751f49ed71672256b57c6a19aa6809cabf30ba"}


def test_virtual_camera_bat_pins_the_driver():
    text = _text(VCAM_BAT)
    base = re.search(r'(?m)^set "VC_BASE=(.*)"\r?$', text).group(1)
    assert base == f"https://raw.githubusercontent.com/schellingb/UnityCapture/{VCAM_COMMIT}/Install"
    for bits, sha in VCAM_SHA.items():
        assert re.search(rf'(?m)^set "VC_SHA{bits}=(.*)"\r?$', text).group(1) == sha
    assert 'set "VC_NAME=Dragon Ball Powers"' in text
    # Only the registry step runs elevated, through one UAC prompt with a fixed argument.
    assert "-Verb RunAs" in text and "--elevated-install" in text and "fltmc.exe" in text
    assert '"/i:UnityCaptureName=%VC_NAME%"' in text


def test_win11_camera_bat_pins_the_committed_drivers():
    text = _text(WIN11_BAT)
    for arch, var in (("x64", "CAM_SHA_X64"), ("arm64", "CAM_SHA_ARM64")):
        pinned = re.search(rf'(?m)^set "{var}=(.*)"\r?$', text).group(1)
        assert pinned == _sha256(WIN11_DRIVERS / arch / "DragonBallPowersCamera.dll"), arch
    assert 'set "CAM_NAME=Dragon Ball Powers (Windows 11)"' in text
    assert "regsvr32.exe\" /s /n /i" in text and "regsvr32.exe\" /s /n /u /i" in text
    assert "CurrentBuildNumber" in text and "LSS 22000" in text
    assert "-Verb RunAs" in text and "fltmc.exe" in text


def _probe_code() -> str:
    return re.search(r'(?m)^set "PROBE_CODE=(.*)"\r?$', _text(SETUP_BAT)).group(1)


@pytest.mark.parametrize("reject_venv", [False, True])
def test_setup_probe_code_runs_and_classifies_this_interpreter(reject_venv: bool):
    code = _probe_code()
    assert "%" not in code and '"' not in code  # must survive cmd.exe quoting untouched
    env = dict(os.environ)
    env.pop("DBP_REJECT_VENV", None)
    if reject_venv:
        env["DBP_REJECT_VENV"] = "1"
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                          env=env, timeout=60)
    first_line = proc.stdout.splitlines()[0]
    assert re.match(r"^Python \d+\.\d+\.\d+ \((32|64)-bit, [\w.-]+\)", first_line), proc.stdout

    compatible = (sys.version_info[:2] in ((3, 11), (3, 12)) and struct.calcsize("P") == 8
                  and sysconfig.get_platform() == "win-amd64")
    in_venv = sys.prefix != sys.base_prefix
    has_venv_tools = all(importlib.util.find_spec(m) is not None for m in ("venv", "ensurepip"))
    if not compatible:
        expected = 3
    elif reject_venv and in_venv:
        expected = 4
    elif reject_venv and not has_venv_tools:
        expected = 5
    else:
        expected = 0
    assert proc.returncode == expected, proc.stdout + proc.stderr


def test_run_bat_launch_sequence():
    text = _text(RUN_BAT)
    for needle in (r'set "VENV_PY=%ROOT%.venv\Scripts\python.exe"', '"%VENV_PY%" -m dragon_ball_powers %*',
                   "importlib.util", "find_spec", r"assets\models", "hand_landmarker.task",
                   'call ".venv\\Scripts\\activate.bat"', "setup.bat",
                   'set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"', 'set "PYTHONPATH=%ROOT%src"',
                   r"See the messages above and logs\dragon_ball_powers.log",
                   r"scripts\download_models.py"):
        assert needle in text, needle
    assert "[ERROR] Dragon Ball Powers exited with code %RC%." in text
    for module in ("mediapipe", "cv2", "pygame", "moderngl", "numpy"):
        assert f"'{module}'" in text
    # Order: venv check -> dependency check -> model check -> launch.
    order = [text.index(s) for s in ('if exist "%VENV_PY%"', "find_spec", "hand_landmarker.task",
                                     "-m dragon_ball_powers %*")]
    assert order == sorted(order)


def _dependency_check_code() -> str:
    line = next(l for l in _text(RUN_BAT).split("\r\n") if l.startswith('"%VENV_PY%" -c "'))
    return line[len('"%VENV_PY%" -c "'):-1]


def test_run_bat_dependency_check_code():
    code = _dependency_check_code()
    assert "%" not in code and '"' not in code
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    modules = ("mediapipe", "cv2", "pygame", "moderngl", "numpy")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    assert proc.returncode == (3 if missing else 0), proc.stdout + proc.stderr
    # Without site-packages every package is "missing": exit code 3 and a list.
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 3
    assert proc.stdout.strip() == "[ERROR] Python packages missing in .venv: " + ", ".join(modules)


def test_setup_bat_starts_command_names_with_call():
    # "py"/"python" can be batch file shims (pyenv-win): started without "call",
    # cmd.exe never returns to setup.bat. Every candidate therefore declares
    # whether it is a command name (where.exe + "call") or a quoted .exe path.
    lines = _text(SETUP_BAT).split("\r\n")
    candidates = 0
    for i, line in enumerate(lines):
        name = re.match(r'set "CAND=([^"]+)"$', line)
        if name:
            assert lines[i + 1] == f'set "CAND_EXE={name.group(1).split()[0]}"', line
            candidates += 1
        if re.match(r'set CAND="%\w+%"$', line):
            assert lines[i + 1] == 'set "CAND_EXE="', line
            candidates += 1
    assert candidates == 7
    text = "\n".join(lines)
    assert '"%SYS32%\\where.exe" /q %CAND_EXE% >nul 2>&1\nif errorlevel 1 goto :probe_not_on_path\n' \
           'call %CAND% -I -c "%PROBE_CODE%"' in text
    assert ':python_found\nset PY_CMD=%CAND%\nif defined CAND_EXE set "PY_CALL=1"\n' in text
    assert 'if defined PY_CALL goto :step2_venv_call\n%PY_CMD% -m venv "%VENV_DIR%"\n' in text
    assert ':step2_venv_call\n' in text and '\ncall %PY_CMD% -m venv ".venv"\n' in text
    for line in lines:  # nothing else starts py/python by name
        assert not re.match(r"(?i)\s*(py|python)(\.exe)?\s", line), line


def test_setup_bat_environment_hygiene():
    text = _text(SETUP_BAT)
    first_probe = text.index("call :probe")
    for line in ('set "ERRORLEVEL="', 'set "PYTHONHOME="', 'set "PYTHONPATH="', 'set "PIP_NO_INPUT=1"',
                 # a global pip "user = true" breaks every install into .venv
                 'set "PIP_USER=0"',
                 # read-only probes must never start a Python installation
                 'set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"', 'set "PYLAUNCHER_ALLOW_INSTALL="'):
        assert f"\r\n{line}\r\n" in text, line
        assert text.index(line) < first_probe, line


def test_run_bat_quiet_exit_codes_and_relative_activation():
    lines = _text(RUN_BAT).split("\r\n")
    launch = lines.index('"%VENV_PY%" -m dragon_ball_powers %*')
    error = next(i for i, line in enumerate(lines) if line.startswith("echo [ERROR] Dragon Ball Powers exited"))
    quiet = {m.group(1) for line in lines[launch:error]
             if (m := re.match(r'if "%RC%"=="(-?\d+)" exit /b', line))}
    # 0 = normal end, 130 = Ctrl+C (main.py returns 130 for KeyboardInterrupt),
    # -1073741510 = STATUS_CONTROL_C_EXIT. Everything else shows the error and pauses.
    assert quiet == {"0", "130", "-1073741510"}
    # "call" expands "%" and doubles "^" in its arguments once more: never pass it the
    # absolute folder, the current folder is the project folder anyway.
    assert 'if exist ".venv\\Scripts\\activate.bat" call ".venv\\Scripts\\activate.bat"' in lines
    assert not any(re.search(r'(?i)\bcall\s+"%ROOT%', line) for line in lines)


# ===================================================================== manifest


def _manifest_data() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_is_complete_and_pinned():
    data = _manifest_data()
    models = data["models"]
    assert {m["file"] for m in models} == set(EXPECTED_MODELS)
    for model in models:
        assert set(model) >= {"file", "url", "sha256", "size", "description", "required"}
        url, sha, size = EXPECTED_MODELS[model["file"]]
        assert model["url"] == url
        assert model["sha256"] == sha
        assert model["size"] == size and isinstance(model["size"], int)
        assert model["required"] is True
        assert model["description"].strip()
        assert re.fullmatch(r"[0-9a-f]{64}", model["sha256"])


@pytest.mark.parametrize("name", list(EXPECTED_MODELS))
def test_present_model_files_match_manifest(name: str):
    path = MODELS_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not downloaded")
    _, sha, size = EXPECTED_MODELS[name]
    assert path.stat().st_size == size
    assert _sha256(path) == sha


# ====================================================== download_models.py (CLI)


def _run_script(*args: str, cwd: Path | None = None, isolated: bool = False,
                timeout: float = 120) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "DBP_MODELS_DIR"}
    cmd = [sys.executable] + (["-I"] if isolated else []) + [str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)


def test_script_check_passes_on_project_models():
    present = _present_models()
    if not present:
        pytest.skip("no models downloaded")
    proc = _run_script("--check", "--models-dir", str(MODELS_DIR), "--only", *present)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in present:
        assert re.search(rf"OK\s+{re.escape(name)}", proc.stdout)


def test_script_runs_standalone_and_uses_only_the_stdlib(tmp_path: Path):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
    # Isolated mode, foreign working directory: no package, no PYTHONPATH needed.
    proc = _run_script("--help", cwd=tmp_path, isolated=True, timeout=60)
    assert proc.returncode == 0
    assert "--check" in proc.stdout and "--models-dir" in proc.stdout


def _copy_model(tmp_path: Path) -> tuple[str, Path]:
    present = _present_models()
    if not present:
        pytest.skip("no models downloaded")
    name = min(present, key=lambda n: EXPECTED_MODELS[n][2])  # smallest file
    target = tmp_path / "models"
    target.mkdir()
    shutil.copyfile(MODELS_DIR / name, target / name)
    return name, target / name


def test_script_check_detects_corrupted_copy(tmp_path: Path):
    name, copy = _copy_model(tmp_path)
    proc = _run_script("--check", "--manifest", str(MANIFEST), "--models-dir", str(copy.parent), "--only", name)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with copy.open("r+b") as handle:  # flip one byte in the middle: same size, other hash
        handle.seek(copy.stat().st_size // 2)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_CUR)
        handle.write(bytes([byte[0] ^ 0xFF]))
    proc = _run_script("--check", "--manifest", str(MANIFEST), "--models-dir", str(copy.parent), "--only", name)
    assert proc.returncode == 1
    assert "DAMAGED" in proc.stdout and "SHA256" in proc.stdout


def test_script_check_detects_truncated_and_missing_files(tmp_path: Path):
    name, copy = _copy_model(tmp_path)
    with copy.open("r+b") as handle:
        handle.truncate(1000)
    proc = _run_script("--check", "--models-dir", str(copy.parent))
    assert proc.returncode == 1
    assert re.search(rf"DAMAGED\s+{re.escape(name)} - size is 1,000 bytes", proc.stdout)
    for other in EXPECTED_MODELS:
        if other != name:
            assert re.search(rf"MISSING\s+{re.escape(other)}", proc.stdout)
    # --check never downloads anything
    assert sorted(p.name for p in copy.parent.iterdir()) == [name]


def test_script_usage_errors(tmp_path: Path):
    assert _run_script("--check", "--force").returncode == 2
    proc = _run_script("--check", "--only", "no_such_model")
    assert proc.returncode == 2 and "unknown model" in proc.stderr
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    proc = _run_script("--check", "--manifest", str(bad))
    assert proc.returncode == 2 and "not valid JSON" in proc.stderr


# ================================================= download_models.py (in-process)


@pytest.fixture(scope="module")
def dm():
    """The script imported as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location("dbp_download_models_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no scripts/__pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    yield module
    sys.modules.pop(spec.name, None)


@dataclass
class Route:
    body: bytes
    status: int = 200
    content_length: int | None = -1      # -1 = len(body), None = no header
    send_bytes: int | None = None        # write only this many bytes, then close


class ModelServer:
    """Tiny HTTP/1.0 server on 127.0.0.1 that serves configurable responses."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.hits: collections.Counter[str] = collections.Counter()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                path = self.path.lstrip("/")
                server.hits[path] += 1
                route = server.routes.get(path)
                if route is None:
                    self.send_error(404, "Not Found")
                    return
                self.send_response(route.status)
                if route.content_length is not None:
                    length = len(route.body) if route.content_length == -1 else route.content_length
                    self.send_header("Content-Length", str(length))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                data = route.body if route.send_bytes is None else route.body[:route.send_bytes]
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the client gave up on purpose (e.g. size check failed)

            def log_message(self, *args):  # keep test output clean
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{path}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = ModelServer()
    yield srv
    srv.close()


def _payload(size: int = 300_000, seed: int = 1) -> bytes:
    block = hashlib.sha256(str(seed).encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"models": entries}), encoding="utf-8")
    return path


def _entry(file: str, url: str, body: bytes, **extra) -> dict:
    entry = {"file": file, "url": url, "sha256": hashlib.sha256(body).hexdigest(), "size": len(body),
             "description": "test model", "required": True}
    entry.update(extra)
    return entry


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".part"))


def test_download_missing_model(dm, server, tmp_path, capsys):
    body = _payload()
    server.routes["a.task"] = Route(body)
    manifest = _write_manifest(tmp_path / "m.json", [_entry("a.task", server.url("a.task"), body)])
    models = tmp_path / "models"
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models)]) == 0
    assert (models / "a.task").read_bytes() == body
    assert _leftovers(models) == []
    assert server.hits["a.task"] == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "downloaded and verified" in out and "100%" in out
    # A second run finds the valid file and does not download again.
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models)]) == 0
    assert server.hits["a.task"] == 1


def test_download_replaces_damaged_file_and_force(dm, server, tmp_path, capsys):
    body = _payload(seed=2)
    server.routes["b.task"] = Route(body)
    manifest = _write_manifest(tmp_path / "m.json", [_entry("b.task", server.url("b.task"), body)])
    models = tmp_path / "models"
    models.mkdir()
    (models / "b.task").write_bytes(body[:-1] + b"\x00")  # same size, wrong hash
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models)]) == 0
    assert (models / "b.task").read_bytes() == body
    assert server.hits["b.task"] == 1
    assert "DAMAGED" in capsys.readouterr().out
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--force"]) == 0
    assert server.hits["b.task"] == 2
    assert (models / "b.task").read_bytes() == body


def test_download_rejects_wrong_content_and_keeps_nothing(dm, server, tmp_path, capsys):
    body = _payload(seed=3)
    server.routes["c.task"] = Route(_payload(seed=4))  # same size, different bytes
    manifest = _write_manifest(tmp_path / "m.json", [_entry("c.task", server.url("c.task"), body)])
    models = tmp_path / "models"
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--retries", "0"]) == 1
    assert not (models / "c.task").exists()
    assert _leftovers(models) == []
    out = capsys.readouterr().out
    assert "FAILED" in out and "SHA256" in out


def test_download_keeps_old_file_when_new_download_is_bad(dm, server, tmp_path):
    body = _payload(seed=5)
    server.routes["d.task"] = Route(_payload(seed=6))
    manifest = _write_manifest(tmp_path / "m.json", [_entry("d.task", server.url("d.task"), body)])
    models = tmp_path / "models"
    models.mkdir()
    old = b"old damaged content"
    (models / "d.task").write_bytes(old)
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--retries", "0"]) == 1
    assert (models / "d.task").read_bytes() == old  # never replaced by an unverified file
    assert _leftovers(models) == []


def test_failed_forced_download_keeps_valid_file(dm, server, tmp_path, capsys):
    body = _payload(seed=13)
    server.routes["k.task"] = Route(b"", status=503)
    manifest = _write_manifest(tmp_path / "m.json", [_entry("k.task", server.url("k.task"), body)])
    models = tmp_path / "models"
    models.mkdir()
    (models / "k.task").write_bytes(body)
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--force", "--retries", "0"]) == 0
    out = capsys.readouterr().out
    assert "HTTP 503" in out and "KEPT" in out
    assert (models / "k.task").read_bytes() == body


@pytest.mark.parametrize("route, message", [
    (Route(b"", status=404), "HTTP 404"),
    (Route(b"", status=407), "the proxy requires a login"),
    (Route(_payload(seed=7), content_length=123), "offers 123 bytes"),
    (Route(_payload(seed=7), send_bytes=1000), "incomplete download"),
    (Route(_payload(seed=7) + b"extra bytes", content_length=None), "more than the expected"),
], ids=["http-404", "http-407", "wrong-content-length", "truncated", "oversized"])
def test_download_reports_server_problems(dm, server, tmp_path, capsys, route, message):
    body = _payload(seed=7)
    server.routes["e.task"] = route
    manifest = _write_manifest(tmp_path / "m.json", [_entry("e.task", server.url("e.task"), body)])
    models = tmp_path / "models"
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--retries", "0"]) == 1
    out = capsys.readouterr().out
    assert message in out, out
    assert not (models / "e.task").exists()
    assert _leftovers(models) == []


def test_download_retries_transient_errors(dm, server, tmp_path, capsys, monkeypatch):
    body = _payload(seed=8)
    server.routes["f.task"] = Route(body, send_bytes=10)  # always truncated
    manifest = _write_manifest(tmp_path / "m.json", [_entry("f.task", server.url("f.task"), body)])
    monkeypatch.setattr(dm.time, "sleep", lambda seconds: None)
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(tmp_path / "models"),
                    "--retries", "2"]) == 1
    assert server.hits["f.task"] == 3
    out = capsys.readouterr().out
    assert "retrying" in out and "proxy" in out


def test_download_connection_refused_gives_network_hint(dm, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with socket.socket() as sock:  # a port that was free a moment ago: connection refused
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    body = _payload(seed=9)
    manifest = _write_manifest(tmp_path / "m.json",
                               [_entry("g.task", f"http://127.0.0.1:{port}/g.task", body)])
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(tmp_path / "models"),
                    "--retries", "0", "--timeout", "10"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "HTTPS_PROXY" in out


def test_describe_network_error_proxy_login(dm):
    # https:// through a proxy: a refused CONNECT is an OSError, not an HTTPError.
    exc = urllib.error.URLError(OSError("Tunnel connection failed: 407 Proxy Authentication Required"))
    message, retryable = dm.describe_network_error(exc, "https://storage.googleapis.com/m/x.task")
    assert retryable is False  # a missing login does not fix itself
    assert "storage.googleapis.com" in message and "USER:PASSWORD@" in message
    message, retryable = dm.describe_network_error(
        urllib.error.URLError(OSError("Tunnel connection failed: 502 Bad Gateway")), "https://h/x.task")
    assert retryable is True and "502 Bad Gateway" in message


class LoginProxy:
    """HTTP proxy on 127.0.0.1 that answers every CONNECT with 407 (login required)."""

    def __init__(self) -> None:
        self.tunnels: list[str] = []
        tunnels = self.tunnels

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_CONNECT(self):  # noqa: N802 - http.server API
                tunnels.append(self.path)
                self.send_response(407, "Proxy Authentication Required")
                self.send_header("Proxy-Authenticate", 'Basic realm="test"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_download_through_a_proxy_that_needs_a_login(dm, tmp_path, capsys, monkeypatch):
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dm.time, "sleep", lambda seconds: None)
    proxy = LoginProxy()
    try:
        monkeypatch.setenv("HTTPS_PROXY", proxy.url)
        monkeypatch.setenv("https_proxy", proxy.url)
        body = _payload(seed=14)
        manifest = _write_manifest(tmp_path / "m.json",
                                   [_entry("p.task", "https://models.example.invalid/p.task", body)])
        assert dm.main(["--manifest", str(manifest), "--models-dir", str(tmp_path / "models"),
                        "--retries", "2"]) == 1
    finally:
        proxy.close()
    out = capsys.readouterr().out
    assert "the proxy refused the connection to models.example.invalid" in out
    assert "HTTPS_PROXY=http://USER:PASSWORD@" in out
    assert proxy.tunnels == ["models.example.invalid:443"]  # not retried
    assert not (tmp_path / "models" / "p.task").exists()


def test_optional_model_failure_does_not_fail(dm, server, tmp_path, capsys):
    good, missing = _payload(seed=10), _payload(seed=11)
    server.routes["h.task"] = Route(good)
    manifest = _write_manifest(tmp_path / "m.json", [
        _entry("h.task", server.url("h.task"), good),
        _entry("i.task", server.url("i.task"), missing, required=False),
    ])
    models = tmp_path / "models"
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--retries", "0"]) == 0
    out = capsys.readouterr().out
    assert "WARNING: optional model(s) not available: i.task" in out
    assert dm.main(["--manifest", str(manifest), "--models-dir", str(models), "--check"]) == 0


def test_stale_part_files_are_cleaned_up(dm, server, tmp_path):
    body = _payload(seed=12)
    server.routes["j.task"] = Route(body)
    spec = dm.ModelSpec(file="j.task", url=server.url("j.task"),
                        sha256=hashlib.sha256(body).hexdigest(), size=len(body))
    models = tmp_path / "models"
    models.mkdir()
    stale, fresh = models / "j.task.old123.part", models / "j.task.new456.part"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"y")
    old = time.time() - 2 * dm.STALE_PART_AGE
    os.utime(stale, (old, old))
    dm.download_model(spec, models, opener=dm.build_opener(), stream=io.StringIO())
    assert not stale.exists()
    assert fresh.exists()  # may belong to a download running right now
    assert dm.verify_model(spec, models).ok


@pytest.mark.parametrize("entry, message", [
    ({"file": "../evil.task"}, "plain file name"),
    ({"file": "sub/evil.task"}, "plain file name"),
    ({"file": ".hidden"}, "plain file name"),
    ({"url": "http://example.com/x.task"}, "https://"),
    ({"url": "file:///etc/passwd"}, "https://"),
    ({"sha256": "abc"}, "64 hexadecimal"),
    ({"size": 0}, "positive integer"),
    ({"size": True}, "positive integer"),
    ({"required": "yes"}, "true or false"),
])
def test_manifest_validation(dm, tmp_path, entry, message):
    base = {"file": "ok.task", "url": "https://example.com/ok.task", "sha256": "0" * 64, "size": 1}
    base.update(entry)
    path = _write_manifest(tmp_path / "m.json", [base])
    with pytest.raises(dm.ManifestError, match=re.escape(message)):
        dm.load_manifest(path)


def test_manifest_rejects_duplicates_and_empty(dm, tmp_path):
    entry = {"file": "a.task", "url": "https://example.com/a", "sha256": "0" * 64, "size": 1}
    with pytest.raises(dm.ManifestError, match="twice"):
        dm.load_manifest(_write_manifest(tmp_path / "dup.json", [entry, dict(entry, file="A.TASK")]))
    with pytest.raises(dm.ManifestError, match="non-empty"):
        dm.load_manifest(_write_manifest(tmp_path / "empty.json", []))


def test_project_manifest_loads(dm):
    specs = dm.load_manifest(MANIFEST)
    assert [s.file for s in specs] == list(EXPECTED_MODELS)
    assert [s.file for s in dm.select_models(specs, ["HAND_LANDMARKER", "pose_landmarker_heavy.task"])] == [
        "hand_landmarker.task", "pose_landmarker_heavy.task"]
    with pytest.raises(dm.ManifestError, match="unknown model"):
        dm.select_models(specs, ["nope"])


def test_models_dir_resolution(dm, tmp_path, monkeypatch):
    manifest = tmp_path / "x" / "models.json"
    monkeypatch.delenv("DBP_MODELS_DIR", raising=False)
    assert dm.resolve_models_dir(None, manifest) == manifest.parent.resolve()
    monkeypatch.setenv("DBP_MODELS_DIR", str(tmp_path / "env"))
    assert dm.resolve_models_dir(None, manifest) == (tmp_path / "env").resolve()
    assert dm.resolve_models_dir(tmp_path / "cli", manifest) == (tmp_path / "cli").resolve()
    assert dm.resolve_models_dir(None, dm.DEFAULT_MANIFEST) == (tmp_path / "env").resolve()
    monkeypatch.delenv("DBP_MODELS_DIR")
    assert dm.resolve_models_dir(None, dm.DEFAULT_MANIFEST) == MODELS_DIR.resolve()


# ========================================================= docs and VS Code config


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_vscode_settings():
    settings = _json(VSCODE / "settings.json")
    assert settings["python.defaultInterpreterPath"] == "${workspaceFolder}\\.venv\\Scripts\\python.exe"
    assert settings["python.analysis.extraPaths"] == ["${workspaceFolder}/src"]
    assert settings["python.testing.pytestEnabled"] is True
    assert settings["python.testing.pytestArgs"] == ["tests"]
    assert settings["terminal.integrated.env.windows"]["PYTHONPATH"].endswith("src")
    assert settings["files.exclude"]["**/__pycache__"] is True
    assert settings["[bat]"]["files.eol"] == "\r\n"


def test_vscode_launch_configurations():
    launch = _json(VSCODE / "launch.json")
    configs = {c["name"]: c for c in launch["configurations"]}
    expected_args = {"Dragon Ball Powers": [], "Dragon Ball Powers (CPU renderer)": ["--renderer", "cpu"],
                     "Self-check": ["--check"]}
    assert set(configs) == set(expected_args)
    for name, args in expected_args.items():
        config = configs[name]
        assert config["type"] == "debugpy" and config["request"] == "launch"
        assert config["module"] == "dragon_ball_powers"
        assert config.get("args", []) == args
        assert config["cwd"] == "${workspaceFolder}"
        assert config["env"]["PYTHONPATH"] == "${workspaceFolder}/src"


def test_vscode_extensions():
    assert {"ms-python.python", "ms-python.debugpy"} <= set(_json(VSCODE / "extensions.json")["recommendations"])


def test_asset_readmes():
    models_readme = (MODELS_DIR / "README.md").read_text(encoding="utf-8")
    assert "Apache License 2.0" in models_readme and "download_models.py" in models_readme
    for name in EXPECTED_MODELS:
        assert name in models_readme
    fonts_dir = ROOT / "assets" / "fonts"
    fonts_readme = (fonts_dir / "README.md").read_text(encoding="utf-8")
    assert "SIL Open Font License, Version 1.1" in fonts_readme and "github.com/google/fonts" in fonts_readme
    for name in re.findall(r"`([\w-]+\.(?:ttf|txt))`", fonts_readme):
        assert (fonts_dir / name).is_file(), name
    assert "Bangers-OFL.txt" in fonts_readme and "Rajdhani-OFL.txt" in fonts_readme


# ======================================================== batch files under cmd.exe

#: Folder name with the characters that typically break batch files.
TRICKY_NAME = "dbp test (x86) & co!"


def _cmd_exe() -> str | None:
    if sys.platform == "win32":
        return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
    if sys.platform.startswith("linux") and shutil.which("cmd.exe") and shutil.which("wslpath"):
        return shutil.which("cmd.exe")
    return None


@pytest.fixture
def win_sandbox(tmp_path: Path):
    """(posix path, windows path) of an empty folder cmd.exe can use as working directory."""
    cmd = _cmd_exe()
    if cmd is None:
        pytest.skip("cmd.exe is not available")
    if sys.platform == "win32":
        yield tmp_path, str(tmp_path)
        return
    # WSL: cmd.exe cannot use Linux paths as its working directory -> Windows %TEMP%.
    try:
        win_temp = subprocess.run([cmd, "/d", "/c", "echo %TEMP%"], capture_output=True, text=True,
                                  timeout=60, cwd="/mnt/c" if Path("/mnt/c").is_dir() else None,
                                  stdin=subprocess.DEVNULL).stdout.strip().splitlines()[-1].strip()
        posix_temp = subprocess.run(["wslpath", "-u", win_temp], capture_output=True, text=True,
                                    timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        pytest.skip(f"cannot reach the Windows temp folder: {exc}")
    if not posix_temp or not Path(posix_temp).is_dir():
        pytest.skip("cannot reach the Windows temp folder")
    folder = Path(posix_temp) / f"dbp-tooling-test-{uuid.uuid4().hex[:10]}"
    folder.mkdir()
    try:
        yield folder, win_temp.rstrip("\\") + "\\" + folder.name
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class ProjectCopy(NamedTuple):
    proj: Path          # the copied project (folder name TRICKY_NAME), as a local path
    posix_root: Path    # the sandbox folder that contains it, as a local path
    win_root: str       # the sandbox folder as a Windows path (for cmd.exe)


def _write_batch(path: Path, lines: list[str]) -> None:
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))


def _write_wrapper(posix_root: Path, name: str, extra_lines: tuple[str, ...] = (),
                   no_pause: str = "1") -> None:
    # The wrapper starts the scripts from a different current directory; the dead
    # proxy is a safety net: nothing in these tests may reach the network.
    _write_batch(posix_root / name, [
        "@echo off",
        f'set "DBP_NO_PAUSE={no_pause}"',
        'set "HTTPS_PROXY=http://127.0.0.1:9"',
        'set "HTTP_PROXY=http://127.0.0.1:9"',
        *extra_lines,
        'cd /d "%TEMP%"',
        f'"%~dp0{TRICKY_NAME}\\%~1" %2 %3 %4 %5 %6',
    ])


@pytest.fixture
def project_copy(win_sandbox) -> ProjectCopy:
    """A minimal copy of the project (batch files + what they reference) in a tricky path."""
    posix_root, win_root = win_sandbox
    proj = posix_root / TRICKY_NAME
    (proj / "scripts").mkdir(parents=True)
    for name in ("setup.bat", "run.bat", "install_virtual_camera.bat", "install_win11_camera.bat", "requirements.txt",
                 "requirements-dev.txt"):
        shutil.copyfile(ROOT / name, proj / name)
    for arch in ("x64", "arm64"):
        (proj / "drivers" / "win11" / arch).mkdir(parents=True)
        shutil.copyfile(WIN11_DRIVERS / arch / "DragonBallPowersCamera.dll",
                        proj / "drivers" / "win11" / arch / "DragonBallPowersCamera.dll")
    shutil.copyfile(SCRIPT, proj / "scripts" / "download_models.py")
    _write_wrapper(posix_root, "wrap.bat")
    return ProjectCopy(proj, posix_root, win_root)


def _run_cmd(pc: ProjectCopy, batch: str, *args: str) -> tuple[int, str]:
    """Run ``<sandbox>\\<batch> args`` with cmd.exe; returns (exit code, stdout + stderr)."""
    cmd = _cmd_exe()
    batch_path = pc.win_root + "\\" + batch
    if sys.platform == "win32":
        line = f'"{cmd}" /d /v:off /s /c ""{batch_path}" {" ".join(args)}"'
        proc = subprocess.run(line, capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    else:
        proc = subprocess.run([cmd, "/d", "/v:off", "/c", batch_path, *args], cwd=pc.posix_root,
                              capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace").replace("\r\n", "\n")
    return proc.returncode, output


def _run_batch(pc: ProjectCopy, script: str, *args: str, wrapper: str = "wrap.bat") -> tuple[int, str]:
    return _run_cmd(pc, wrapper, script, *args)


def _with_fake_pythons(pc: ProjectCopy, shims: dict[str, str], no_pause: str = "1") -> str:
    """Put batch-file stand-ins for ``py`` / ``python`` on an otherwise Python-free PATH.

    ``shims`` maps a command name to the body of ``<name>.bat`` (CRLF is added).
    They behave like pyenv-win's ``python.bat`` shims, so they also prove that
    setup.bat comes back from probing a batch file. TEMP points to the empty
    folder ``<sandbox>\\tmp`` (see :func:`_temp_leftovers`). Returns the wrapper name.
    """
    fake_bin = pc.posix_root / "fakebin"
    fake_bin.mkdir(exist_ok=True)
    (pc.posix_root / "tmp").mkdir(exist_ok=True)
    for name, body in shims.items():
        _write_batch(fake_bin / f"{name}.bat", body.strip("\n").split("\n"))
    _write_wrapper(pc.posix_root, "wrap-fake.bat", (
        'set "PATH=%~dp0fakebin;%SystemRoot%\\System32"',
        'set "TEMP=%~dp0tmp"',
        'set "TMP=%~dp0tmp"',
    ), no_pause=no_pause)
    return "wrap-fake.bat"


def _temp_leftovers(pc: ProjectCopy) -> list[str]:
    """Files setup.bat left in its TEMP folder (its probe files must be cleaned up)."""
    return sorted(p.name for p in (pc.posix_root / "tmp").iterdir())


def _snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in folder.rglob("*")}


def test_run_bat_without_venv_shows_friendly_error(project_copy):
    code, out = _run_batch(project_copy, "run.bat", "--renderer", "cpu")
    assert code == 1, out
    assert "[ERROR] Dragon Ball Powers is not set up yet." in out
    assert "Run setup.bat first" in out
    assert TRICKY_NAME in out  # the folder is reported correctly despite "&", "(" and "!"


def test_virtual_camera_bat_help_and_bad_option(project_copy):
    code, out = _run_batch(project_copy, "install_virtual_camera.bat", "--help")
    assert code == 0, out
    for word in ("--uninstall", "--dry-run", "DBP_NO_PAUSE", "Exit codes", "Dragon Ball Powers"):
        assert word in out
    code, out = _run_batch(project_copy, "install_virtual_camera.bat", "--bogus")
    assert code == 2 and 'Unknown option "--bogus"' in out, out


def test_virtual_camera_bat_dry_run_changes_nothing(project_copy):
    before = _snapshot(project_copy.proj)
    code, out = _run_batch(project_copy, "install_virtual_camera.bat", "--dry-run")
    assert code == 0, out
    assert "UnityCaptureFilter64.dll" in out and "/i:UnityCaptureName=Dragon Ball Powers" in out
    assert "Dry run finished" in out
    code, out = _run_batch(project_copy, "install_virtual_camera.bat", "--uninstall", "--dry-run")
    assert code == 0 and "regsvr32 /u" in out, out
    code, out = _run_batch(project_copy, "install_virtual_camera.bat", "--uninstall")
    assert code == 0 and "Nothing to remove" in out, out      # no driver files: no UAC prompt
    assert _snapshot(project_copy.proj) == before                # nothing downloaded or created


def test_win11_camera_bat_help_bad_option_and_dry_run(project_copy):
    code, out = _run_batch(project_copy, "install_win11_camera.bat", "--help")
    assert code == 0 and "--uninstall" in out and "Windows 11" in out and "Exit codes" in out, out
    code, out = _run_batch(project_copy, "install_win11_camera.bat", "--bogus")
    assert code == 2 and 'Unknown option "--bogus"' in out, out
    before = _snapshot(project_copy.proj)
    code, out = _run_batch(project_copy, "install_win11_camera.bat", "--dry-run")
    assert code == 0 and "SHA256 verified" in out and "Dry run finished" in out, out
    assert re.search(r"Windows build \d+", out), out
    code, out = _run_batch(project_copy, "install_win11_camera.bat", "--uninstall", "--dry-run")
    assert code == 0 and "regsvr32 /s /n /u /i" in out, out
    assert _snapshot(project_copy.proj) == before


def test_setup_bat_help(project_copy):
    code, out = _run_batch(project_copy, "setup.bat", "--help")
    assert code == 0, out
    for word in ("--dry-run", "--force", "--no-pause", "DBP_NO_PAUSE", "Exit codes", "[6/6]"):
        assert word in out


def test_setup_bat_rejects_unknown_option(project_copy):
    code, out = _run_batch(project_copy, "setup.bat", "--bogus", "--dry-run")
    assert code == 2, out
    assert 'Unknown option "--bogus"' in out


def test_setup_bat_dry_run_changes_nothing(project_copy):
    proj = project_copy[0]
    (proj / ".venv" / "Lib").mkdir(parents=True)  # an incomplete old environment
    (proj / ".venv" / "Lib" / "keep.txt").write_text("x", encoding="utf-8")
    before = _snapshot(proj)
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", "--force")
    assert code == 0, out
    for step in range(1, 7):
        assert f"[{step}/6]" in out
    assert "DRY RUN" in out and "Dry run finished" in out
    assert "Nothing was downloaded, installed or changed." in out
    assert "would delete the folder .venv" in out
    assert "would run:" in out
    assert TRICKY_NAME in out
    assert _snapshot(proj) == before
    assert not (proj / ".python").exists()


# Batch-file stand-ins for the Python candidates (see _with_fake_pythons). They print
# what the real probe code would print and exit with its exit codes.
COMPATIBLE_PYTHON = """
@echo off
echo Python 3.12.7 (64-bit, win-amd64)
exit /b 0
"""
STORE_STUB = """
@echo off
1>&2 echo Python was not found; run without arguments to install from the Microsoft Store.
exit /b 9009
"""
BLOCKED_PYTHON = """
@echo off
1>&2 echo This program is blocked by group policy. For more information, contact your system administrator.
exit /b 1260
"""
VENV_PYTHON = """
@echo off
echo Python 3.12.1 (64-bit, win-amd64), virtual environment
if "%DBP_REJECT_VENV%"=="1" exit /b 4
exit /b 0
"""
PY_312_32BIT_311_MISSING = """
@echo off
if "%~1"=="-3.12" goto :v312
1>&2 echo No suitable Python runtime found
exit /b 103
:v312
echo Python 3.12.9 (32-bit, win32)
exit /b 3
"""
PY_WITHOUT_ANY_PYTHON = """
@echo off
1>&2 echo No installed Python found!
exit /b 112
"""
PY_312_OK = """
@echo off
if not "%~1"=="-3.12" exit /b 103
echo Python 3.12.10 (64-bit, win-amd64)
rem setup.bat must ask base interpreters to reject virtual environments
if "%DBP_REJECT_VENV%"=="1" exit /b 0
exit /b 99
"""


def test_setup_bat_comes_back_from_batch_file_shims(project_copy):
    # pyenv-win puts python.bat on PATH. Started without "call", cmd.exe left the
    # :probe subroutine for good: the rest of setup.bat ran with its output still
    # redirected into the probe files in TEMP (invisible, files left behind).
    wrapper = _with_fake_pythons(project_copy, {"python": COMPATIBLE_PYTHON})
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", wrapper=wrapper)
    assert code == 0, out
    assert _temp_leftovers(project_copy) == []
    assert "- py -3.12: not found" in out
    assert "py -3.11" not in out  # no py launcher at all: 3.11 is not asked for
    assert "- python on PATH: Python 3.12.7 (64-bit, win-amd64) - OK" in out
    assert "Using python on PATH (Python 3.12.7 (64-bit, win-amd64))" in out
    assert '[dry-run] would run: python -m venv ".venv"' in out
    assert "Dry run finished: a real run would perform the steps listed above." in out


def test_setup_bat_prefers_the_py_launcher(project_copy):
    wrapper = _with_fake_pythons(project_copy, {"py": PY_312_OK, "python": COMPATIBLE_PYTHON})
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", wrapper=wrapper)
    assert code == 0, out
    assert "- py -3.12: Python 3.12.10 (64-bit, win-amd64) - OK" in out
    assert "Using py -3.12 (Python 3.12.10 (64-bit, win-amd64))" in out
    assert '[dry-run] would run: py -3.12 -m venv ".venv"' in out
    assert "python on PATH" not in out  # the first usable candidate wins


@pytest.mark.parametrize("shims, expected, absent", [
    pytest.param({"py": PY_312_32BIT_311_MISSING, "python": STORE_STUB}, [
        "- py -3.12: Python 3.12.9 (32-bit, win32) - not supported (needs 64-bit Python 3.11 or 3.12)",
        "- py -3.11: the required Python is not installed (exit code 103)",
        '- python on PATH: only the Microsoft Store placeholder ("App execution alias"), not a real Python',
    ], [], id="32bit-py-and-store-stub"),
    pytest.param({"py": PY_WITHOUT_ANY_PYTHON, "python": VENV_PYTHON}, [
        "- py -3.12: the required Python is not installed (exit code 112)",
        "- python on PATH: Python 3.12.1 (64-bit, win-amd64), virtual environment - skipped "
        "(a virtual environment, not an installation)",
    ], ["py -3.11"], id="empty-py-launcher-and-venv-on-path"),
    pytest.param({"python": BLOCKED_PYTHON}, [
        "- py -3.12: not found",
        "- python on PATH: BLOCKED by Windows (App Control / group policy)",
    ], ["py -3.11", "Windows refused"], id="blocked-python-on-path"),
    pytest.param({}, ["- py -3.12: not found", "- python on PATH: not found"], ["py -3.11"], id="no-python"),
])
def test_setup_bat_python_search(project_copy, shims, expected, absent):
    wrapper = _with_fake_pythons(project_copy, shims)
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", wrapper=wrapper)
    assert code == 0, out
    assert _temp_leftovers(project_copy) == []
    for line in expected:
        assert line in out, out
    for text in absent:
        assert text not in out, out
    # Nothing usable: the private Python would be downloaded (the URL has one "%").
    assert "[dry-run] No compatible Python found. A real run would:" in out
    assert f"[dry-run]   download {PBS_URL}\n" in out
    assert f"check its SHA256 = {PBS_SHA256}" in out
    assert "Dry run finished: a real run would perform the steps listed above." in out


def _break_venv(proj: Path) -> None:
    scripts = proj / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("not a program", encoding="ascii")


def test_setup_bat_dry_run_recreates_a_broken_venv(project_copy):
    _break_venv(project_copy.proj)
    wrapper = _with_fake_pythons(project_copy, {"python": COMPATIBLE_PYTHON})
    before = _snapshot(project_copy.proj)
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", wrapper=wrapper)
    assert code == 0, out
    assert re.search(r"- existing \.venv: (does not start \(exit code -?\d+\)|BLOCKED by Windows)", out), out
    assert "The existing .venv cannot be used; it will be recreated." in out
    assert "[dry-run] would delete the folder .venv (it is inside the project)" in out
    assert '[dry-run] would run: python -m venv ".venv"' in out
    assert _snapshot(project_copy.proj) == before
    assert _temp_leftovers(project_copy) == []


def test_setup_bat_refuses_to_delete_a_junction(project_copy):
    pc = project_copy
    outside = pc.posix_root / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me", encoding="ascii")
    _write_batch(pc.posix_root / "junction.bat", [
        "@echo off",
        f'if "%~1"=="make" mklink /J "%~dp0{TRICKY_NAME}\\.venv" "%~dp0outside"',
        f'if "%~1"=="remove" rmdir "%~dp0{TRICKY_NAME}\\.venv"',
    ])
    code, out = _run_cmd(pc, "junction.bat", "make")
    if code != 0:
        pytest.skip(f"cannot create a junction here: {out}")
    try:
        wrapper = _with_fake_pythons(pc, {"python": COMPATIBLE_PYTHON})
        code, out = _run_batch(pc, "setup.bat", "--dry-run", "--force", wrapper=wrapper)
        assert code == 3, out
        assert "[ERROR] Refusing to delete" in out and "would delete" not in out
        assert "[ERROR] The virtual environment .venv could not be created." in out
        assert _temp_leftovers(pc) == []
    finally:
        _run_cmd(pc, "junction.bat", "remove")
    assert (outside / "precious.txt").read_text(encoding="ascii") == "keep me"


def test_run_bat_reports_a_broken_venv(project_copy):
    _break_venv(project_copy.proj)
    code, out = _run_batch(project_copy, "run.bat")
    assert code == 1, out
    assert ("[ERROR] The Python environment in .venv does not work (exit code" in out
            or "[ERROR] Windows blocked .venv\\Scripts\\python.exe" in out), out
    assert "setup.bat" in out
    assert "Starting Dragon Ball Powers" not in out


@pytest.mark.parametrize("script, args, no_pause, pauses", [
    ("run.bat", (), "", True),             # error while not set up: keep the window open
    ("run.bat", (), "1", False),
    ("setup.bat", ("--dry-run",), "", True),  # end of setup: keep the window open
    ("setup.bat", ("--dry-run",), "true", False),
    ("setup.bat", ("--dry-run", "--no-pause"), "", False),
    ("setup.bat", ("--help",), "", False),
], ids=["run-error", "run-DBP_NO_PAUSE", "setup", "setup-DBP_NO_PAUSE-true", "setup-no-pause", "setup-help"])
def test_batch_files_pause_unless_disabled(project_copy, script, args, no_pause, pauses):
    # stdin is at EOF, so "pause" prints its prompt ("Press any key to continue . . .",
    # localized, always ending in ". . .") and returns at once instead of waiting.
    wrapper = _with_fake_pythons(project_copy, {"python": COMPATIBLE_PYTHON}, no_pause=no_pause)
    code, out = _run_batch(project_copy, script, *args, wrapper=wrapper)
    assert code == (1 if script == "run.bat" else 0), out
    assert out.rstrip().endswith(". . .") is pauses, out[-300:]


def test_setup_bat_dry_run_dev_and_incomplete_project(project_copy):
    (project_copy.proj / "scripts" / "download_models.py").unlink()
    wrapper = _with_fake_pythons(project_copy, {"python": COMPATIBLE_PYTHON})
    code, out = _run_batch(project_copy, "setup.bat", "--dry-run", "--dev", wrapper=wrapper)
    assert code == 0, out
    assert "[4/6] Installing the Python packages from requirements-dev.txt ..." in out
    assert '-m pip install -r requirements-dev.txt' in out
    assert "[ERROR] scripts\\download_models.py is missing - the project folder is incomplete." in out
    assert "[6/6]" in out  # a dry run reports every step ...
    # ... and names the first one a real run would stop at
    assert "Dry run finished: a real run would stop at step 5 - scripts\\download_models.py is missing" in out


# ================================================================ shell scripts


SHELL_SCRIPTS = [SETUP_SH, RUN_SH]
PBS_UNIX_SHA256 = {
    "aarch64-apple-darwin": "ba32036ad8f4c434d97a1ee7feef8b7edc0fac16777639b97e39c74a108e4440",
    "x86_64-apple-darwin": "9686ddd8639b855905447ea4a16eb54a804a9ef12afbe7c9ecb1412d9c1fbc98",
    "x86_64-unknown-linux-gnu": "5d546fc004fc2435fa142eb8f42ed8de5eb4725963dc22f7e6671014a80eb560",
    "aarch64-unknown-linux-gnu": "34a43010cc05982d8b5debbad7d09f7b1200117ebf018f7acfd656895e408205",
}


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None or sys.platform == "win32":
        pytest.skip("bash (macOS / Linux) is not available")
    return bash


@pytest.mark.parametrize("path", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_script_is_lf_ascii_and_parses(path: Path):
    data = path.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data, "CR found: macOS / Linux need LF line endings"
    assert data.startswith(b"#!/usr/bin/env bash\n")
    proc = subprocess.run([_bash(), "-n", str(path)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_setup_sh_pins_the_same_python_as_setup_bat():
    text = SETUP_SH.read_text(encoding="ascii")
    assert 'PBS_VERSION="3.12.15+20261013"' in text and 'PBS_TAG="20261013"' in text
    assert "/releases/download/20261013/" in PBS_URL                     # setup.bat's release
    for target, sha in PBS_UNIX_SHA256.items():
        assert re.search(rf"{re.escape(target)}\)\s+echo \"{sha}\"", text), target


def test_setup_sh_probe_code_accepts_this_interpreter():
    code = re.search(r'(?m)^PROBE_CODE="(.*)"$', SETUP_SH.read_text(encoding="ascii")).group(1)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.stdout.startswith(f"Python {sys.version_info[0]}.{sys.version_info[1]}.")
    assert proc.returncode == (0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 3)


def test_shell_scripts_safe_paths(tmp_path: Path):
    bash = _bash()
    proj = tmp_path / "my project & (copy)"
    (proj / "scripts").mkdir(parents=True)
    for name in ("setup.sh", "run.sh", "requirements.txt", "requirements-dev.txt"):
        shutil.copyfile(ROOT / name, proj / name)
    shutil.copyfile(SCRIPT, proj / "scripts" / "download_models.py")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env["PATH"] = "/usr/bin:/bin"          # no python3.12 / python3.11 on PATH: forces the download path

    def run(script: str, *args: str) -> tuple[int, str]:
        proc = subprocess.run([bash, str(proj / script), *args], capture_output=True, text=True, env=env,
                              stdin=subprocess.DEVNULL, timeout=120)
        return proc.returncode, proc.stdout + proc.stderr

    code, out = run("setup.sh", "--help")
    assert code == 0 and "--dry-run" in out and "bash run.sh" in out, out
    code, out = run("setup.sh", "--bogus")
    assert code == 2 and 'Unknown option "--bogus"' in out, out
    code, out = run("run.sh", "--windowed")
    assert code == 1 and "is not set up yet" in out and str(proj) in out, out
    before = _snapshot(proj)
    code, out = run("setup.sh", "--dry-run")
    assert code == 0 and "Dry run finished" in out, out
    assert "[dry-run]" in out and "[6/6]" in out
    assert _snapshot(proj) == before                                    # nothing changed
