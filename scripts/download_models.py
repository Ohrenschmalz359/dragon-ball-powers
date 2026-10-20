#!/usr/bin/env python3
"""Download and verify the MediaPipe model files used by Dragon Ball Powers.

The models (file name, URL, size, SHA256) are listed in
``assets/models/models.json``. For every model this script

* keeps a file that is present and matches size and SHA256 ("OK"),
* otherwise downloads it into a temporary ``.part`` file next to the target,
  verifies size and SHA256 and then atomically moves it into place, so an
  interrupted or corrupted download never leaves a broken model behind.

Only the Python standard library is used, so the script runs before (and
without) the app's own dependencies; ``setup.bat`` runs it after ``pip install``.

Exit codes: 0 = every required model is present and verified, 1 = a required
model is missing, damaged or could not be downloaded, 2 = invalid arguments or
manifest, 130 = interrupted (Ctrl+C).

Examples::

    python scripts/download_models.py              # fetch what is missing or damaged
    python scripts/download_models.py --check      # verify only, no network access
    python scripts/download_models.py --only hand_landmarker --force
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import shutil
import socket
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence, TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "models" / "models.json"
#: Same override as ``dragon_ball_powers.config.MODELS_DIR``: the app and this
#: script must agree on where the models live.
MODELS_DIR_ENV = "DBP_MODELS_DIR"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

CHUNK_SIZE = 256 * 1024
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 2
#: ``.part`` files older than this are leftovers of killed runs and are removed.
STALE_PART_AGE = 3600.0
MB = 1024 * 1024
USER_AGENT = "DragonBallPowers-ModelDownloader/1.0 Python-urllib/{}.{}".format(*sys.version_info[:2])
PROXY_HINT = ("If this PC needs a proxy for internet access, set it first, for example "
              "'set HTTPS_PROXY=http://proxy.example.com:8080' (a proxy configured manually "
              "in the Windows settings is used automatically).")
PROXY_LOGIN_HINT = ("the proxy requires a login: set it with your credentials first, for example "
                    "'set HTTPS_PROXY=http://USER:PASSWORD@proxy.example.com:8080'")

_HEX_DIGITS = frozenset("0123456789abcdef")
_SAFE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class ManifestError(ValueError):
    """``models.json`` is missing or invalid, or a requested model is unknown."""


class DownloadError(RuntimeError):
    """A model could not be downloaded, or the download failed verification."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        #: True for transient problems (network hiccups) worth another attempt.
        self.retryable = retryable


@dataclass(frozen=True)
class ModelSpec:
    """One entry of the model manifest."""

    file: str
    url: str
    sha256: str
    size: int
    description: str = ""
    required: bool = True

    @property
    def stem(self) -> str:
        return Path(self.file).stem


@dataclass(frozen=True)
class Verification:
    """Result of checking one model file on disk."""

    status: str          # "ok" | "missing" | "size" | "hash" | "error"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ------------------------------------------------------------------ manifest


def load_manifest(path: Path) -> list[ModelSpec]:
    """Read and validate a model manifest (raises :class:`ManifestError`)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ManifestError(f"model manifest not found: {path}") from None
    except OSError as exc:
        raise ManifestError(f"cannot read the model manifest {path}: {exc}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from None
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f'{path}: expected a JSON object with a non-empty "models" list')
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        spec = _parse_entry(entry, f"{Path(path).name}: models[{index}]")
        key = spec.file.lower()  # Windows file names are case-insensitive
        if key in seen:
            raise ManifestError(f"{Path(path).name}: model file {spec.file!r} is listed twice")
        seen.add(key)
        specs.append(spec)
    return specs


def _parse_entry(entry: object, where: str) -> ModelSpec:
    if not isinstance(entry, dict):
        raise ManifestError(f"{where}: expected an object")
    for key in ("file", "url", "sha256", "size"):
        if key not in entry:
            raise ManifestError(f"{where}: missing field {key!r}")

    name = entry["file"]
    # A plain file name only: the manifest must never be able to write outside
    # the models folder ("../x", "C:\x", hidden names, ...).
    if (not isinstance(name, str) or not name or len(name) > 128 or name.startswith(".")
            or not set(name) <= _SAFE_NAME_CHARS):
        raise ManifestError(f"{where}: 'file' must be a plain file name (letters, digits, '._-'), got {name!r}")

    url = entry["url"]
    if not isinstance(url, str):
        raise ManifestError(f"{where}: 'url' must be a string")
    _check_url(url, where)

    digest = entry["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or not set(digest.lower()) <= _HEX_DIGITS:
        raise ManifestError(f"{where}: 'sha256' must be 64 hexadecimal characters")

    size = entry["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ManifestError(f"{where}: 'size' must be a positive integer (bytes)")

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ManifestError(f"{where}: 'description' must be a string")
    required = entry.get("required", True)
    if not isinstance(required, bool):
        raise ManifestError(f"{where}: 'required' must be true or false")
    return ModelSpec(file=name, url=url, sha256=digest.lower(), size=size,
                     description=description, required=required)


def _check_url(url: str, where: str) -> None:
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme == "https" and host:
        return
    # Plain http only for local test servers; real downloads always use TLS.
    if parts.scheme == "http" and _is_loopback(host):
        return
    raise ManifestError(f"{where}: 'url' must be an https:// address, got {url!r}")


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def select_models(specs: Sequence[ModelSpec], names: Sequence[str] | None) -> list[ModelSpec]:
    """Filter ``specs`` by file names or stems (case-insensitive); ``None`` = all."""
    if not names:
        return list(specs)
    by_name: dict[str, ModelSpec] = {}
    for spec in specs:
        by_name[spec.file.lower()] = spec
        by_name[spec.stem.lower()] = spec
    selected: list[ModelSpec] = []
    unknown: list[str] = []
    for name in names:
        spec = by_name.get(name.strip().lower())
        if spec is None:
            unknown.append(name)
        elif spec not in selected:
            selected.append(spec)
    if unknown:
        available = ", ".join(spec.file for spec in specs)
        raise ManifestError(f"unknown model(s): {', '.join(unknown)} (available: {available})")
    return selected


def resolve_models_dir(cli_value: Path | None, manifest: Path) -> Path:
    """``--models-dir`` > ``DBP_MODELS_DIR`` > the folder of the manifest."""
    if cli_value is not None:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get(MODELS_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(manifest).resolve().parent


# -------------------------------------------------------------- verification


def file_sha256(path: Path) -> tuple[str, int]:
    """Return ``(hex SHA256, size in bytes)`` of a file, reading it in chunks."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(4 * CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def verify_model(spec: ModelSpec, models_dir: Path) -> Verification:
    """Check presence, size and SHA256 of one model file."""
    path = Path(models_dir) / spec.file
    try:
        stat = path.stat()
    except FileNotFoundError:
        return Verification("missing", "not downloaded yet")
    except OSError as exc:
        return Verification("error", f"cannot access the file: {exc}")
    if not path.is_file():
        return Verification("error", "exists but is not a regular file")
    # The size check is free and catches truncated downloads without hashing.
    if stat.st_size != spec.size:
        return Verification("size", f"size is {stat.st_size:,} bytes instead of {spec.size:,} "
                                    "(incomplete or damaged)")
    try:
        digest, _ = file_sha256(path)
    except OSError as exc:
        return Verification("error", f"cannot read the file: {exc}")
    if digest != spec.sha256:
        return Verification("hash", f"SHA256 mismatch, the file is damaged or a different version "
                                    f"({digest[:12]}... instead of {spec.sha256[:12]}...)")
    return Verification("ok")


# ------------------------------------------------------------------ download


class _Progress:
    """Live single-line progress on a console; a few milestone lines otherwise."""

    def __init__(self, total: int, stream: TextIO) -> None:
        self.total = max(1, total)
        self.stream = stream
        try:
            self.interactive = bool(stream.isatty())
        except (AttributeError, ValueError):
            self.interactive = False
        self.started = time.perf_counter()
        self.last_draw = 0.0
        self.next_milestone = 25
        self.drawn = False

    def update(self, done: int) -> None:
        now = time.perf_counter()
        percent = min(100, done * 100 // self.total)
        if self.interactive:
            if now - self.last_draw < 0.1 and done < self.total:
                return
            self.last_draw = now
            self.stream.write("\r      " + self._text(done, percent, now))
            self.stream.flush()
            self.drawn = True
        elif percent >= self.next_milestone:
            while percent >= self.next_milestone:
                self.next_milestone += 25
            self.stream.write("      " + self._text(done, percent, now) + "\n")
            self.stream.flush()

    def finish(self) -> None:
        if self.drawn:
            self.stream.write("\n")
            self.stream.flush()
            self.drawn = False

    def _text(self, done: int, percent: int, now: float) -> str:
        speed = done / max(now - self.started, 1e-3)
        return f"{percent:3d}%  {done / MB:6.1f} / {self.total / MB:.1f} MB  {speed / MB:5.1f} MB/s"


def build_opener() -> urllib.request.OpenerDirector:
    """URL opener with certificate verification and the system proxy settings.

    A fresh opener (instead of ``urlopen``'s cached global one) picks up the
    current proxy environment (``HTTPS_PROXY`` / ``NO_PROXY``; on Windows also
    the manually configured proxy from the registry).
    """
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()))


def describe_network_error(exc: BaseException, url: str) -> tuple[str, bool]:
    """Turn a urllib/socket exception into ``(friendly message, retryable)``."""
    host = urllib.parse.urlsplit(url).hostname or url
    if isinstance(exc, urllib.error.HTTPError):
        message = f"the server {host} answered HTTP {exc.code} {exc.reason}"
        if exc.code in (403, 404, 410):
            message += " - the download address may have changed (update the project / models.json)"
        elif exc.code == 407:
            message += " - " + PROXY_LOGIN_HINT
        return message, exc.code >= 500 or exc.code in (408, 429)
    reason: object = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    # For https:// through a proxy, http.client reports a refused CONNECT tunnel as
    # OSError("Tunnel connection failed: 407 ..."), not as an HTTPError.
    if isinstance(reason, OSError) and str(reason).startswith("Tunnel connection failed: 407"):
        return f"the proxy refused the connection to {host} - {PROXY_LOGIN_HINT}", False
    if isinstance(reason, ssl.SSLCertVerificationError):
        return (f"the secure connection to {host} could not be verified ({reason.verify_message}). "
                "A proxy or antivirus that inspects HTTPS traffic can cause this; its certificate "
                "must be trusted by Windows"), False
    if isinstance(reason, ssl.SSLError):
        return f"the secure connection to {host} failed ({reason})", True
    if isinstance(reason, socket.gaierror):
        return f"cannot find the server {host} - is the PC connected to the internet?", True
    if isinstance(reason, TimeoutError):  # socket.timeout is an alias since 3.10
        return f"the connection to {host} timed out", True
    if isinstance(reason, ConnectionRefusedError):
        return f"the connection to {host} was refused", True
    if isinstance(reason, (OSError, http.client.HTTPException)):
        return f"network error while downloading from {host}: {reason}", True
    return f"download from {host} failed: {reason}", True


def download_model(spec: ModelSpec, models_dir: Path, *, opener: urllib.request.OpenerDirector,
                   timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
                   stream: TextIO | None = None) -> None:
    """Download ``spec`` into ``models_dir`` (verified, atomic). Raises :class:`DownloadError`."""
    stream = stream or sys.stdout
    models_dir = Path(models_dir)
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(f"cannot create the folder {models_dir}: {exc}") from exc
    _remove_stale_parts(spec, models_dir)
    _check_free_space(models_dir, spec.size)
    attempt = 0
    while True:
        attempt += 1
        try:
            _download_once(spec, models_dir, opener, timeout, stream)
            return
        except DownloadError as exc:
            if not exc.retryable or attempt > retries:
                raise
            wait = 2.0 * attempt
            stream.write(f"      {exc} - retrying in {wait:.0f} s (attempt {attempt + 1} of {retries + 1})\n")
            stream.flush()
            time.sleep(wait)


def _download_once(spec: ModelSpec, models_dir: Path, opener: urllib.request.OpenerDirector,
                   timeout: float, stream: TextIO) -> None:
    target = models_dir / spec.file
    try:
        fd, part_name = tempfile.mkstemp(prefix=spec.file + ".", suffix=".part", dir=models_dir)
    except OSError as exc:
        raise DownloadError(f"cannot write to {models_dir}: {exc}") from exc
    part = Path(part_name)
    progress = _Progress(spec.size, stream)
    try:
        try:
            with os.fdopen(fd, "wb") as out:
                digest, received = _fetch(spec, out, opener, timeout, progress)
        except DownloadError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            message, retryable = describe_network_error(exc, spec.url)
            raise DownloadError(message, retryable) from exc
        finally:
            progress.finish()
        if received != spec.size:
            raise DownloadError(f"incomplete download: received {received:,} of {spec.size:,} bytes "
                                "(the connection was interrupted)", retryable=True)
        if digest != spec.sha256:
            # Over HTTPS this means the server sends a different file; retrying won't help.
            raise DownloadError("the downloaded file failed the SHA256 check (the server sends a "
                                "different version than this project expects)")
        _apply_default_mode(part)
        _replace(part, target)
    finally:
        try:
            part.unlink()
        except FileNotFoundError:
            pass  # moved into place
        except OSError:
            pass  # e.g. still opened by a virus scanner; removed as stale next time


def _fetch(spec: ModelSpec, out: BinaryIO, opener: urllib.request.OpenerDirector,
           timeout: float, progress: _Progress) -> tuple[str, int]:
    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with opener.open(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        encoding = (response.headers.get("Content-Encoding") or "identity").strip().lower()
        if length and length.strip().isdigit() and encoding == "identity" and int(length) != spec.size:
            # Fail before downloading megabytes we would reject anyway.
            raise DownloadError(f"the server offers {int(length):,} bytes but {spec.size:,} are expected "
                                "(the file on the server changed; update the project / models.json)")
        digest = hashlib.sha256()
        received = 0
        while True:
            block = response.read(CHUNK_SIZE)
            if not block:
                break
            received += len(block)
            if received > spec.size:
                raise DownloadError(f"the server sent more than the expected {spec.size:,} bytes "
                                    "(not the file this project expects)")
            digest.update(block)
            try:
                out.write(block)
            except OSError as exc:
                raise DownloadError(f"cannot write the download to disk: {exc}") from exc
            progress.update(received)
    return digest.hexdigest(), received


def _replace(part: Path, target: Path) -> None:
    """Atomically move the verified download into place (retrying briefly on Windows locks)."""
    for attempt in range(6):
        try:
            os.replace(part, target)
            return
        except PermissionError as exc:
            # A virus scanner may still hold the fresh file, or the app has the old one open.
            if attempt == 5:
                raise DownloadError(f"cannot replace {target} ({exc.strerror}); close Dragon Ball "
                                    "Powers and try again") from exc
            time.sleep(0.25)
        except OSError as exc:
            raise DownloadError(f"cannot move the download to {target}: {exc}") from exc


def _apply_default_mode(path: Path) -> None:
    """Give the download normal file permissions (mkstemp creates it owner-only)."""
    if os.name == "nt":
        return  # Windows: the file inherits the folder's ACL; mode bits do not apply
    mask = os.umask(0)
    os.umask(mask)
    try:
        os.chmod(path, 0o666 & ~mask)
    except OSError:
        pass


def _remove_stale_parts(spec: ModelSpec, models_dir: Path) -> None:
    now = time.time()
    for part in models_dir.glob(f"{spec.file}.*.part"):
        try:
            if now - part.stat().st_mtime > STALE_PART_AGE:
                part.unlink()
        except OSError:
            pass


def _check_free_space(directory: Path, needed: int) -> None:
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return  # unknown: let the download try
    if free < needed + MB:
        raise DownloadError(f"not enough free disk space in {directory}: {free / MB:.1f} MB free, "
                            f"{needed / MB:.1f} MB needed")


# ---------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description="Download and verify the MediaPipe models used by Dragon Ball Powers "
                    "(listed in assets/models/models.json).",
        epilog="Exit codes: 0 = all required models OK, 1 = missing/damaged/download failed, "
               "2 = invalid arguments or manifest.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="only verify the files (size + SHA256), never download; "
                           "exit code 1 if a required model is missing or damaged")
    mode.add_argument("--force", action="store_true",
                      help="download again even if a file is present and valid")
    parser.add_argument("--only", nargs="+", action="extend", metavar="NAME",
                        help="process only these models (file name, with or without .task); "
                             "may be given more than once")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help=f"folder of the model files (default: ${MODELS_DIR_ENV} if set, "
                             "else the folder of the manifest = assets/models)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="model list to use (default: assets/models/models.json)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"network timeout in seconds (default: {DEFAULT_TIMEOUT:.0f})")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help=f"extra attempts after a network error (default: {DEFAULT_RETRIES})")
    return parser


def _make_output_safe() -> None:
    # Console output is ASCII, but paths may contain characters the code page
    # of a redirected stdout cannot encode; never crash because of that.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _line(status: str, spec: ModelSpec, text: str = "") -> str:
    return f"  {status:<9} {spec.file}" + (f" - {text}" if text else "")


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.retries < 0:
        parser.error("--retries must be 0 or more")
    _make_output_safe()
    out = sys.stdout

    try:
        specs = load_manifest(args.manifest)
        selected = select_models(specs, args.only)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    models_dir = resolve_models_dir(args.models_dir, args.manifest)
    print(f"Models folder: {models_dir}", file=out)

    opener = None if args.check else build_opener()
    failed: list[ModelSpec] = []
    try:
        for spec in selected:
            if args.force:
                print(_line("FORCED", spec, "downloading again (--force)"), file=out)
            else:
                result = verify_model(spec, models_dir)
                if result.ok:
                    print(_line("OK", spec, f"{spec.size / MB:.1f} MB, verified"), file=out)
                    continue
                label = {"missing": "MISSING", "size": "DAMAGED", "hash": "DAMAGED"}.get(result.status, "ERROR")
                print(_line(label, spec, result.detail), file=out)
                if opener is None:  # --check: report only, never touch the network
                    failed.append(spec)
                    continue
            host = urllib.parse.urlsplit(spec.url).hostname
            print(f"  {'':<9} downloading {spec.size / MB:.1f} MB from {host} ...", file=out)
            out.flush()
            try:
                download_model(spec, models_dir, opener=opener, timeout=args.timeout,
                               retries=args.retries, stream=out)
            except DownloadError as exc:
                print(_line("FAILED", spec, str(exc)), file=out)
                if exc.retryable:
                    print(f"  {'':<9} {PROXY_HINT}", file=out)
                # The replace is atomic, so after a failed --force the old file is untouched.
                if args.force and verify_model(spec, models_dir).ok:
                    print(_line("KEPT", spec, "the existing file is still valid and stays in use"), file=out)
                    continue
                failed.append(spec)
                continue
            print(_line("OK", spec, "downloaded and verified"), file=out)
    except KeyboardInterrupt:
        print("\nInterrupted - nothing half-downloaded was kept.", file=out)
        return EXIT_INTERRUPTED

    failed_required = [spec for spec in failed if spec.required]
    failed_optional = [spec for spec in failed if not spec.required]
    if failed_optional:
        print("WARNING: optional model(s) not available: "
              + ", ".join(spec.file for spec in failed_optional), file=out)
    if failed_required:
        names = ", ".join(spec.file for spec in failed_required)
        print(f"ERROR: {len(failed_required)} required model(s) missing or damaged: {names}", file=out)
        if args.check:
            print("Fix: run setup.bat, or: python scripts/download_models.py", file=out)
        else:
            print("Run this script (or setup.bat) again to retry.", file=out)
        return EXIT_FAILED
    ok_count = len(selected) - len(failed)
    if failed_optional:
        print(f"All required models are present and verified ({ok_count} of {len(selected)} OK).", file=out)
    else:
        print(f"All {ok_count} model(s) are present and verified.", file=out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
