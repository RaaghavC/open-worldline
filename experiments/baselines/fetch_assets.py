"""Download explicitly listed public baseline artifacts and verify their bytes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repository, revision, artifacts, output, *, verify_only=False, manifest_payload=None):
    output = Path(output).resolve()
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)

    def one(artifact):
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact paths must stay inside the output directory")
        target = output / relative
        if not target.resolve().is_relative_to(output):
            raise ValueError("Artifact path follows a symlink outside the output directory")
        if not verify_only:
            target.parent.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(output):
            raise ValueError("Artifact path follows a symlink outside the output directory")
        if target.exists():
            if target.stat().st_size != artifact["size"] or file_hash(target) != artifact["sha256"]:
                raise FileExistsError(f"Existing file does not match the pinned artifact: {relative}")
        else:
            if verify_only:
                raise FileNotFoundError(f"Missing pinned artifact: {relative}")
            print(f"Downloading {relative} ({artifact['size']} bytes)", flush=True)
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{quote(relative.as_posix(), safe='/')}"
            # An exclusive random temporary file cannot reuse a stale .download
            # symlink. Publish only verified bytes, without replacing a file that
            # another process created while the download was running.
            descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".download", dir=target.parent)
            os.close(descriptor)
            partial = Path(name)
            try:
                subprocess.run([
                    "curl", "-q", "--fail", "--location", "--silent", "--show-error",
                    "--proto", "=https", "--proto-redir", "=https",
                    "--retry", "2", "--connect-timeout", "20", "--max-time", "1800",
                    url, "--output", str(partial),
                ], check=True)
                if partial.stat().st_size != artifact["size"] or file_hash(partial) != artifact["sha256"]:
                    raise ValueError(f"Downloaded artifact failed verification: {relative}")
                os.link(partial, target)
            finally:
                partial.unlink(missing_ok=True)
        return artifact

    checked = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, item) for item in artifacts]):
            item = future.result()
            checked.append(item)
            print(f"Verified {item['path']}", flush=True)
    if verify_only:
        return
    manifest = json.dumps(manifest_payload if manifest_payload is not None else {
        "repository": repository, "revision": revision,
        "artifacts": sorted(checked, key=lambda item: item["path"]),
    }, indent=2) + "\n"
    # Replacing the directory entry also avoids following an old manifest symlink.
    descriptor, name = tempfile.mkstemp(prefix=".manifest.", suffix=".json", dir=output)
    temporary_manifest = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(manifest)
        temporary_manifest.replace(output / "manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
