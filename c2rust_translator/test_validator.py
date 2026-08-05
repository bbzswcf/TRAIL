from __future__ import annotations

import os
import re
import select
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple


MAX_TEST_OUTPUT_CAPTURE_BYTES = 2_000_000
MAX_TEST_OUTPUT_STREAM_BYTES = 20_000_000
TEST_OUTPUT_READ_CHUNK_BYTES = 65_536


class TestValidator:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)

    def snapshot_runtime_artifacts(self) -> Dict[str, Any]:
        return {
            "tmp": self._snapshot_tree(self.project_dir / "tmp"),
            "temp": self._snapshot_tree(self.project_dir / "temp"),
            "vsync": self._snapshot_tree(self.project_dir / ".vsync"),
            "src_bin_non_rust": self._snapshot_tree(
                self.project_dir / "src" / "bin",
                include=lambda path: path.is_dir() or path.suffix != ".rs",
            ),
        }

    def restore_runtime_artifacts(self, snapshot: Dict[str, Any]) -> None:
        if not snapshot:
            return
        self._restore_tree(self.project_dir / "tmp", snapshot.get("tmp"))
        self._restore_tree(self.project_dir / "temp", snapshot.get("temp"))
        self._restore_tree(self.project_dir / ".vsync", snapshot.get("vsync"))
        self._restore_tree(
            self.project_dir / "src" / "bin",
            snapshot.get("src_bin_non_rust"),
            include=lambda path: path.is_dir() or path.suffix != ".rs",
            preserve_root=True,
        )

    def run_all_tests(self, timeout: int = 120) -> Tuple[bool, str, Dict]:
        env = os.environ.copy()
        env["RUSTFLAGS"] = "-Awarnings"

        try:
            result = subprocess.Popen(
                ["cargo", "test", "--", "--nocapture", "--test-threads=1"],
                cwd=self.project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            output, timed_out, output_exceeded = _communicate_with_output_limit(result, timeout)
            if timed_out:
                timeout_output = f"Tests timeout after {timeout} seconds"
                if output:
                    timeout_output = f"{timeout_output}\n{output}"
                return False, timeout_output, {}
            if output_exceeded:
                output_limit = (
                    "Tests output exceeded "
                    f"{MAX_TEST_OUTPUT_STREAM_BYTES} bytes and was terminated"
                )
                if output:
                    output_limit = f"{output_limit}\n{output}"
                return False, output_limit, self._parse_test_stats(output)
        except Exception as exc:
            return False, f"Test execution error: {exc}", {}

        return result.returncode == 0, output, self._parse_test_stats(output)

    def _snapshot_tree(self, root: Path, include=None) -> Dict[str, Any]:
        if not root.exists() and not root.is_symlink():
            return {"exists": False, "entries": {}}

        entries: Dict[str, Dict[str, Any]] = {}
        paths = [root]
        if root.is_dir() and not root.is_symlink():
            paths.extend(sorted(root.rglob("*")))

        for path in paths:
            if include is not None and not include(path):
                continue
            rel = "." if path == root else path.relative_to(root).as_posix()
            if path.is_symlink():
                entries[rel] = {"type": "symlink", "target": os.readlink(path)}
            elif path.is_dir():
                entries[rel] = {"type": "dir"}
            elif path.is_file():
                entries[rel] = {"type": "file", "content": path.read_bytes()}

        return {"exists": True, "entries": entries}

    def _restore_tree(
        self,
        root: Path,
        snapshot: Dict[str, Any] | None,
        include=None,
        preserve_root: bool = False,
    ) -> None:
        if not snapshot:
            return

        if not snapshot.get("exists"):
            if root.exists() or root.is_symlink():
                if root.is_dir() and not root.is_symlink():
                    shutil.rmtree(root)
                else:
                    root.unlink()
            return

        entries: Dict[str, Dict[str, Any]] = snapshot.get("entries", {})
        if root.exists() and not root.is_symlink():
            current = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
            for path in current:
                if include is not None and not include(path):
                    continue
                rel = path.relative_to(root).as_posix()
                if rel in entries:
                    continue
                if path.is_dir() and not path.is_symlink():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                else:
                    path.unlink(missing_ok=True)
        elif root.exists() or root.is_symlink():
            root.unlink()

        for rel, entry in sorted(entries.items(), key=lambda item: item[0].count("/")):
            if rel == ".":
                path = root
            else:
                path = root / rel
            typ = entry.get("type")
            if typ == "dir":
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            if typ == "symlink":
                os.symlink(str(entry.get("target", "")), path)
            elif typ == "file":
                path.write_bytes(entry.get("content", b""))

        if not preserve_root and "." not in entries and root.exists() and root.is_dir():
            try:
                root.rmdir()
            except OSError:
                pass

    def _parse_test_stats(self, output: str) -> Dict:
        stats = {"total": 0, "passed": 0, "failed": 0, "ignored": 0}
        matches = re.findall(
            r"test result:\s*(?:ok|FAILED)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;\s*(\d+)\s+ignored",
            output or "",
        )

        non_zero = False
        for passed, failed, ignored in matches:
            p = int(passed)
            f = int(failed)
            i = int(ignored)
            if p + f + i == 0:
                continue
            non_zero = True
            stats["passed"] += p
            stats["failed"] += f
            stats["ignored"] += i

        if not non_zero and matches:
            passed, failed, ignored = matches[-1]
            stats["passed"] = int(passed)
            stats["failed"] = int(failed)
            stats["ignored"] = int(ignored)

        stats["total"] = stats["passed"] + stats["failed"] + stats["ignored"]
        if stats["total"] == 0:
            running = [int(value) for value in re.findall(r"running\s+(\d+)\s+tests", output or "")]
            if running:
                stats["total"] = max(running)
                if re.search(r"error:\s+test failed|FAILED|panicked at|stack overflow", output or ""):
                    stats["failed"] = max(stats["failed"], 1)
                    stats["passed"] = max(stats["total"] - stats["failed"] - stats["ignored"], 0)

        return stats


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        process.kill()


def _communicate_with_output_limit(
    process: subprocess.Popen,
    timeout: int,
) -> tuple[str, bool, bool]:
    if process.stdout is None:
        try:
            process.wait(timeout=timeout)
            return "", False, False
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            process.wait()
            return "", True, False

    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    deadline = time.monotonic() + timeout
    head_limit = MAX_TEST_OUTPUT_CAPTURE_BYTES // 2
    tail_limit = MAX_TEST_OUTPUT_CAPTURE_BYTES - head_limit
    head = bytearray()
    tail = bytearray()
    total_seen = 0

    def append(data: bytes) -> None:
        nonlocal total_seen
        if not data:
            return
        total_seen += len(data)
        if len(head) < head_limit:
            take = min(head_limit - len(head), len(data))
            head.extend(data[:take])
        tail.extend(data)
        if len(tail) > tail_limit:
            del tail[: len(tail) - tail_limit]

    def output_text() -> str:
        if total_seen <= MAX_TEST_OUTPUT_CAPTURE_BYTES:
            data = bytes(head)
            if len(head) < total_seen:
                data += bytes(tail)
            return data.decode("utf-8", errors="replace")
        omitted = total_seen - len(head) - len(tail)
        marker = (
            f"\n\n[... truncated {omitted} bytes of test output; "
            f"kept first {len(head)} and last {len(tail)} bytes ...]\n\n"
        ).encode()
        return (bytes(head) + marker + bytes(tail)).decode("utf-8", errors="replace")

    output_exceeded = False
    timed_out = False
    while True:
        if process.poll() is not None:
            while True:
                try:
                    data = os.read(fd, TEST_OUTPUT_READ_CHUNK_BYTES)
                except BlockingIOError:
                    break
                if not data:
                    break
                append(data)
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _kill_process_group(process)
            process.wait()
            break

        ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
        if not ready:
            continue

        try:
            data = os.read(fd, TEST_OUTPUT_READ_CHUNK_BYTES)
        except BlockingIOError:
            continue
        if not data:
            continue
        append(data)
        if total_seen > MAX_TEST_OUTPUT_STREAM_BYTES:
            output_exceeded = True
            _kill_process_group(process)
            process.wait()
            break

    return output_text(), timed_out, output_exceeded
