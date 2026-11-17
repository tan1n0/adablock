"""Run a small CUDA workload only while selected GPUs have no other processes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
DEFAULT_PID_FILE = ROOT / "tmp/gpu_keepalive.pid"
DEFAULT_LOG_FILE = ROOT / "tmp/gpu_keepalive.log"


@dataclass
class Worker:
    process: subprocess.Popen
    ready_file: Path
    baseline_pids: set[int]
    started: float
    host_pid: int | None = None


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def gpu_snapshot() -> dict[str, set[int]]:
    """Return GPU UUIDs and all process IDs reported by nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
            check=True, timeout=10,
        )
        root = ET.fromstring(result.stdout)
        snapshot = {}
        for gpu in root.findall("gpu"):
            uuid = gpu.findtext("uuid")
            if not uuid:
                raise ValueError("missing GPU UUID")
            snapshot[uuid] = {
                int(process.findtext("pid"))
                for process in gpu.findall("./processes/process_info")
            }
        if not snapshot:
            raise ValueError("no GPUs found")
        return snapshot
    except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError, TypeError) as exc:
        raise RuntimeError(f"cannot inspect GPU processes: {exc}") from exc


def parse_devices(value: str, count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(count))
    try:
        devices = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--devices must be 'all' or comma-separated GPU indices") from exc
    if not devices or len(set(devices)) != len(devices) or any(index < 0 or index >= count for index in devices):
        raise ValueError(f"--devices must contain unique indices in 0..{count - 1}")
    return devices


def run_worker(args: argparse.Namespace) -> None:
    # Imported only by workers: the supervisor never creates a CUDA context.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the worker")
    device = torch.device("cuda:0")
    with torch.no_grad():
        a = torch.randn(args.matrix_size, args.matrix_size, device=device)
        b = torch.randn_like(a)
        output = torch.empty_like(a)
        torch.mm(a, b, out=output)
        torch.cuda.synchronize(device)
        args._ready_file.write_text("ready")
        cycle = 0
        while True:
            cycle_started = time.monotonic()
            work_deadline = cycle_started + args.work_ms / 1000
            operations = 0
            while time.monotonic() < work_deadline:
                torch.mm(a, b, out=output)
                torch.cuda.synchronize(device)
                operations += 1
            cycle += 1
            if cycle % args.log_every == 0:
                log(f"GPU {args._worker}: cycle={cycle} operations={operations}")
            remaining = args.interval - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)


def stop_worker(worker: Worker) -> None:
    process = worker.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    worker.ready_file.unlink(missing_ok=True)


def start_worker(uuid: str, args: argparse.Namespace, baseline_pids: set[int]) -> Worker:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    ready_fd, ready_path = tempfile.mkstemp(prefix="gpu_keepalive_")
    os.close(ready_fd)
    command = [
        sys.executable, str(SCRIPT), "--_worker", uuid, "--_ready-file", ready_path,
        "--interval", str(args.interval), "--work-ms", str(args.work_ms),
        "--matrix-size", str(args.matrix_size), "--log-every", str(args.log_every),
    ]
    try:
        process = subprocess.Popen(command, env=environment)
    except OSError:
        Path(ready_path).unlink(missing_ok=True)
        raise
    return Worker(process, Path(ready_path), set(baseline_pids), time.monotonic())


def remove_own_pid_file(path: Path) -> None:
    try:
        if int(path.read_text().strip()) == os.getpid():
            path.unlink()
    except (FileNotFoundError, ValueError, OSError):
        pass


def supervise(args: argparse.Namespace) -> None:
    workers: dict[str, Worker] = {}
    retry_after: dict[str, float] = {}
    busy_pids: dict[str, set[int]] = {}
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    started = time.monotonic()
    deadline = started + args.duration if args.duration is not None else float("inf")
    try:
        first_snapshot = gpu_snapshot()
        selected = parse_devices(args.devices, len(first_snapshot))
        uuids = list(first_snapshot)
        log(f"supervisor started: GPUs={selected}, poll={args.poll_interval}s")
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                snapshot = gpu_snapshot()
            except RuntimeError as exc:
                log(f"{exc}; stopping all workers until monitoring recovers")
                for uuid, worker in list(workers.items()):
                    stop_worker(worker)
                    del workers[uuid]
                stop_event.wait(args.poll_interval)
                continue
            for index in selected:
                uuid = uuids[index]
                pids = snapshot.get(uuid)
                worker = workers.get(uuid)
                if worker is not None and worker.process.poll() is not None:
                    log(f"GPU {index}: worker exited with code {worker.process.returncode}")
                    stop_worker(worker)
                    del workers[uuid]
                    retry_after[uuid] = time.monotonic() + 10
                    worker = None
                if worker is not None and worker.host_pid is None:
                    if worker.ready_file.read_text().strip() == "ready":
                        new_pids = (pids or set()) - worker.baseline_pids
                        if len(new_pids) == 1:
                            worker.host_pid = new_pids.pop()
                            log(f"GPU {index}: worker GPU PID identified as {worker.host_pid}")
                        elif len(new_pids) > 1 or time.monotonic() - worker.started > 10:
                            log(f"GPU {index}: cannot identify worker among {sorted(new_pids)}; yielding")
                            stop_worker(worker)
                            del workers[uuid]
                            retry_after[uuid] = time.monotonic() + 10
                    continue
                if worker is not None:
                    other_pids = (pids or set()) - {worker.host_pid}
                    if pids is None:
                        other_pids.add(-1)  # A missing GPU is never treated as idle.
                    if other_pids:
                        log(f"GPU {index}: other process detected {sorted(other_pids)}; yielding")
                        stop_worker(worker)
                        del workers[uuid]
                    continue
                if pids:
                    if busy_pids.get(uuid) != pids:
                        log(f"GPU {index}: occupied by {sorted(pids)}; waiting")
                        busy_pids[uuid] = set(pids)
                    continue
                if pids is not None and time.monotonic() >= retry_after.get(uuid, 0):
                    busy_pids.pop(uuid, None)
                    workers[uuid] = start_worker(uuid, args, pids)
                    log(f"GPU {index}: idle; worker started (container PID {workers[uuid].process.pid})")
            stop_event.wait(min(args.poll_interval, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        log("interrupt received")
    finally:
        for worker in workers.values():
            stop_worker(worker)
        if args._supervisor:
            remove_own_pid_file(args.pid_file)
        log("supervisor stopped")


def daemon_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if str(SCRIPT).encode() in command and b"--_supervisor" in command:
            return pid
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def start_daemon(args: argparse.Namespace) -> None:
    if (pid := daemon_pid(args.pid_file)) is not None:
        print(f"GPU keepalive is already running (PID {pid})")
        return
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SCRIPT), "--_supervisor", "--devices", args.devices,
        "--interval", str(args.interval), "--work-ms", str(args.work_ms),
        "--matrix-size", str(args.matrix_size), "--poll-interval", str(args.poll_interval),
        "--log-every", str(args.log_every), "--pid-file", str(args.pid_file),
    ]
    if args.duration is not None:
        command.extend(["--duration", str(args.duration)])
    with args.log_file.open("a") as output:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    args.pid_file.write_text(f"{process.pid}\n")
    print(f"GPU keepalive started in background (PID {process.pid}); log: {args.log_file}")


def stop_daemon(path: Path) -> None:
    pid = daemon_pid(path)
    if pid is None:
        path.unlink(missing_ok=True)
        print("GPU keepalive is not running")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if daemon_pid(path) is None:
            print("GPU keepalive stopped")
            return
        time.sleep(0.1)
    print(f"stop requested for PID {pid}; check again with --status")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--daemon", action="store_true", help="start in background")
    mode.add_argument("--stop", action="store_true", help="stop the background supervisor")
    mode.add_argument("--status", action="store_true", help="show background supervisor status")
    parser.add_argument("--devices", default="0", help="physical GPU indices, e.g. 0,1, or 'all' (default: 0)")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between workload cycles (default: 10)")
    parser.add_argument("--work-ms", type=float, default=50.0, help="compute time per cycle in ms (default: 50)")
    parser.add_argument("--matrix-size", type=int, default=512, help="square matrix dimension (default: 512)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="seconds between process checks (default: 2)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds; default: run until stopped")
    parser.add_argument("--log-every", type=int, default=6, help="worker progress log interval in cycles (default: 6)")
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE, help="background PID file")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="background log file")
    parser.add_argument("--_supervisor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--_ready-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.stop:
            stop_daemon(args.pid_file)
        elif args.status:
            pid = daemon_pid(args.pid_file)
            print(f"GPU keepalive running (PID {pid})" if pid else "GPU keepalive is not running")
        else:
            if min(args.interval, args.work_ms, args.matrix_size, args.poll_interval, args.log_every) <= 0:
                raise ValueError("interval, work-ms, matrix-size, poll-interval, and log-every must be positive")
            if args.duration is not None and args.duration <= 0:
                raise ValueError("duration must be positive")
            if args._worker:
                run_worker(args)
            elif args.daemon:
                start_daemon(args)
            else:
                supervise(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
