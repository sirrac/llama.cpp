#!/usr/bin/env python3
# Run llama-completion under each --load-mode and report load/prompt/eval timings, page faults,
# peak RSS and how many hot / cold (routed expert) bytes of the model file ended up in the page cache.
#
# Cold runs evict the model file with posix_fadvise(DONTNEED), which needs no root but only drops pages
# that no running process has mapped.

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

# keep in sync with llama_residency_classify(): routed experts are the MUL_MAT_ID weights
COLD_RE = re.compile(r"\.ffn_(gate|up|down|gate_up)_(ch)?exps\.")

PAGE = mmap.PAGESIZE
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def page_classes(model: str) -> np.ndarray:
    # 0 = metadata/padding, 1 = cold (only routed expert bytes), 2 = hot (holds any non-expert tensor byte)
    size = os.path.getsize(model)
    cls = np.zeros((size + PAGE - 1) // PAGE, dtype=np.uint8)
    reader = GGUFReader(model)
    split = reader.get_field("split.count")
    if split is not None and int(split.contents()) > 1:
        sys.exit(f"{model}: split models are not supported, merge them with llama-gguf-split --merge first")
    hot = []
    for t in reader.tensors:
        first, last = t.data_offset, t.data_offset + t.n_bytes
        if COLD_RE.search(t.name):
            cls[first // PAGE:(last - 1) // PAGE + 1] = np.maximum(cls[first // PAGE:(last - 1) // PAGE + 1], 1)
        else:
            hot.append((first, last))
    for first, last in hot:
        cls[first // PAGE:(last - 1) // PAGE + 1] = 2
    return cls


def resident_pages(model: str) -> np.ndarray:
    size = os.path.getsize(model)
    fd = os.open(model, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap failed")
        n_pages = (size + PAGE - 1) // PAGE
        vec = (ctypes.c_ubyte * n_pages)()
        if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        libc.munmap(ctypes.c_void_p(addr), size)
        return np.frombuffer(vec, dtype=np.uint8) & 1
    finally:
        os.close(fd)


def evict(model: str) -> None:
    fd = os.open(model, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def residency(model: str, cls: np.ndarray) -> dict:
    res = resident_pages(model)
    out = {}
    for name, c in (("hot", 2), ("cold", 1)):
        total = int(np.count_nonzero(cls == c))
        resident = int(np.count_nonzero(res[cls == c]))
        out[f"{name}_mib"] = total * PAGE / 2**20
        out[f"{name}_resident_mib"] = resident * PAGE / 2**20
    return out


def log_ts(line: str) -> float:
    # common log timestamps are m.ss.mmm.uuu since start
    m, s, ms, us = (int(x) for x in line.split(" ", 1)[0].split("."))
    return m * 60 + s + ms / 1e3 + us / 1e6


def parse(stderr: str) -> dict:
    out = {}
    lines = [ln for ln in stderr.splitlines() if re.match(r"^\d+\.\d+\.\d+\.\d+ ", ln)]
    # the first llama_context is created by the -fit dry run, the last one after the real load
    ctx = [ln for ln in lines if "llama_context: constructing llama_context" in ln]
    if ctx:
        out["load_ms"] = log_ts(ctx[-1]) * 1e3
    first = next((ln for ln in lines if "n_remain:" in ln), None)
    if first:
        out["ttft_ms"] = log_ts(first) * 1e3
    pats = {
        "prompt_ms": r"prompt eval time\s*=\s*([\d.]+) ms",
        "eval_tps": r"\beval time\s*=.*?([\d.]+) tokens per second",
        "wall_s": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.*)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): (\d+)",
        "major_faults": r"Major \(requiring I/O\) page faults: (\d+)",
        "minor_faults": r"Minor \(reclaiming a frame\) page faults: (\d+)",
    }
    for k, p in pats.items():
        m = re.findall(p, stderr)
        if m:
            v = m[-1]
            if k == "wall_s":
                parts = [float(x) for x in v.split(":")]
                v = sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
            out[k] = float(v)
    m = re.search(r"adaptive: hot ([\d.]+) MiB \(([\d.]+) MiB locked\), cold ([\d.]+) MiB, ([\d.]+) MiB copied", stderr)
    if m:
        out["adaptive"] = dict(zip(["hot_mib", "locked_mib", "cold_mib", "copied_mib"], map(float, m.groups())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--bin", default="build-cpu/bin/llama-completion")
    ap.add_argument("--modes", default="none,mmap,mlock,dio,adaptive")
    ap.add_argument("--state", choices=["cold", "warm"], default="cold", help="evict the model file before each run (cold) or not (warm)")
    ap.add_argument("-p", "--prompt", default="Explain how virtual memory works in an operating system.")
    ap.add_argument("-n", "--n-predict", type=int, default=32)
    ap.add_argument("-t", "--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--memory-max", help="run inside a systemd --user scope with this MemoryMax (e.g. 8G)")
    ap.add_argument("--extra", default="", help="extra arguments for llama-completion, pass as --extra=\"-nr ...\"")
    ap.add_argument("--timeout", type=float, default=600, help="seconds before a run is killed")
    ap.add_argument("-o", "--output", help="append JSON lines here")
    args = ap.parse_args()

    # mincore reports every page of a file as resident unless we own it or may write it
    if os.geteuid() != 0 and os.stat(args.model).st_uid != os.geteuid() and not os.access(args.model, os.W_OK):
        print(f"warning: {args.model} is not owned by you, mincore residency numbers will be meaningless", file=sys.stderr)

    cls = page_classes(args.model)
    results = []
    for mode in args.modes.split(","):
        for rep in range(args.repeat):
            if args.state == "cold":
                evict(args.model)
            before = residency(args.model, cls)
            cmd = []
            if args.memory_max:
                cmd += ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={args.memory_max}", "-p", "MemorySwapMax=0"]
            cmd += ["/usr/bin/time", "-v", args.bin, "-m", args.model, "-lm", mode, "-p", args.prompt,
                    "-n", str(args.n_predict), "-t", str(args.threads), "--temp", "0", "--seed", "1",
                    "-no-cnv", "--no-warmup", "--perf", "-v"] + args.extra.split()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = proc.communicate(timeout=args.timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
                rc = "timeout"
            r = {"mode": mode, "rep": rep, "state": args.state, "memory_max": args.memory_max, "extra": args.extra, "rc": rc}
            r.update(parse(stderr))
            r["resident_before"] = before
            r["resident_after"] = residency(args.model, cls)
            r["output"] = stdout[-400:]
            results.append(r)
            if rc != 0:
                print(stderr[-2000:], file=sys.stderr)
            if args.output:
                with open(args.output, "a") as f:
                    f.write(json.dumps(r) + "\n")

    hdr = f"{'mode':9} {'rc':>3} {'wall s':>7} {'load ms':>9} {'TTFT ms':>9} {'tok/s':>7} {'maj flt':>8} {'maxRSS MiB':>10} {'hot res/total MiB':>18} {'cold res/total MiB':>19}"
    print(hdr)
    for r in results:
        a = r["resident_after"]
        print(f"{r['mode']:9} {r['rc']:>3} {r.get('wall_s', 0):7.2f} {r.get('load_ms', 0):9.0f} {r.get('ttft_ms', 0):9.0f} {r.get('eval_tps', 0):7.2f} "
              f"{r.get('major_faults', 0):8.0f} {r.get('max_rss_kb', 0) / 1024:10.0f} "
              f"{a['hot_resident_mib']:8.0f}/{a['hot_mib']:<9.0f} {a['cold_resident_mib']:9.0f}/{a['cold_mib']:<9.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
