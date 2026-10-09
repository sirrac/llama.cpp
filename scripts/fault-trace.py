#!/usr/bin/env python3
# Break down major page faults on a GGUF model file by load phase and tensor class.
#
# record (perf_event_paranoid <= 2 is enough for a process started by perf):
#   perf record -e major-faults:u -d -k CLOCK_MONOTONIC -o run.data -- llama-completion -m model.gguf ... -v 2> run.log
# analyze:
#   fault-trace.py -m model.gguf -d run.data -l run.log

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

# keep in sync with llama_residency_classify(): routed experts are the MUL_MAT_ID weights
COLD_RE = re.compile(r"\.ffn_(gate|up|down|gate_up)_(ch)?exps\.")
CLASSES = ["other", "expert", "token_embd", "hot"]
PAGE = 4096


def page_classes(model: str) -> np.ndarray:
    cls = np.zeros((os.path.getsize(model) + PAGE - 1) // PAGE, dtype=np.uint8)
    for t in GGUFReader(model).tensors:
        c = CLASSES.index("expert" if COLD_RE.search(t.name) else "token_embd" if t.name.startswith("token_embd.") else "hot")
        a, b = t.data_offset // PAGE, (t.data_offset + t.n_bytes - 1) // PAGE + 1
        # a page shared by two tensors takes the hotter class
        cls[a:b] = np.maximum(cls[a:b], c)
    return cls


def log_ts(line: str) -> float:
    m, s, ms, us = (int(x) for x in line.split(" ", 1)[0].split("."))
    return m * 60 + s + ms / 1e3 + us / 1e6


def phases(log: str) -> tuple[float, float]:
    lines = [ln for ln in Path(log).read_text(errors="replace").splitlines() if re.match(r"^\d+\.\d+\.\d+\.\d+ ", ln)]
    load_end = [log_ts(ln) for ln in lines if "llama_context: constructing llama_context" in ln][-1]
    first_tok = next(log_ts(ln) for ln in lines if "n_remain:" in ln)
    return load_end, first_tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-d", "--data", required=True, help="perf.data with major-faults:u -d samples")
    ap.add_argument("-l", "--log", required=True, help="stderr of llama-completion -v (for phase boundaries)")
    args = ap.parse_args()

    out = subprocess.run(["perf", "script", "-i", args.data, "--show-mmap-events", "--show-task-events", "-F", "comm,pid,time,addr,event"],
                         capture_output=True, text=True, check=True).stdout
    name = Path(args.model).name
    t_exec = None
    base = None
    samples = []
    for ln in out.splitlines():
        if t_exec is None and "PERF_RECORD_COMM exec" in ln and "llama-completi" in ln:
            t_exec = float(re.search(r"\s(\d+\.\d+):", ln).group(1))
        elif "PERF_RECORD_MMAP2" in ln and ln.rstrip().endswith(name):
            m = re.search(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+|0) ", ln)
            base = (int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16))
        elif "major-faults" in ln and base is not None:
            m = re.search(r"\s(\d+\.\d+):\s+major-faults\S*:\s+([0-9a-f]+)", ln)
            if m:
                addr = int(m.group(2), 16)
                if base[0] <= addr < base[0] + base[1]:
                    samples.append((float(m.group(1)), addr - base[0] + base[2]))
    if t_exec is None or base is None:
        sys.exit("could not find the llama-completion exec or the model mmap in the trace")

    load_end, first_tok = phases(args.log)
    cls = page_classes(args.model)
    counts = {}
    for t, off in samples:
        rel = t - t_exec
        ph = "load" if rel < load_end else "prompt" if rel < first_tok else "decode"
        c = CLASSES[cls[off // PAGE]]
        counts[(ph, c)] = counts.get((ph, c), 0) + 1

    print(f"{Path(args.data).stem}: {len(samples)} major faults on {name} (load ends {load_end:.1f} s, first token {first_tok:.1f} s)")
    print(f"{'phase':8}" + "".join(f"{c:>12}" for c in reversed(CLASSES)) + f"{'total':>10}")
    for ph in ("load", "prompt", "decode"):
        row = [counts.get((ph, c), 0) for c in reversed(CLASSES)]
        print(f"{ph:8}" + "".join(f"{v:12d}" for v in row) + f"{sum(row):10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
