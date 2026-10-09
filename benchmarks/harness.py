import subprocess
import time
import re
import json
import os
import argparse

def parse_benchmark_output(stdout: str, stderr: str) -> dict:
    raw_output = stdout + "\n" + stderr

    metrics = {
        "peak_rss_mb": None,
        "major_faults": 0,
        "minor_faults": 0,
    }

    # 1. Parse Major & Minor Page Faults from perf stat
    maj_faults = re.search(r'([\d,.]+)\s+major-faults', raw_output)
    min_faults = re.search(r'([\d,.]+)\s+minor-faults', raw_output)
    if maj_faults:
        metrics["major_faults"] = int(re.sub(r'[,.]', '', maj_faults.group(1)))
    if min_faults:
        metrics["minor_faults"] = int(re.sub(r'[,.]', '', min_faults.group(1)))

    # 2. Parse Peak RSS from GNU /usr/bin/time -v
    rss_match = re.search(r'Maximum resident set size \(kbytes\):\s+(\d+)', raw_output)
    if rss_match:
        metrics["peak_rss_mb"] = round(int(rss_match.group(1)) / 1024.0, 2)

    return metrics

def run_profile_trial(model_path: str, load_mode: str = "mmap", memory_limit_mb: int = None, prompt: str = "Explain virtual memory in two sentences."):
    cmd = []
    
    # 1. Memory constraint wrapper
    if memory_limit_mb:
        cmd.extend(["systemd-run", "--quiet", "--user", "--scope", "-p", f"MemoryMax={memory_limit_mb}M"])

    # 2. System counter tools + llama-cli
    cmd.extend([
        "/usr/bin/time", "-v",
        "perf", "stat", "-e", "page-faults,major-faults,minor-faults",
        "./build/bin/llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", "32",
        "-st"
    ])

    # 3. Load mode flags
    if load_mode in ["no-mmap", "none"]:
        cmd.append("--no-mmap")
    elif load_mode == "mlock":
        cmd.append("--mlock")
    elif load_mode == "adaptive":
        cmd.append("--adaptive-mmap")
    elif load_mode == "mmap":
        cmd.append("--mmap")

    print(f"Executing Trial: Mode={load_mode} | MemLimit={memory_limit_mb or 'Unlimited'}MB...")
    
    start_wall = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    elapsed_wall = time.time() - start_wall

    metrics = parse_benchmark_output(res.stdout, res.stderr)
    metrics["total_wall_time_s"] = round(elapsed_wall, 3)
    metrics["load_mode"] = load_mode
    metrics["memory_limit_mb"] = memory_limit_mb or "unlimited"

    print("--- Trial Results ---")
    print(f"Wall Time:     {metrics['total_wall_time_s']} s")
    print(f"Peak RSS:      {metrics['peak_rss_mb']} MB")
    print(f"Major Faults:  {metrics['major_faults']}")
    print(f"Minor Faults:  {metrics['minor_faults']}")
    print("---------------------\n")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama.cpp OS Residency Benchmark Harness")
    parser.add_argument("--model", type=str, default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf", help="Path to GGUF model file")
    parser.add_argument("--mode", type=str, default="mmap", choices=["mmap", "no-mmap", "none", "mlock", "adaptive"], help="Model load mode")
    parser.add_argument("--mem-limit", type=int, default=None, help="Memory limit in MB (via systemd cgroups)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    result = run_profile_trial(args.model, load_mode=args.mode, memory_limit_mb=args.mem_limit)
    
    out_filename = f"results/{args.mode}_{args.mem_limit or 'unlimited'}MB.json"
    with open(out_filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved run data to {out_filename}")