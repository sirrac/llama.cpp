#!/usr/bin/env bash
set -u

# Run from ~/llama.cpp:
# bash scripts/run-residency-core.sh models/YOUR_MOE_MODEL.gguf build/bin/llama-completion

MODEL="${1:?Usage: bash scripts/run-residency-core.sh MODEL.gguf [LLAMA_BINARY]}"
BIN="${2:-build/bin/llama-completion}"
OUTDIR="results"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: Model not found: $MODEL" >&2
    exit 1
fi

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: llama-completion binary not found or not executable: $BIN" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
source .venv/bin/activate

TAG="$(basename "$MODEL" .gguf)"
TAG="${TAG//[^a-zA-Z0-9_-]/_}"

echo "Model: $MODEL"
echo "Binary: $BIN"
echo "Results: $OUTDIR"
echo
echo "Phase 1: cold-cache comparisons, 5 repetitions per mode"

for MODE in mmap none mlock adaptive; do
    FILE="$OUTDIR/${TAG}_cold_${MODE}.jsonl"
    rm -f "$FILE"

    echo
    echo "=== Cold cache: $MODE ==="
    python scripts/residency-bench.py \
        --model "$MODEL" \
        --bin "$BIN" \
        --modes "$MODE" \
        --state cold \
        --n-predict 32 \
        --threads 8 \
        --repeat 5 \
        --timeout 600 \
        --output "$FILE"

    echo "Saved: $FILE"
done

echo
echo "Phase 2: warm-cache comparisons, 3 repetitions per mode"

for MODE in mmap adaptive; do
    FILE="$OUTDIR/${TAG}_warm_${MODE}.jsonl"
    rm -f "$FILE"

    echo
    echo "=== Warm cache: $MODE ==="
    python scripts/residency-bench.py \
        --model "$MODEL" \
        --bin "$BIN" \
        --modes "$MODE" \
        --state warm \
        --n-predict 32 \
        --threads 8 \
        --repeat 3 \
        --timeout 600 \
        --output "$FILE"

    echo "Saved: $FILE"
done

echo
echo "Finished. Check every run's rc value before interpreting results."
echo "Cold and warm JSONL files are in $OUTDIR/"
echo "Next: generate charts using scripts/plot-residency-results.py"