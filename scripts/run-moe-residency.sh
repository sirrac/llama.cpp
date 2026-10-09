```bash
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

MODEL="${1:-}"
BIN="${2:-build/bin/llama-completion}"

THREADS="${THREADS:-8}"
N_PREDICT="${N_PREDICT:-64}"
COLD_REPEAT="${COLD_REPEAT:-5}"
WARM_REPEAT="${WARM_REPEAT:-3}"
TIMEOUT="${TIMEOUT:-900}"

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash scripts/run-moe-residency.sh PATH/TO/MODEL.gguf [PATH/TO/llama-completion]"
    exit 2
fi

[[ "$MODEL" = /* ]] || MODEL="$ROOT/$MODEL"
[[ "$BIN" = /* ]] || BIN="$ROOT/$BIN"

for f in "$MODEL" "$BIN" scripts/residency-bench.py; do
    [[ -f "$f" ]] || {
        echo "ERROR: Missing required file: $f"
        exit 2
    }
done

[[ -x "$BIN" ]] || {
    echo "ERROR: Binary is not executable: $BIN"
    exit 2
}

STAMP="$(date +%Y%m%d-%H%M%S)"
TAG="$(basename "$MODEL" .gguf | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
OUTDIR="results/moe-residency/${TAG}_${STAMP}"
mkdir -p "$OUTDIR"

echo "Model: $MODEL"
echo "Binary: $BIN"
echo "Results: $OUTDIR"

run_case() {
    local state="$1"
    local mode="$2"
    local repeat="$3"

    echo "Running state=$state mode=$mode repeats=$repeat"

    python3 scripts/residency-bench.py \
        --model "$MODEL" \
        --bin "$BIN" \
        --modes "$mode" \
        --state "$state" \
        --n-predict "$N_PREDICT" \
        --threads "$THREADS" \
        --repeat "$repeat" \
        --timeout "$TIMEOUT" \
        --output "$OUTDIR/${state}_${mode}.jsonl"
}


run_case cold mmap "$COLD_REPEAT"
run_case cold adaptive "$COLD_REPEAT"

run_case warm mmap "$WARM_REPEAT"
run_case warm adaptive "$WARM_REPEAT"

if [[ -f scripts/plot-residency-results.py ]]; then
    python3 scripts/plot-residency-results.py \
        "$OUTDIR"/*.jsonl \
        --out-dir "$OUTDIR/plots"
fi

echo
echo "Benchmark finished."
echo "Results saved to: $OUTDIR"
```
