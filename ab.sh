#!/usr/bin/env bash
# Paired A/B benchmark of two trees of Layr-Labs/flock-challenge-multi on one
# GitHub-hosted Linux box, using the repo's committed trusted verifier.
# Env: BASE_REF, CAND_REF, GROUPS, RUNS, WARMUP, LOG2
set -euo pipefail

BASE_REF="${BASE_REF:?set BASE_REF}"
CAND_REF="${CAND_REF:?set CAND_REF}"
GROUPS="${GROUPS:-6}"
RUNS="${RUNS:-12}"
WARMUP="${WARMUP:-2}"
LOG2="${LOG2:-18}"

echo "=== CPU ==="
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core' || true
grep -o -m1 'avx512[a-z0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' ' || true
echo
echo "nproc=$(nproc)"

echo "=== toolchain ==="
cargo_env="${CARGO_HOME:-${HOME}/cargo/env}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
cargo_env="${CARGO_HOME}/env"
if [[ -f "$cargo_env" ]]; then . "$cargo_env"; fi
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  . "$cargo_env"
fi
rustup toolchain install 1.97.0 --profile minimal
rustc +1.97.0 --version

build_arm() {
  local dir="$1"
  (
    cd "$dir"
    . "$cargo_env"
    export CARGO_INCREMENTAL=0
    export CARGO_NET_RETRY=10
    cargo +1.97.0 fetch --locked --manifest-path Cargo.toml
    RUSTFLAGS="-C target-cpu=native" CARGO_INCREMENTAL=0 CARGO_NET_OFFLINE=true \
      cargo +1.97.0 build --locked --offline --profile challenge \
      --manifest-path Cargo.toml -p flock-benchmark-worker
    cp -f target/challenge-candidate/challenge/flock-benchmark-worker "../worker_${dir}"
  )
  chmod +x "worker_${dir}"
}

echo "=== building base ($BASE_REF) ==="
build_arm base
echo "=== building cand ($CAND_REF) ==="
build_arm cand
ls -l worker_base worker_cand

verifier="base/benchmark-tools/trusted/flock_benchmark_verifier"
chmod +x "$verifier"
mkdir -p results
threads="$(nproc)"
: > results/pairs.tsv

# Balanced alternation: odd groups run base first, even groups cand first.
for g in $(seq 1 "$GROUPS"); do
  if (( g % 2 == 1 )); then order=("base" "cand"); else order=("cand" "base"); fi
  for arm in "${order[@]}"; do
    rm -f "score_${arm}.json" "summary_${arm}.md"
    mkdir -p scratch
    if ! "$verifier" "worker_${arm}" scratch "score_${arm}.json" "summary_${arm}.md" \
        "$LOG2" "$threads" "$WARMUP" "$RUNS" >/dev/null 2>"results/${arm}_g${g}.log"; then
      echo "VERIFIER FAILED arm=$arm group=$g"
      tail -20 "results/${arm}_g${g}.log"
      exit 1
    fi
    python3 - "$arm" "$g" >> results/pairs.tsv <<'EOF'
import json, sys
arm, g = sys.argv[1], sys.argv[2]
with open(f"score_{arm}.json") as f:
    s = json.load(f)
med = s["metrics"]["median_seconds"]
score = s["score"]
print(f"{g}\t{arm}\t{med:.9f}\t{score:.6f}")
EOF
    tail -1 results/pairs.tsv
  done
done

echo "=== paired analysis (positive delta = cand SLOWER) ==="
python3 - "$GROUPS" <<'EOF'
import sys
groups = {}
with open("results/pairs.tsv") as f:
    for line in f:
        g, arm, med, score = line.split("\t")
        groups.setdefault(int(g), {})[arm] = (float(med), float(score))
base, cand = [], []
for g in sorted(groups):
    if "base" in groups[g] and "cand" in groups[g]:
        base.append(groups[g]["base"][0])
        cand.append(groups[g]["cand"][0])
n = len(base)
if n == 0:
    print("no paired groups")
    sys.exit(1)
ratios = [c / b for b, c in zip(base, cand)]
ratios.sort()
med_ratio = statistics.median(ratios)
mean_ratio = sum(ratios) / n
wins = sum(1 for r in ratios if r < 1.0)
import math
sd = statistics.stdev(ratios) if n > 1 else float("nan")
se = sd / math.sqrt(n) if n > 1 else float("nan")
print(f"groups={n} wins_cand={wins}/{n}")
print(f"median_ratio={med_ratio:.6f} ({(med_ratio-1)*100:+.3f}%)")
print(f"mean_ratio={mean_ratio:.6f} ({(mean_ratio-1)*100:+.3f}%)")
if not math.isnan(se):
    lo, hi = med_ratio - 1.25 * se, med_ratio + 1.25 * se
    print(f"median_ratio CI~[{lo:.6f},{hi:.6f}] (norm approx, x1.25)")
for g in sorted(groups):
    b = groups[g].get("base", (float('nan'),))[0]
    c = groups[g].get("cand", (float('nan'),))[0]
    if b == b and c == c:
        print(f"  group {g}: base={b*1000:.3f} ms cand={c*1000:.3f} ms ratio={c/b:.6f}")
EOF
