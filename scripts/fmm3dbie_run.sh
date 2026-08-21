#!/usr/bin/env bash

# Laplace SL self+near SETUP timing for the 2021 fmm3dbie locally-corrected quadrature,
# single-core:
#   2021 fmm3dbie getnearquad_lap_comb_dir (iquadtype=1 = GGQ self + adaptive near, dpars=(1,0) SLP)
#   on the paper's own stellarator (igeomtype=2, iasp=3/iref=2 -> N=2400 patches, p=8, 86400 nodes).
#   The harness sweeps eps over 1e-3,1e-5,1e-7,1e-9,1e-11 internally and prints, per eps,
#   S_slp = npts / t_slp (the Table-1c metric). t_slp excludes the near-list findnear and
#   get_far_order, both built once before the timer.
#
# This produces ONLY the fmm3dbie numbers. The SCTL comparison rows that used to live here have
# been dropped:
#   - The SCTL hybrid Y-bifurcation cold-Setup() row (former section B) is not run here.
#   - The SCTL twisted cubed-sphere cold-Setup() row (former section C) is subsumed by
#     scripts/bench-scheme-compare.sh, whose convergence table already reports the single-layer
#     setup throughput on a twisted cubed sphere (order 12, twist pi/6) for the Hybrid scheme
#     (and RP/Adaptive/Duffy) -- the same Nnodes/(SetupSingular+SetupNear) self+near metric.
#
# Single-core enforcement = OMP_NUM_THREADS=1 + MKL_NUM_THREADS=1 (no OpenMP/MKL threads)
#                           + taskset -c $CORE (pin the whole process to one physical core).
#
# Usage:  scripts/fmm3dbie_run.sh

#SBATCH --job-name=fmm3dbie
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=out/fmm3dbie-%j.log
#SBATCH --error=out/fmm3dbie-%j.log

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
source ./sctl_source
mkdir -p out

# --- pick ONE core from the cores actually AVAILABLE to this process (its CPU-affinity mask,
# /proc/self/status Cpus_allowed_list). Under Slurm this is EXACTLY the allocated cpuset; on a bare
# workstation it is all cores. Using the affinity mask -- not lscpu, which reports the whole node --
# guarantees the taskset target is inside the allocation, so pinning can never fail with EINVAL when
# Slurm hands out a high-numbered core. The first available core is skipped only when there is slack
# (>=4 cores), to keep OS/interrupt noise off the timed core.
CORES=()
_avail=$(grep -i '^Cpus_allowed_list' /proc/self/status 2>/dev/null | awk '{print $2}')
if [ -n "$_avail" ]; then
  IFS=',' read -ra _rng <<< "$_avail"
  for r in "${_rng[@]}"; do
    if [[ $r == *-* ]]; then CORES+=( $(seq "${r%-*}" "${r#*-}") ); else CORES+=( "$r" ); fi
  done
fi
[ "${#CORES[@]}" -eq 0 ] && mapfile -t CORES < <(seq 0 "$(($(nproc) - 1))")   # fallback: all logical CPUs
NCORES=${#CORES[@]}
CORE_BASE=$([ "$NCORES" -ge 4 ] && echo 1 || echo 0)
COREFMM=${CORES[$(( CORE_BASE % NCORES ))]}
echo "# available cores ($NCORES: ${CORES[*]}) -> COREFMM=$COREFMM"

# --- build the 2021 int4 harness against the fmm3d / fmm3dbie git submodules ---
# The submodules are pinned under bench/fmm3dbie/extern/{FMM3D,fmm3dbie}; their int4 static libs
# (lib-static/*.a) must be built once per bench/fmm3dbie/README.md. default-integer = int4, so it
# MUST link the int4 libs -- mixing with an int8 build segfaults in pts_tree3d. The makefile carries
# the matching -std=legacy -O3 -march=native flags.
BIN=bench/fmm3dbie/aquad_lap_2021
make -C bench/fmm3dbie aquad_lap_2021

# --- run single-threaded, pinned to ONE core ---
#   OMP_NUM_THREADS=1  -> getnearquad's parallel loops run serial
#   MKL_NUM_THREADS=1  -> no MKL/BLAS threads
#   OMP_PROC_BIND=true -> the (single) thread does not migrate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true \
taskset -c $COREFMM "$BIN"
