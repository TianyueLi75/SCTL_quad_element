# fmm3dbie Laplace S_init benchmark (stellarator)

The fmm3dbie side of a head-to-head near-quadrature *setup*-time comparison against the
SCTL BIE solver in this repo. It reproduces the S_init (Table 1c) metric of Greengard,
O'Neil, Rachh et al. 2021 for the **single-layer Laplace** kernel.

- **Driver:** `aquad_lap_stell_perf_test.f` (fmm3dbie 2021, int4).
- **Geometry:** the paper's own stellarator, generated internally (`setup_geom`,
  igeomtype=2, iasp=3 / iref=2) → **2400 patches, norder=7 (p=8), 86400 nodes**. No input
  files are needed — the surface is built in-process.
- **Timed quantity:** `getnearquad_lap_comb_dir` with dpars=(1,0) (SLP Laplace), near list
  fixed (rfac=2.75, rfac0=1.25). Reports `S_slp = npts / t_slp` for each eps in the sweep
  1e-3, 1e-5, 1e-7, 1e-9, 1e-11.
- This is the apples-to-apples counterpart of SCTL's `SetupSingular + SetupNear` phases.

## Run (standalone)

The prebuilt `aquad_lap_2021` runs with no input files. For paper-comparable, single-core
numbers:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <core> ./aquad_lap_2021
```

It prints `npatches`/`npts` and a table of `eps  npts  t_slp(s)  S_slp(pts/s)`.

## Rebuild

The binary is git-ignored (kept in the working tree only). To rebuild you need the 2021
**int4** static libs — `fmm3dbie-2021` and its paired `FMM3D-2021` — which are *not*
vendored here. They live in the `quad-junctions` checkout; point the build at them:

```
module load gcc/13.3.0                 # gfortran + MKLROOT   (or: . <repo>/sctl_source)
make QJ_EXTERN=$HOME/quad-junctions/extern
```

`QJ_EXTERN` defaults to `../../../quad-junctions/extern`; override it if the sibling
checkout is elsewhere. The int4 requirement is load-bearing: default-integer is int4, so
linking against the current int8 FMM3D segfaults in `pts_tree3d`.
