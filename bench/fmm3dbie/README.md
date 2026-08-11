# fmm3dbie Laplace S_init benchmark (stellarator)

Times fmm3dbie's single-layer **Laplace** near-quadrature setup on the stellarator of
Greengard, O'Neil, Rachh et al. 2021, reproducing the S_init (Table 1c) metric.

- **Driver:** `aquad_lap_stell_perf_test.f` (fmm3dbie 2021, int4).
- **Geometry:** the paper's stellarator, generated internally (`setup_geom`,
  igeomtype=2, iasp=3 / iref=2) → **2400 patches, norder=7 (p=8), 86400 nodes**. No
  input files — the surface is built in-process.
- **Timed:** `getnearquad_lap_comb_dir`, dpars=(1,0) (SLP Laplace), near list fixed
  (rfac=2.75, rfac0=1.25). Reports `S_slp = npts / t_slp` for eps in
  1e-3, 1e-5, 1e-7, 1e-9, 1e-11.

## Run

The prebuilt `aquad_lap_2021` needs no input files. For single-core, paper-comparable
numbers:

```
module load gcc/13.3.0 intel-oneapi-mkl/2024.2.2          # gfortran + MKL runtime
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <core> ./aquad_lap_2021
```

It prints `npatches`/`npts` and a table of `eps  npts  t_slp(s)  S_slp(pts/s)`.

## Rebuild

The driver links two **int4** static libs, pinned as git submodules under `extern/` at
their 2021 commits. The int4 build is load-bearing: default-integer is int4, matching the
driver; the current int8 upstream segfaults in `pts_tree3d`.

| submodule | repo | commit |
|---|---|---|
| `extern/FMM3D`    | github.com/flatironinstitute/FMM3D | `36a438b` (2021-06-28) |
| `extern/fmm3dbie` | github.com/fastalgorithms/fmm3dbie | `59a12b6` (2021-06-25) |

```
# from the repo root — fetch the pinned 2021 sources
git submodule update --init bench/fmm3dbie/extern/FMM3D bench/fmm3dbie/extern/fmm3dbie
cd bench/fmm3dbie

module load gcc/13.3.0 intel-oneapi-mkl/2024.2.2
make -C extern/FMM3D lib                                      # -> extern/FMM3D/lib-static/libfmm3d.a
make -C extern/fmm3dbie lib PREFIX_FMM=$PWD/extern/FMM3D/lib-static
make                                                         # -> aquad_lap_2021
```

Override `FMM3D_DIR` / `FMM3DBIE_DIR` (default `extern/FMM3D`, `extern/fmm3dbie`) if you
build the libs elsewhere.
