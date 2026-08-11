# Laplace S_init benchmark on the paper's stellarator (fmm3dbie 2021, int4).
# Must link the int4 FMM3D-2021 lib (default integer = int4); mixing with the current
# int8 FMM3D segfaults in pts_tree3d. Matches the 2021 .make flags.
#
#   module load gcc/13.3.0        # provides gfortran + MKLROOT   (or: . <repo>/sctl_source)
#   make                          # needs the 2021 int4 static libs (see QJ_EXTERN below)
#
# Run single-threaded (paper-comparable Table 1c):
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <core> ./aquad_lap_2021

FC = gfortran
FFLAGS = -fPIC -O3 -funroll-loops -march=native -fopenmp -std=legacy

# The 2021 int4 static libs are NOT vendored here; point at the quad-junctions checkout
# (override on the command line: make QJ_EXTERN=/path/to/quad-junctions/extern).
QJ_EXTERN ?= ../../../quad-junctions/extern
FMMBIE_LIB = $(QJ_EXTERN)/fmm3dbie-2021/lib-static/libfmm3dbie.a
FMM3D_LIB  = $(QJ_EXTERN)/FMM3D-2021/lib-static/libfmm3d.a

MKL  = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LIBS = $(FMMBIE_LIB) $(FMM3D_LIB) $(MKL) -lgomp -lgfortran

aquad_lap_2021: aquad_lap_stell_perf_test.f
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f aquad_lap_2021 *.o *.mod
