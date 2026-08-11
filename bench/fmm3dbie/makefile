# Laplace S_init benchmark on the paper's stellarator (fmm3dbie 2021, int4).
# Links the int4 FMM3D-2021 + fmm3dbie-2021 static libs (default integer = int4);
# the current int8 FMM3D segfaults in pts_tree3d. Matches the 2021 .make flags.
#
#   module load gcc/13.3.0 intel-oneapi-mkl/2024.2.2   # gfortran + MKLROOT
#   make                                               # build the libs first (see README)
#
# Run single-threaded (paper-comparable Table 1c):
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <core> ./aquad_lap_2021

FC = gfortran
FFLAGS = -fPIC -O3 -funroll-loops -march=native -fopenmp -std=legacy

# int4 2021 static libs, pinned as git submodules under extern/ (build them per README).
# Override the dirs if you built the libs elsewhere.
FMM3D_DIR    ?= extern/FMM3D
FMM3DBIE_DIR ?= extern/fmm3dbie
FMMBIE_LIB = $(FMM3DBIE_DIR)/lib-static/libfmm3dbie.a
FMM3D_LIB  = $(FMM3D_DIR)/lib-static/libfmm3d.a

MKL  = -L$(MKLROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
LIBS = $(FMMBIE_LIB) $(FMM3D_LIB) $(MKL) -lgomp -lgfortran

aquad_lap_2021: aquad_lap_stell_perf_test.f
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f aquad_lap_2021 *.o *.mod
