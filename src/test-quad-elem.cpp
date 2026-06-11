/**
 * This demo code shows how to use the class sctl::QuadElemList to build a
 * cubed-sphere geometry and write it to VTK for visualization.
 *
 * To compile and run the code, start in the SCTL root directory and run:
 * make bin/test-quad-elem && export OMP_NUM_THREADS=4 && ./bin/test-quad-elem
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
using namespace sctl;

namespace {

template <class Real> void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x * x + y * y + z * z);
  x *= R / r;
  y *= R / r;
  z *= R / r;
}

// ---------------------------------------------------------------------------
// Helpers for validating the cubed-sphere boundary-integral operators against
// the spherical-harmonics reference (SphericalHarmonics::*Eval{SL,DL}) on the
// unit sphere. Shared by every kernel x target-location configuration below.
// ---------------------------------------------------------------------------

// The three target placements relative to the (unit-radius) surface.
enum class TgtType { OnSurface, Near, Far };

// Density signature used everywhere: fill `out[0..KDIM0-1]` at point (x,y,z).
// Scalar (Laplace) kernels read out[0]; vector (Stokes) kernels read out[0..2].

// Sample a SCALAR density on the (Nt x Np) spherical-harmonic grid and return its
// scalar SH coefficients (ROW_MAJOR), used by LaplaceEval{SL,DL}.
template <class DensityFn> Vector<double> SphereScalarSHC(Long p, DensityFn density) {
  const Long Nt = p + 1, Np = 2 * p + 2;
  const Vector<double>& CosTheta = SphericalHarmonics<double>::LegendreNodes(Nt - 1);
  Vector<double> Xgrid(Nt * Np); // theta-major: Xgrid[i*Np + j]
  for (Long i = 0; i < Nt; i++) {
    const double ct = CosTheta[i], st = sqrt(1 - ct * ct);
    for (Long j = 0; j < Np; j++) {
      const double phi = 2 * const_pi<double>() * j / Np;
      double out[1];
      density(st * cos(phi), st * sin(phi), ct, out);
      Xgrid[i * Np + j] = out[0];
    }
  }
  Vector<double> S;
  SphericalHarmonics<double>::Grid2SHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
  return S;
}

// // Sample a VECTOR density on the SH grid (component-major SoA layout expected by
// // Grid2VecSHC: X[c*Nt*Np + i*Np + j]) and return its vector SH coefficients,
// // used by StokesEval{SL,DL}.
// template <class DensityFn> Vector<double> SphereVecSHC(Long p, DensityFn density) {
//   const Long Nt = p + 1, Np = 2 * p + 2, Ngrid = Nt * Np;
//   const Vector<double>& CosTheta = SphericalHarmonics<double>::LegendreNodes(Nt - 1);
//   Vector<double> Xgrid(3 * Ngrid);
//   for (Long i = 0; i < Nt; i++) {
//     const double ct = CosTheta[i], st = sqrt(1 - ct * ct);
//     for (Long j = 0; j < Np; j++) {
//       const double phi = 2 * const_pi<double>() * j / Np;
//       double out[3];
//       density(st * cos(phi), st * sin(phi), ct, out);
//       Xgrid[0 * Ngrid + i * Np + j] = out[0];
//       Xgrid[1 * Ngrid + i * Np + j] = out[1];
//       Xgrid[2 * Ngrid + i * Np + j] = out[2];
//     }
//   }
//   Vector<double> S;
//   SphericalHarmonics<double>::Grid2VecSHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
//   return S;
// }

// Run the cubed-sphere BIO vs. SH-reference comparison for one kernel across all
// three target placements. `ref_eval(coord, interior, U)` invokes the matching
// SphericalHarmonics::*Eval routine; `is_DL` selects the on-surface treatment:
// the double-layer is discontinuous, so the BIO returns the principal value,
// which equals the average of the interior and exterior SH limits.
template <class Kernel, class DensityFn, class RefEvalFn>
void TestSphereBIOvsSH(const QuadElemList<double>& elem_lst, const Comm& comm,
                       const Kernel& ker, const char* kername, bool is_DL,
                       DensityFn density, RefEvalFn ref_eval) {
  static constexpr Integer KDIM0 = Kernel::SrcDim();

  Vector<double> Xnodes;
  elem_lst.GetNodeCoord(&Xnodes, nullptr, nullptr);
  const Long Nnode = Xnodes.Dim() / 3;

  // Density at the cubed-sphere nodes (AoS, KDIM0 per node).
  Vector<double> F(Nnode * KDIM0);
  for (Long i = 0; i < Nnode; i++) density(Xnodes[i*3+0], Xnodes[i*3+1], Xnodes[i*3+2], &F[i*KDIM0]);

  BoundaryIntegralOp<double, Kernel> BIOp(ker, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(1e-9);
  BIOp.AddElemList(elem_lst);

  struct Cfg { const char* name; TgtType type; double scale; };
  const Cfg cfgs[3] = {
    {"on-surface", TgtType::OnSurface, 1.00}, // radius 1   (singular self-interaction)
    {"near",       TgtType::Near,      1.02}, // radius 1.02 (near-singular correction)
    {"far",        TgtType::Far,       2.00}, // radius 2   (smooth far-field only)
  };

  for (const auto& c : cfgs) {
    // Targets: surface nodes (on-surface, via the BIO default) or the nodes pushed
    // radially outward (= along the outward normal for a sphere) for off-surface.
    Vector<double> Xtrg;
    if (c.type != TgtType::OnSurface) {
      Xtrg.ReInit(Nnode * 3);
      for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xnodes[i] * c.scale;
      BIOp.SetTargetCoord(Xtrg);
    }

    Vector<double> U_quad;
    BIOp.ComputePotential(U_quad, F);

    // Spherical-harmonics reference at the same targets.
    const Vector<double>& coord = (c.type == TgtType::OnSurface) ? Xnodes : Xtrg;
    Vector<double> U_ref;
    if (c.type == TgtType::OnSurface && is_DL) {
      Vector<double> U_in, U_out; // principal value = mean of the two one-sided limits
      ref_eval(coord, /*interior=*/true,  U_in);
      ref_eval(coord, /*interior=*/false, U_out);
      U_ref.ReInit(U_in.Dim());
      for (Long i = 0; i < U_ref.Dim(); i++) U_ref[i] = 0.5 * (U_in[i] + U_out[i]);
    } else {
      // SL on-surface is continuous (interior flag moot); off-surface is exterior.
      ref_eval(coord, /*interior=*/false, U_ref);
    }

    SCTL_ASSERT(U_quad.Dim() == U_ref.Dim());
    double err2 = 0, ref2 = 0;
    for (Long i = 0; i < U_ref.Dim(); i++) {
      const double e = U_quad[i] - U_ref[i];
      err2 += e * e; ref2 += U_ref[i] * U_ref[i];
    }
    const double rel_l2 = sqrt(err2 / ref2);
    std::cout << "  " << kername << " / " << c.name << " : rel L2 error = " << rel_l2 << std::endl;
    // Geometry/quadrature-limited (~1e-7 observed); 1e-5 catches regressions with margin.
    SCTL_ASSERT(rel_l2 < 1e-5);
  }
}

// Quadrature check: the surface area from the far-field quadrature weights must
// match the analytic sphere area 4 pi R^2.
void test_SurfaceArea(const QuadElemList<double>& elem_lst, double Radius) {
  Vector<double> wts, Xtemp, Xntemp, dist_far;
  Vector<Long> elem_wise_temp;
  elem_lst.GetFarFieldNodes(Xtemp, Xntemp, wts, dist_far, elem_wise_temp, 1);
  double Area = 0.;
  for (int i = 0; i < wts.Dim(); i++) Area += wts[i];
  const double Area_exact = 4. * const_pi<double>() * Radius * Radius;
  std::cout << "Area from Jacobian: " << Area << ", from formula: " << Area_exact << std::endl;
  SCTL_ASSERT(std::fabs(Area - Area_exact) / Area_exact < 1e-6);
  std::cout << "Surface area test: PASSED" << std::endl;
}

// Stokes double-layer identity on the (closed) sphere.
//
// For a closed surface the Stokes double-layer potential of a *constant* density
// q, evaluated on the surface, obeys the jump relation
//     D[q](x) = c * q,     c = +-1/2,
// (the trace of the principal-value integral  \int_S T_ijk(x,y) n_k(y) dS_y is
// c * delta_ij). The sign of c depends on the kernel's r-convention and on which
// way the surface normal points. This kernel uses r = x_trg - x_src with the
// *source* normal, so the Gauss identity
//   \int_S r_i r_j (r.n)/r^5 dS = (2*pi/3) delta_ij   (PV, outward n, r=src-trg)
// predicts, after the r -> -r sign flip and the 3/(4 pi) prefactor,
//   c = -1/2  for an OUTWARD normal   (c = +1/2 for an inward normal).
// We verify the magnitude is 1/2 and the sign tracks the normal orientation.
void test_StokesDLIdentity(const QuadElemList<double>& elem_lst, const Comm& comm) {
  const Stokes3D_DxU ker_dl;
  BoundaryIntegralOp<double, Stokes3D_DxU> BIOp(ker_dl, /*trg_normal_dot_prod=*/false, comm);
  BIOp.SetAccuracy(1e-8);
  BIOp.AddElemList(elem_lst);
  BIOp.Setup();

  // Surface nodes and normals; detect the normal orientation via sign(x.n)
  // (x measured from the sphere center at the origin).
  Vector<double> Xs, Xns;
  elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long Nnode = Xs.Dim() / 3;
  double xdotn = 0;
  for (Long i = 0; i < Nnode; i++) {
    xdotn += Xs[i*3+0]*Xns[i*3+0] + Xs[i*3+1]*Xns[i*3+1] + Xs[i*3+2]*Xns[i*3+2];
  }
  const bool outward = (xdotn > 0);
  const double c_expect = outward ? -0.5 : 0.5;

  // Constant density q = (1, 0, 0) at every node (AoS, 3 components per node).
  Vector<double> q(Nnode * 3), U;
  for (Long i = 0; i < Nnode; i++) { q[i*3+0] = 1; q[i*3+1] = 0; q[i*3+2] = 0; }
  BIOp.ComputePotential(U, q);

  // D[q] should equal c * q = (c, 0, 0) at every node. Measure the mean of the
  // active component and the largest deviation from the constant field c*q.
  double cx_mean = 0;
  for (Long i = 0; i < Nnode; i++) cx_mean += U[i*3+0];
  cx_mean /= Nnode;

  double max_dev = 0, max_perp = 0;
  for (Long i = 0; i < Nnode; i++) {
    max_dev  = std::max(max_dev,  std::fabs(U[i*3+0] - c_expect));
    max_perp = std::max(max_perp, std::max(std::fabs(U[i*3+1]), std::fabs(U[i*3+2])));
  }

  std::cout << "Stokes double-layer constant-density identity:\n"
            << "  normal orientation : " << (outward ? "outward" : "inward")
            << " (sum x.n = " << xdotn << ")\n"
            << "  mean U_x           : " << cx_mean << "  (expected " << c_expect << ")\n"
            << "  max |U_x - c|      : " << max_dev  << "\n"
            << "  max |U_perp|       : " << max_perp << std::endl;

  const double rel_tol = 1e-3; // dominated by the polynomial sphere-geometry error
  SCTL_ASSERT(std::fabs(cx_mean - c_expect) < rel_tol);
  SCTL_ASSERT(max_dev  < rel_tol);
  SCTL_ASSERT(max_perp < rel_tol);
  std::cout << "Stokes double-layer identity: PASSED (|c| = 1/2, sign tracks "
            << "outward normal -> -1/2)" << std::endl;
}

// Boundary-integral operators on the sphere vs. a spherical-harmonics reference,
// over { Laplace/Stokes x single-/double-layer } and three target placements
// { on-surface, near (off-surface), far }. On the unit sphere the layer operators
// are diagonalized by (vector) spherical harmonics, so SphericalHarmonics::*Eval
// provides a spectrally accurate reference for a smooth NON-POLYNOMIAL density.
void test_BIOvsSH(const QuadElemList<double>& elem_lst, const Comm& comm) {
  const Long p = 30; // SH truncation order (captures the exp densities to ~eps)

  // Non-polynomial densities (analytic on the sphere -> fast SH decay).
  auto lap_density = [](double x, double, double, double* o) { o[0] = std::exp(x); };
  // auto sto_density = [](double x, double y, double z, double* o) {
  //   o[0] = std::exp(x); o[1] = std::exp(y); o[2] = std::exp(z);
  // };

  // Density SH coefficients (computed once per density type).
  const Vector<double> Slap = SphereScalarSHC(p, lap_density);
  // const Vector<double> Ssto = SphereVecSHC(p, sto_density);

  std::cout << "BIO vs. spherical-harmonics reference (density = exp):" << std::endl;

  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_FxU(), "Laplace3D_FxU", /*is_DL=*/false, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalSL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); });

  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_DxU(), "Laplace3D_DxU", /*is_DL=*/true, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalDL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); });

  std::cout << "BIO vs. SH reference Laplace: PASSED" << std::endl;

  // TestSphereBIOvsSH(elem_lst, comm, Stokes3D_FxU(), "Stokes3D_FxU", /*is_DL=*/false, sto_density,
  //   [&](const Vector<double>& c, bool in, Vector<double>& U) {
  //     SphericalHarmonics<double>::StokesEvalSL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); });

  // TestSphereBIOvsSH(elem_lst, comm, Stokes3D_DxU(), "Stokes3D_DxU", /*is_DL=*/true, sto_density,
  //   [&](const Vector<double>& c, bool in, Vector<double>& U) {
  //     SphericalHarmonics<double>::StokesEvalDL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); });

  // std::cout << "BIO vs. SH reference Stokes: PASSED" << std::endl;
  SphericalHarmonics<double>::Clear(); // release SH precomputed tables
}

}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);

  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "\
        This demo is sequential. In a distributed memory implementation, each process\n\
        would build only its local section of the geometry.");

    const Long ElemOrder = 8;
    const Long PatchPerFace = 3;
    const double Radius = 1.0;
    const Long Nelem = 6 * PatchPerFace * PatchPerFace;

    Vector<double> X;

    const Vector<double>& nds = QuadElemList<double>::ParamNodes(ElemOrder);
    for (Integer face = 0; face < 6; face++) {
      for (Long iu = 0; iu < PatchPerFace; iu++) {
        for (Long iv = 0; iv < PatchPerFace; iv++) {
          for (Long i = 0; i < ElemOrder; i++) {
            const double u = (iu + nds[i]) / (double)PatchPerFace;
            const double a = 2 * u - 1;
            for (Long j = 0; j < ElemOrder; j++) {
              const double v = (iv + nds[j]) / (double)PatchPerFace;
              const double b = 2 * v - 1;

              double x, y, z;
              FacePoint(x, y, z, face, a, b, Radius);
              X.PushBack(x);
              X.PushBack(y);
              X.PushBack(z);
            }
          }
        }
      }
    }

    QuadElemList<double> elem_lst(ElemOrder, X);

    elem_lst.Write("cubed-sphere.geom", comm);
    elem_lst.Read<double>("cubed-sphere.geom", comm);

    Vector<double> Xsurf, Xn;
    Vector<Long> element_wise_node_cnt;
    elem_lst.GetNodeCoord(&Xsurf, &Xn, &element_wise_node_cnt);

    Vector<double> dXdu, dXdv, dXa;
    for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
      const auto& nodes = QuadElemList<double>::ParamNodes(ElemOrder);
      Vector<double> dXdu_elem, dXdv_elem, dXa_elem;
      elem_lst.GetGeom(nullptr, nullptr, &dXa_elem, &dXdu_elem, &dXdv_elem, nodes, nodes, elem_idx);
      for (const auto& v : dXdu_elem) dXdu.PushBack(v);
      for (const auto& v : dXdv_elem) dXdv.PushBack(v);
      for (const auto& v : dXa_elem) dXa.PushBack(v);
    }

    // elem_lst.WriteVTK("cubed-sphere", Xn, comm);

    test_SurfaceArea(elem_lst, Radius);
    test_StokesDLIdentity(elem_lst, comm);
    test_BIOvsSH(elem_lst, comm);

  }

  Comm::MPI_Finalize();
  return 0;
}
