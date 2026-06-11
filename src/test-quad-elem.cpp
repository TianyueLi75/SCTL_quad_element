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
#include <iomanip>
#include <vector>
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

// Build a cubed-sphere QuadElemList of radius `Radius`, with `PatchPerFace` x
// `PatchPerFace` quad patches on each of the 6 cube faces and `ElemOrder` nodes
// per parameter direction within each patch.
template <class Real>
QuadElemList<Real> BuildCubedSphere(Long ElemOrder, Long PatchPerFace, Real Radius) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++) {
    for (Long iu = 0; iu < PatchPerFace; iu++) {
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        for (Long i = 0; i < ElemOrder; i++) {
          const Real u = (iu + nds[i]) / (Real)PatchPerFace;
          const Real a = 2 * u - 1;
          for (Long j = 0; j < ElemOrder; j++) {
            const Real v = (iv + nds[j]) / (Real)PatchPerFace;
            const Real b = 2 * v - 1;

            Real x, y, z;
            FacePoint(x, y, z, face, a, b, Radius);
            X.PushBack(x);
            X.PushBack(y);
            X.PushBack(z);
          }
        }
      }
    }
  }
  return QuadElemList<Real>(ElemOrder, X);
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

// Sample a VECTOR density on the SH grid (component-major SoA layout expected by
// Grid2VecSHC: X[c*Nt*Np + i*Np + j]) and return its vector SH coefficients,
// used by StokesEval{SL,DL}.
template <class DensityFn> Vector<double> SphereVecSHC(Long p, DensityFn density) {
  const Long Nt = p + 1, Np = 2 * p + 2, Ngrid = Nt * Np;
  const Vector<double>& CosTheta = SphericalHarmonics<double>::LegendreNodes(Nt - 1);
  Vector<double> Xgrid(3 * Ngrid);
  for (Long i = 0; i < Nt; i++) {
    const double ct = CosTheta[i], st = sqrt(1 - ct * ct);
    for (Long j = 0; j < Np; j++) {
      const double phi = 2 * const_pi<double>() * j / Np;
      double out[3];
      density(st * cos(phi), st * sin(phi), ct, out);
      Xgrid[0 * Ngrid + i * Np + j] = out[0];
      Xgrid[1 * Ngrid + i * Np + j] = out[1];
      Xgrid[2 * Ngrid + i * Np + j] = out[2];
    }
  }
  Vector<double> S;
  SphericalHarmonics<double>::Grid2VecSHC(Xgrid, Nt, Np, p, S, SHCArrange::ROW_MAJOR);
  return S;
}

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
  auto sto_density = [](double x, double y, double z, double* o) {
    o[0] = std::exp(x); o[1] = std::exp(y); o[2] = std::exp(z);
  };

  // Density SH coefficients (computed once per density type).
  const Vector<double> Slap = SphereScalarSHC(p, lap_density);
  const Vector<double> Ssto = SphereVecSHC(p, sto_density);

  std::cout << "BIO vs. spherical-harmonics reference (density = exp):" << std::endl;

  // Visualize the adaptive near/self interaction refinement on one element. The
  // refinement is kernel-independent, so this is done once (not per kernel). The
  // tolerance matches BIOp.SetAccuracy(1e-9) in TestSphereBIOvsSH so the picture
  // reflects the rules the solve actually uses.
  {
    const Long evis = 0;                       // element to visualize
    const Integer ord = elem_lst.Order();
    // Near: a surface point pushed off the surface along its outward normal.
    Vector<double> us{0.5}, vs{0.5}, Xc, Xn;
    elem_lst.GetGeom(&Xc, &Xn, nullptr, nullptr, nullptr, us, vs, evis);
    Vector<double> Xtrg(3);
    for (int k = 0; k < 3; k++) Xtrg[k] = Xc[k] + 0.02 * Xn[k];
    elem_lst.WriteNearInteracVTK("near-interac-elem0", evis, Xtrg, 1e-9, comm);
    // Self: an interior node parameter (u0,v0).
    const auto& nds = QuadElemList<double>::ParamNodes(ord);
    const double u0 = nds[ord/2], v0 = nds[ord/2];
    elem_lst.WriteSelfInteracVTK("self-interac-elem0", evis, u0, v0, 1e-9, comm);
    std::cout << "  wrote near-interac-elem0-* and self-interac-elem0-* VTK files" << std::endl;
  }

  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_FxU(), "Laplace3D_FxU", /*is_DL=*/false, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalSL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); });

  TestSphereBIOvsSH(elem_lst, comm, Laplace3D_DxU(), "Laplace3D_DxU", /*is_DL=*/true, lap_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::LaplaceEvalDL(Slap, SHCArrange::ROW_MAJOR, p, c, in, U); });

  std::cout << "BIO vs. SH reference Laplace: PASSED" << std::endl;

  TestSphereBIOvsSH(elem_lst, comm, Stokes3D_FxU(), "Stokes3D_FxU", /*is_DL=*/false, sto_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::StokesEvalSL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); });

  TestSphereBIOvsSH(elem_lst, comm, Stokes3D_DxU(), "Stokes3D_DxU", /*is_DL=*/true, sto_density,
    [&](const Vector<double>& c, bool in, Vector<double>& U) {
      SphericalHarmonics<double>::StokesEvalDL(Ssto, SHCArrange::ROW_MAJOR, p, c, in, U); });

  std::cout << "BIO vs. SH reference Stokes: PASSED" << std::endl;
  SphericalHarmonics<double>::Clear(); // release SH precomputed tables
}

// ---------------------------------------------------------------------------
// Manufactured-solution test for the interior OR exterior Dirichlet problem,
// solved with a combined-field (single- + double-layer) boundary integral
// equation and GMRES. Mirrors the structure of biop/Lap3d.py and biop/Stk3d.py,
// but replaces the spherical-harmonics spectral operators with the quad-element
// ComputePotential.
//
// Point sources (charges for Laplace, Stokeslets for Stokes) are placed on the
// side of the surface OPPOSITE the solution domain -- inside the sphere for the
// exterior problem, outside for the interior problem -- so their field u_e is a
// smooth, exact solution of the PDE on the domain of interest. We sample u_e on
// the surface as Dirichlet data and solve
//      ( c*I + SL_scal*S + DL_scal*D ) sigma = u_e|_surface,
// where S, D are the principal-value layer operators returned by ComputePotential
// and c is the double-layer jump for the solution side: for an outward normal
// c = +1/2*DL_scal (exterior, sgn = +1) or c = -1/2*DL_scal (interior, sgn = -1).
// By uniqueness of the Dirichlet problem the recovered combined potential equals
// u_e throughout the domain, which we verify at a target sphere of radius
// `eval_radius` (> 1 for exterior, < 1 for interior).
//
// SIGN REQUIREMENT on (SL_scal, DL_scal): the on-surface CFIE operator is only
// uniquely solvable for the right relative sign. If sigma is in its null space the
// interior trace vanishes (u == 0 inside), so the *exterior* field carries Cauchy
// data (DL_scal*sigma, -SL_scal*sigma) and a Green's identity gives
//   \int_ext |grad u|^2 = SL_scal * DL_scal * \int_surface sigma^2     (Laplace),
// the analogous viscous-dissipation identity for Stokes. Hence:
//   exterior problem -> SL_scal, DL_scal SAME sign (e.g. +1, +1);
//   interior problem -> SL_scal, DL_scal OPPOSITE sign (e.g. -1, +1).
// With the same sign the interior operator has a nontrivial null space (a nonzero
// exterior field) independent of the source, so zeroing the net charge/force does
// NOT cure the interior breakdown -- the relative sign must be flipped.
//
// Returns the rel-L2 error at the target sphere. `quadr_tol` is the quadrature
// tolerance and also sets the GMRES tolerance.
template <class KerSL, class KerDL>
double TestManufactured(const QuadElemList<double>& elem_lst, const Comm& comm,
                        const KerSL& ker_sl, const KerDL& ker_dl, const char* name,
                        const Vector<double>& Xsrc, const Vector<double>& Fsrc,
                        bool interior, double eval_radius, const double quadr_tol = 1e-9,
                        double SL_scal = 1.0, const double DL_scal = 1.0) {
  static constexpr Integer KDIM = KerSL::SrcDim(); // 1 (Laplace) or 3 (Stokes)

  // Surface nodes & normals; outward-normal orientation sets the DL jump sign.
  Vector<double> Xs, Xns;
  elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long Nnode = Xs.Dim() / 3;
  double xdotn = 0;
  for (Long i = 0; i < Nnode; i++)
    xdotn += Xs[i*3+0]*Xns[i*3+0] + Xs[i*3+1]*Xns[i*3+1] + Xs[i*3+2]*Xns[i*3+2];
  // Exterior trace is +1/2 for an outward normal; the interior trace flips the sign.
  const double sgn = interior ? -1.0 : 1.0;
  const double jump = (xdotn > 0 ? 0.5 : -0.5) * DL_scal * sgn;

  // Manually check the formulation, if interior, change SL sign to avoid 0 eigenvalue. throw warning
  if (interior && SL_scal*DL_scal > 0.) {
    std::cout << "Warning: Interior problem has artificial null space when SL and DL same sign. Flipping SL sign. " << std::endl;
    SL_scal = -1.*SL_scal;
  }

  // Dirichlet data: trace of the point-source field at the surface nodes.
  // (The single-layer kernel is the charge / Stokeslet Green's function; its
  // `Eval` ignores the source-normal argument, so Xsrc is passed as a dummy.)
  Vector<double> bc;
  ker_sl.Eval(bc, Xs, Xsrc, Xsrc, Fsrc);

  // Combined-field operator pieces (on-surface principal values).
  BoundaryIntegralOp<double, KerSL> SLOp(ker_sl, /*trg_normal_dot_prod=*/false, comm);
  BoundaryIntegralOp<double, KerDL> DLOp(ker_dl, /*trg_normal_dot_prod=*/false, comm);
  SLOp.SetAccuracy(quadr_tol); DLOp.SetAccuracy(quadr_tol);
  SLOp.AddElemList(elem_lst); DLOp.AddElemList(elem_lst);

  const auto ApplyK = [&](Vector<double>* U, const Vector<double>& sigma) {
    Vector<double> Us, Ud;
    SLOp.ComputePotential(Us, sigma);
    DLOp.ComputePotential(Ud, sigma);
    if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
    (*U) = SL_scal*Us + DL_scal*Ud + jump*sigma;
  };

  GMRES<double> solver(comm, false);
  Vector<double> sigma;
  Long iter = 0;
  const double gmres_tol = quadr_tol * 10.;
  const Long gmres_max_iter = 100;
  solver(&sigma, ApplyK, bc, gmres_tol, gmres_max_iter, false, &iter);

  // Evaluate the recovered combined potential at the target sphere and compare to
  // the exact field u_e. The surface nodes sit at radius 1, so scaling them sets
  // the target radius; placements just off the surface (|eval_radius - 1| small)
  // exercise the near-singular correction, larger offsets the smooth far field.
  Vector<double> Xtrg(Nnode * 3);
  for (Long i = 0; i < Nnode * 3; i++) Xtrg[i] = Xs[i] * eval_radius;
  SLOp.SetTargetCoord(Xtrg); DLOp.SetTargetCoord(Xtrg);
  Vector<double> Us, Ud;
  SLOp.ComputePotential(Us, sigma);
  DLOp.ComputePotential(Ud, sigma);
  Vector<double> U(Nnode * KDIM);
  for (Long i = 0; i < U.Dim(); i++) U[i] = SL_scal*Us[i] + DL_scal*Ud[i];

  // Reference: the same point-source field evaluated directly at the targets.
  Vector<double> Uref;
  ker_sl.Eval(Uref, Xtrg, Xsrc, Xsrc, Fsrc);

  double err2 = 0, ref2 = 0;
  for (Long i = 0; i < U.Dim(); i++) { const double e = U[i] - Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; }
  const double rel_l2 = sqrt(err2 / ref2);
  // std::cout << "  " << name << " (R=" << eval_radius << ", GMRES iters = " << iter
  //           << ") : rel L2 error = " << rel_l2 << std::endl;
  return rel_l2;
}

// Laplace combined-field Dirichlet manufactured solution: point charges on the
// far side of the surface, recover their potential on the solution side. Tested
// for both the exterior problem (charges inside the sphere) and the interior
// problem (charges outside the sphere).
void test_LaplaceManufactured(const QuadElemList<double>& elem_lst, const Comm& comm) {
  // const Vector<double> Fsrc{1.0, -0.7};
  const Vector<double> Fsrc{1.0, -1.0};

  // Exterior: charges strictly inside the unit sphere -> exact decaying exterior
  // field; verify at a near (near-singular) and a far (smooth) target radius > 1.
  const Vector<double> Xsrc_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10};
  std::cout << "Manufactured solution (Laplace, exterior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radius=*/1.001) < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radius=*/2.000) < 1e-5);

  // Interior: charges strictly outside the unit sphere -> exact smooth interior
  // field; verify at a near and a far target radius < 1. The interior CFIE is only
  // uniquely solvable when SL_scal and DL_scal have OPPOSITE signs (see note above
  // TestManufactured), so the single-layer enters with SL_scal = -1.
  const Vector<double> Xsrc_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60};
  std::cout << "Manufactured solution (Laplace, interior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radius=*/0.999,
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0) < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                "Laplace SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radius=*/0.500,
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0) < 1e-5);

  std::cout << "Laplace manufactured-solution test: PASSED" << std::endl;
}

// Stokes combined-field Dirichlet manufactured solution: Stokeslets on the far
// side of the surface, recover their velocity field on the solution side. Tested
// for both the exterior problem (Stokeslets inside) and the interior problem
// (Stokeslets outside); the net force is nonzero so the single-layer part of the
// combined-field representation is exercised.
void test_StokesManufactured(const QuadElemList<double>& elem_lst, const Comm& comm) {
  // const Vector<double> Fsrc{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};
  const Vector<double> Fsrc{1.0, 0.5, -0.3,  -1.0, -0.5, 0.3};

  // Exterior: Stokeslets strictly inside the unit sphere.
  const Vector<double> Xsrc_ext{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10};
  std::cout << "Manufactured solution (Stokes, exterior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radius=*/1.001) < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_ext, Fsrc, /*interior=*/false, /*eval_radius=*/2.000) < 1e-5);

  // Interior: Stokeslets strictly outside the unit sphere. As for Laplace, the
  // interior CFIE needs SL_scal and DL_scal of OPPOSITE sign, so SL_scal = -1.
  const Vector<double> Xsrc_int{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60};
  std::cout << "Manufactured solution (Stokes, interior Dirichlet):" << std::endl;
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radius=*/0.999,
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0) < 1e-4);
  SCTL_ASSERT(TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                "Stokes SL+DL", Xsrc_int, Fsrc, /*interior=*/true, /*eval_radius=*/0.500,
                /*quadr_tol=*/1e-9, /*SL_scal=*/-1.0, /*DL_scal=*/1.0) < 1e-5);

  std::cout << "Stokes manufactured-solution test: PASSED" << std::endl;
}

// Convergence study: for a fixed ElemOrder, refine the cubed-sphere by increasing
// PatchPerFace and report the manufactured-solution exterior rel-L2 error at each
// resolution (near and far target placements). With ElemOrder fixed this measures
// h-refinement convergence of the geometry/quadrature.
void test_ManufacturedConvergence(const Comm& comm,
                                  const std::vector<Long>& PatchPerFaceList = {1, 2, 3, 4, 5},
                                  Long ElemOrder = 10) {
  const double Radius = 1.0;
  const double base_tol = 1e-8;

  // Laplace: point charges strictly inside the unit sphere.
  // const Vector<double> Xsrc_lap{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10}; // exterior problem, interior src.
  const Vector<double> Xsrc_lap{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60}; // interior problem, exterior src
  const Vector<double> Fsrc_lap{1.0, -0.7};
  // Stokes: Stokeslets strictly inside the unit sphere.
  // const Vector<double> Xsrc_sto{0.10, 0.20, 0.15,  -0.20, 0.10, -0.10}; // exterior problem, interior src.
  const Vector<double> Xsrc_sto{1.50, 0.40, 0.30,  -1.20, 0.80, -0.60}; // interior problem, exterior src
  const Vector<double> Fsrc_sto{1.0, 0.5, -0.3,  -0.4, 0.2, 0.1};

  std::cout << "\nManufactured-solution convergence study (ElemOrder = " << ElemOrder << "):\n";
  std::cout << std::scientific;
  std::cout << "  kernel    PatchPerFace  Nelem   rel-L2 (near R=0.999)   rel-L2 (far R=0.5)\n";
  for (const Long PatchPerFace : PatchPerFaceList) {
    const QuadElemList<double> elem_lst = BuildCubedSphere<double>(ElemOrder, PatchPerFace, Radius);
    const Long Nelem = 6 * PatchPerFace * PatchPerFace;

    double quadr_tol = base_tol;
    if (PatchPerFace > 5) {
      quadr_tol *= 0.0001;
    } else if (PatchPerFace > 3) {
      quadr_tol *= 0.01;
    }

    const double el_near = TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                             "Laplace SL+DL", Xsrc_lap, Fsrc_lap, /*interior=*/true, 0.999, quadr_tol, 0.0, 1.0);
    const double el_far  = TestManufactured(elem_lst, comm, Laplace3D_FxU(), Laplace3D_DxU(),
                             "Laplace SL+DL", Xsrc_lap, Fsrc_lap, /*interior=*/true, 0.5, quadr_tol, 0.0, 1.0);
    std::cout << "  Laplace   " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
              << "   " << el_near << "        " << el_far << "\n";

    const double es_near = TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                             "Stokes SL+DL", Xsrc_sto, Fsrc_sto, /*interior=*/true, 0.999, quadr_tol, 0.0, 1.0);
    const double es_far  = TestManufactured(elem_lst, comm, Stokes3D_FxU(), Stokes3D_DxU(),
                             "Stokes SL+DL", Xsrc_sto, Fsrc_sto, /*interior=*/true, 0.5, quadr_tol, 0.0, 1.0);
    std::cout << "  Stokes    " << std::setw(12) << PatchPerFace << "  " << std::setw(5) << Nelem
              << "   " << es_near << "        " << es_far << "\n";
  }
  std::cout << "Manufactured-solution convergence study: DONE" << std::endl;
}

}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);

  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(comm.Size() == 1, "\
        This demo is sequential. In a distributed memory implementation, each process\n\
        would build only its local section of the geometry.");

    const Long ElemOrder = 10;
    const Long PatchPerFace = 3;
    const double Radius = 1.0;
    // const Long Nelem = 6 * PatchPerFace * PatchPerFace;

    QuadElemList<double> elem_lst = BuildCubedSphere<double>(ElemOrder, PatchPerFace, Radius);

    // elem_lst.Write("cubed-sphere.geom", comm);
    // elem_lst.Read<double>("cubed-sphere.geom", comm);

    // Vector<double> Xsurf, Xn;
    // Vector<Long> element_wise_node_cnt;
    // elem_lst.GetNodeCoord(&Xsurf, &Xn, &element_wise_node_cnt);

    // Vector<double> dXdu, dXdv, dXa;
    // for (Long elem_idx = 0; elem_idx < Nelem; elem_idx++) {
    //   const auto& nodes = QuadElemList<double>::ParamNodes(ElemOrder);
    //   Vector<double> dXdu_elem, dXdv_elem, dXa_elem;
    //   elem_lst.GetGeom(nullptr, nullptr, &dXa_elem, &dXdu_elem, &dXdv_elem, nodes, nodes, elem_idx);
    //   for (const auto& v : dXdu_elem) dXdu.PushBack(v);
    //   for (const auto& v : dXdv_elem) dXdv.PushBack(v);
    //   for (const auto& v : dXa_elem) dXa.PushBack(v);
    // }

    // elem_lst.WriteVTK("cubed-sphere", Xn, comm);

    test_SurfaceArea(elem_lst, Radius);
    test_StokesDLIdentity(elem_lst, comm);
    test_BIOvsSH(elem_lst, comm);
    // test_LaplaceManufactured(elem_lst, comm);
    // test_StokesManufactured(elem_lst, comm);

    // test_ManufacturedConvergence(comm);

  }

  Comm::MPI_Finalize();
  return 0;
}
