#include <sctl.hpp>
#include "sctl/experimental/quad_element.cpp" // template definitions

using namespace sctl;


template <class Real> Vector<Real> get_testsurf(const Integer order, const Integer nelem_perside) {
    // First define surface
    const auto fsurf = [](const Real x, const Real y) {
        return x*y;
    };

    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, nelem_perside); // Get x-y grid on [0,1]x[0,1]
    // Get z value on x-y grid for surface.
    for (int i=0; i<coord0.Dim()/3; i++) {
        coord0[i*3 + 2] = fsurf(coord0[i*3+0], coord0[i*3+1]);
    }
    return coord0;
}

template <class Real> void test_ParamGrid() {
    // Test tensor grid generation directly on ParamGrid (not via get_testsurf,
    // whose z = x*y belongs to the test surface and not to ParamGrid).
    const Long order = 4;
    const Long nelem_perside = 2;
    const Long N_per_side = order * nelem_perside; // 8 nodes per side
    const Long N_total = N_per_side * N_per_side;  // 64 tensor-grid points

    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, nelem_perside);
    SCTL_ASSERT(coord0.Dim() == N_total * 3);

    // Explicit expected 1-D nodes: order-4 Gauss-Legendre nodes on [-1,1] are
    // +-0.339981043584856 and +-0.861136311594053. Mapped to [0,1] via (x+1)/2
    // and split into 2 equal panels [0,0.5] and [0.5,1], the 8 per-side
    // coordinates are:
    const Real x_param_exp[8] = {
        0.034715922101486804, // panel 0
        0.165004739103786020,
        0.334995260896213980,
        0.465284077898513196,
        0.534715922101486804, // panel 1
        0.665004739103786020,
        0.834995260896213980,
        0.965284077898513196
    };

    // The grid is the tensor product x_param_exp (x) x_param_exp, AoS order with
    // u (xind) the slow index and v (yind) the fast index, with z left at 0.
    const Real tol = 1e-12;
    for (Long xind = 0; xind < N_per_side; xind++) {
        for (Long yind = 0; yind < N_per_side; yind++) {
            const Long idx = (xind * N_per_side + yind) * 3;
            SCTL_ASSERT(fabs(coord0[idx + 0] - x_param_exp[xind]) < tol);
            SCTL_ASSERT(fabs(coord0[idx + 1] - x_param_exp[yind]) < tol);
            SCTL_ASSERT(fabs(coord0[idx + 2] - (Real)0) < tol);
        }
    }
}

template <class Real> void test_GetClosestNode_plane() {
    // Flat patch z = 0 over [0,1]^2 (x = u, y = v); ParamGrid leaves z = 0. 
    const Long COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Take a surface node point, then move it up in +z. ClosestNode should recover surface node
    Vector<Real> X, Xn;
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const int trg_idx = 13; // arbitrary point on surface
    const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>) X.begin() + trg_idx * COORD_DIM, false);
    const Vector<Real> Xntrg(COORD_DIM, (Iterator<Real>) Xn.begin() + trg_idx * COORD_DIM, false);
    const Real utrg = coord0[trg_idx*COORD_DIM + 0];
    const Real vtrg = coord0[trg_idx*COORD_DIM + 1];

    Vector<Real> Xtrg_shifted = Xtrg;
    Xtrg_shifted[2] = 0.1;

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestNode(ustar, vstar, &Xstar, &Nstar, 0, Xtrg_shifted);

    const Real tol = 1e-9;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - 0.1) < tol);
    SCTL_ASSERT(fabs(Xstar[0] - Xtrg[0]) < tol);
    SCTL_ASSERT(fabs(Xstar[1] - Xtrg[1]) < tol);
    SCTL_ASSERT(fabs(Xstar[2] - 0.0) < tol);
    SCTL_ASSERT(fabs(Nstar[0]) - Xntrg[0] < tol);
    SCTL_ASSERT(fabs(Nstar[1]) - Xntrg[1] < tol);
    SCTL_ASSERT(fabs(fabs(Nstar[2]) - 1.0) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>(0.1*0.1 + 0.0013*0.0013 + 0.0005*0.0005);

    const Real dist2 = qel.GetClosestNode(ustar, vstar, &Xstar, &Nstar, 0, Xtrg_shifted);

    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist2  - exp_dist) < tol);
    SCTL_ASSERT(fabs(Xstar[0] - Xtrg[0]) < tol);
    SCTL_ASSERT(fabs(Xstar[1] - Xtrg[1]) < tol);
    SCTL_ASSERT(fabs(Xstar[2] - 0.0) < tol);
}

template <class Real> void test_GetClosestNode_curved() {
    // Curved patch z = u*v over [0,1]^2 (the fsurf from get_testsurf); 
    const Integer COORD_DIM = 3;
    const Long order = 8;
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Take a surface node point x, then move it up in +d*n(x). ClosestNode should recover surface node
    Vector<Real> X, Xn;
    qel.GetNodeCoord(&X, &Xn, nullptr);
    const int trg_idx = 13; // arbitrary point on surface
    const Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>) X.begin() + trg_idx * COORD_DIM, false);
    const Vector<Real> Xntrg(COORD_DIM, (Iterator<Real>) Xn.begin() + trg_idx * COORD_DIM, false);
    const Real utrg = coord0[trg_idx*COORD_DIM + 0];
    const Real vtrg = coord0[trg_idx*COORD_DIM + 1];
    const Real d = 0.001;
    Vector<Real> Xtrg_shifted = Xtrg + d * Xntrg;

    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestNode(ustar, vstar, &Xstar, &Nstar, 0, Xtrg_shifted);

    const Real tol = 1e-8;
    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist  - d ) < tol);
    SCTL_ASSERT(fabs(Xstar[0] - Xtrg[0]) < tol);
    SCTL_ASSERT(fabs(Xstar[1] - Xtrg[1]) < tol);
    SCTL_ASSERT(fabs(Xstar[2] - Xtrg[2]) < tol);
    SCTL_ASSERT(fabs(Nstar[0] - Xntrg[0]) < tol);
    SCTL_ASSERT(fabs(Nstar[1] - Xntrg[1]) < tol);
    SCTL_ASSERT(fabs(Nstar[2] - Xntrg[2]) < tol);

    // Now shift the target away from x and y as well.
    Xtrg_shifted[0] -= 0.0013;
    Xtrg_shifted[1] += 0.0005;
    const Real exp_dist = sqrt<Real>((d*Xntrg[0]-0.0013)*(d*Xntrg[0]-0.0013) + (d*Xntrg[1]+0.0005)*(d*Xntrg[1]+0.0005) + (d*Xntrg[2])*(d*Xntrg[2]));

    const Real dist2 = qel.GetClosestNode(ustar, vstar, &Xstar, &Nstar, 0, Xtrg_shifted);

    SCTL_ASSERT(fabs(ustar - utrg) < tol);
    SCTL_ASSERT(fabs(vstar - vtrg) < tol);
    SCTL_ASSERT(fabs(dist2  - exp_dist) < tol);
    SCTL_ASSERT(fabs(Xstar[0] - Xtrg[0]) < tol);
    SCTL_ASSERT(fabs(Xstar[1] - Xtrg[1]) < tol);
    SCTL_ASSERT(fabs(Xstar[2] - Xtrg[2]) < tol);
}


// Reference near-singular evaluation: directly integrate the boundary-integral
// operator on element `elem_idx` against a single target `Xt`, using a globally
// uniform refinement of the parameter square into nsub x nsub panels, each with a
// plain order-`order` Gauss-Legendre rule (same GL order as the physical
// discretization). The density carried by the element is the order-`order`
// Lagrange interpolant of the nodal values `sigma` (AoS, {s0_0..s0_{K-1}, s1_0..}),
// the SAME density representation NearInterac uses. As nsub increases this sum
// converges to the exact value of that integral; NearInterac's adaptive quadtree
// computes the same integral to tolerance, so the two must agree.
//
// Returns the target potential (Kernel::TrgDim() reals).
template <class Real, class Kernel> Vector<Real> direct_upsampled_potential(
    const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& sigma,
    const Vector<Real>& Xt, const Kernel& ker, const Long nsub) {

    const Integer order = qel.Order();
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    const Long nq = (Long)order * order;
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    const Vector<Real>& wts = LegQuadRule<Real>::wts(order);

    Vector<Real> u(KDIM1);
    u.SetZero();

    Vector<Real> u_param(order), v_param(order);
    for (Long pi = 0; pi < nsub; pi++) {
        for (Long pj = 0; pj < nsub; pj++) {
            for (Integer a = 0; a < order; a++) u_param[a] = (nds[a] + pi) / (Real)nsub;
            for (Integer b = 0; b < order; b++) v_param[b] = (nds[b] + pj) / (Real)nsub;

            // Geometry on this panel's order x order GL grid.
            Vector<Real> X, Xn, Xa;
            qel.GetGeom(&X, &Xn, &Xa, nullptr, nullptr, u_param, v_param, elem_idx);

            // Lagrange weights from the patch nodes to the panel quad nodes:
            // Lu[i*order + a] = L_i(u_param[a]).
            Vector<Real> Lu(order * order), Lv(order * order);
            LagrangeInterp<Real>::Interpolate(Lu, nds, u_param);
            LagrangeInterp<Real>::Interpolate(Lv, nds, v_param);

            // Interpolate the nodal density onto the panel quad nodes.
            Vector<Real> sigma_q(nq * KDIM0);
            sigma_q.SetZero();
            for (Integer a = 0; a < order; a++) {
                for (Integer b = 0; b < order; b++) {
                    const Long q = a * order + b;
                    for (Integer i = 0; i < order; i++) {
                        for (Integer j = 0; j < order; j++) {
                            const Real L = Lu[i * order + a] * Lv[j * order + b];
                            for (Integer k0 = 0; k0 < KDIM0; k0++) {
                                sigma_q[q * KDIM0 + k0] += sigma[(i * order + j) * KDIM0 + k0] * L;
                            }
                        }
                    }
                }
            }

            // Kernel matrix from this panel's sources to the single target.
            // KernelMatrix already applies uKerScaleFactor (matches NearInterac).
            Matrix<Real> Mker; // (nq*KDIM0 x KDIM1)
            ker.template KernelMatrix<Real, false>(Mker, Xt, X, Xn);

            for (Integer a = 0; a < order; a++) {
                for (Integer b = 0; b < order; b++) {
                    const Long q = a * order + b;
                    // Surface quadrature weight; each panel covers 1/nsub of the
                    // patch per direction (the 1/nsub^2 Jacobian).
                    const Real wq = Xa[q] * wts[a] * wts[b] / ((Real)nsub * (Real)nsub);
                    for (Integer k0 = 0; k0 < KDIM0; k0++) {
                        for (Integer k1 = 0; k1 < KDIM1; k1++) {
                            u[k1] += Mker[q * KDIM0 + k0][k1] * sigma_q[q * KDIM0 + k0] * wq;
                        }
                    }
                }
            }
        }
    }
    return u;
}

template <class Real, class Kernel> void test_NearInterac(const Kernel& ker, const bool curved, const char* label) {
    const Integer COORD_DIM = 3;
    const Integer order = 8;
    const Integer KDIM0 = Kernel::SrcDim();
    const Integer KDIM1 = Kernel::TrgDim();
    const Long nnode = (Long)order * order;
    const Long elem_idx = 0;

    // Single element over [0,1]^2: flat plane z = 0, or the curved testsurf z = u*v.
    Vector<Real> coord0 = curved ? get_testsurf<Real>(order, 1)
                                 : QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Target in the near-singular regime: offset a distance d off an interior
    // surface point along the unit surface normal.
    const Real u0 = 0.4, v0 = 0.6, d = 0.1;
    Vector<Real> up{u0}, vp{v0}, Xsurf, Nsurf;
    qel.GetGeom(&Xsurf, &Nsurf, nullptr, nullptr, nullptr, up, vp, elem_idx);
    Vector<Real> Xt(COORD_DIM);
    for (Integer k = 0; k < COORD_DIM; k++) Xt[k] = Xsurf[k] + d * Nsurf[k];

    // Smooth nodal density (AoS over components). The exact values are irrelevant
    // to the comparison: both schemes integrate the identical Lagrange interpolant.
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Vector<Real> sigma(nnode * KDIM0);
    for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
            for (Integer k0 = 0; k0 < KDIM0; k0++) {
                sigma[(i * order + j) * KDIM0 + k0] = cos<Real>(nds[i] + 2 * nds[j] + (Real)0.5 * k0);
            }
        }
    }

    // Adaptive near-interaction matrix and the potential M^T * sigma.
    Matrix<Real> M;
    Vector<Real> normal_trg; // empty: no target-normal contraction for these kernels
    const Real tol = 1e-08;
    QuadElemList<Real>::template NearInterac<Kernel>(M, Xt, normal_trg, ker, tol, elem_idx, &qel);
    SCTL_ASSERT(M.Dim(0) == nnode * KDIM0 && M.Dim(1) == KDIM1); // single target

    Vector<Real> u_near(KDIM1);
    u_near.SetZero();
    for (Long r = 0; r < nnode * KDIM0; r++) {
        for (Integer k1 = 0; k1 < KDIM1; k1++) u_near[k1] += sigma[r] * M[r][k1];
    }

    // Reference: globally upsampled direct quadrature (same GL order).
    const Long nsub = 12;
    Vector<Real> u_ref = direct_upsampled_potential<Real, Kernel>(qel, elem_idx, sigma, Xt, ker, nsub);

    // Relative error in the target potential.
    Real err2 = 0, ref2 = 0;
    for (Integer k1 = 0; k1 < KDIM1; k1++) {
        const Real e = u_near[k1] - u_ref[k1];
        err2 += e * e;
        ref2 += u_ref[k1] * u_ref[k1];
    }
    const Real rel_err = sqrt<Real>(err2) / sqrt<Real>(ref2);

    const Real rel_tol = 1e-6;
    if (!(rel_err < rel_tol)) {
        std::cout << "test_NearInterac (" << label << "): rel_err = " << rel_err << "\n";
    }
    SCTL_ASSERT(rel_err < rel_tol);
}

// Test the singular self-interaction (1D-reduction scheme) against a closed-form
// analytical solution. For the Laplace single-layer kernel Laplace3D_FxU over the
// flat unit square (z = 0 on [0,1]^2, so x = u, y = v) with the constant density
// sigma == 1, the on-surface single-layer potential at a target (x0, y0, 0) is the
// Newtonian potential of a uniform unit square in its own plane:
//
//   u(x0,y0) = 1/(4 pi) \int_0^1 \int_0^1 dx dy / sqrt((x-x0)^2 + (y-y0)^2).
//
// The 1/r kernel has the elementary antiderivative
//   F(X,Y) = X*asinh(Y/X) + Y*asinh(X/Y)
//          = X*ln(Y + sqrt(X^2+Y^2)) + Y*ln(X + sqrt(X^2+Y^2)),  d^2F/dXdY = 1/r,
// so the integral over the shifted rectangle [-x0,1-x0] x [-y0,1-y0] is the
// corner sum F(1-x0,1-y0) - F(1-x0,-y0) - F(-x0,1-y0) + F(-x0,-y0).
//
// The self operator is applied as u = sigma^T M (sigma the nodal density vector).
// Because the Lagrange nodal basis is a partition of unity, the all-ones nodal
// vector represents sigma == 1 exactly, so u[t] = sum_p M[p][t] is the singular
// integral above evaluated at surface node t -- compared to the closed form.
template <class Real> void test_SelfInterac() {
    const Integer order = 8;
    const Long nnode = (Long)order * order;

    // Flat unit square z = 0 over [0,1]^2 (ParamGrid leaves z = 0).
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    const Laplace3D_FxU ker; // scalar single-layer: KDIM0 = KDIM1 = 1
    const Real tol = 1e-10;

    // Self-interaction matrix (scalar kernel, no target-normal contraction).
    Vector<Matrix<Real>> M_lst(1);
    QuadElemList<Real>::template SelfInterac<Laplace3D_FxU>(M_lst, ker, tol, /*trg_dot_prod=*/false, &qel);

    // Shape + finiteness.
    SCTL_ASSERT(M_lst.Dim() == 1);
    const Matrix<Real>& M = M_lst[0];
    SCTL_ASSERT(M.Dim(0) == nnode && M.Dim(1) == nnode);
    for (Long r = 0; r < M.Dim(0); r++) {
        for (Long c = 0; c < M.Dim(1); c++) SCTL_ASSERT(std::isfinite(M[r][c]));
    }

    // Closed-form unit-square single-layer potential at an on-surface target.
    auto Fanti = [](Real X, Real Y) {
        const Real r = sqrt<Real>(X * X + Y * Y);
        return X * log<Real>(Y + r) + Y * log<Real>(X + r);
    };
    auto u_exact = [&](Real x0, Real y0) {
        const Real I = Fanti(1 - x0, 1 - y0) - Fanti(1 - x0, -y0)
                     - Fanti(-x0, 1 - y0) + Fanti(-x0, -y0);
        return I / (4 * const_pi<Real>());
    };

    // Apply the self operator to the constant (all-ones) density and compare the
    // potential at every surface node to the analytical value.
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    Real max_rel = 0;
    for (Integer ti = 0; ti < order; ti++) {
        for (Integer tj = 0; tj < order; tj++) {
            const Long t = ti * order + tj;
            Real u = 0;
            for (Long p = 0; p < nnode; p++) u += M[p][t]; // sigma == 1
            const Real ue = u_exact(nds[ti], nds[tj]);
            const Real rel = fabs(u - ue) / fabs(ue);
            max_rel = std::max<Real>(max_rel, rel);
        }
    }
    std::cout << "  test_SelfInterac (Laplace3D_FxU / plane): max_rel = " << max_rel << "\n";
    const Real rel_tol = 1e-6;
    SCTL_ASSERT(max_rel < rel_tol);
}


// Test access shim for QuadElemList's private static helpers (befriended in the
// header). Forwards to the otherwise-inaccessible members under test. Must live
// in namespace sctl to match the friend declaration in quad_element.hpp.
namespace sctl {
template <class Real> struct QuadElemTestAccess {
    static void LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer order) {
        QuadElemList<Real>::LogSingularQuad1D(param, w, v0, order);
    }
};
}

// Verify the 1D log-singular quadrature rule produced by LogSingularQuad1D.
//
// The rule approximates  I[f] = \int_0^1 f(v) dv  for integrands f with an
// integrable logarithmic singularity at the interior point v0 (Alpert's hybrid
// Gauss-trapezoidal scheme; the interval is split at v0 so the singularity sits
// at a sub-interval endpoint). We integrate a family of test functions whose
// exact integral is known in closed form and check the quadrature error:
//
//   (a) f(v) = log|v - v0|                          [pure log singularity]
//   (b) f(v) = v * log|v - v0|                       [smooth factor x log]
//   (c) f(v) = (1 + v^2) * log|v - v0| + cos(3 v)    [log + smooth part]
//   (d) f(v) = cos(3 v)                              [purely smooth, no sing.]
//
// The closed forms below come from  \int_0^1 v^k log|v - a| dv:
//   k=0:  a*log(a) + (1-a)*log(1-a) - 1
//   k=1:  ((1 - a^2)/2)*log(1-a) + (a^2/2)*log(a) - 1/4 - a/2
//   k=2:  ((1 - a^3)/3)*log(1-a) - (a^3/3)*log(a) ... (assembled inline below)
template <class Real> void test_LogSingularQuad1D() {
    const Real v0 = (Real)0.6;
    const Integer order = 16; // requested correction order (rule snaps internally)

    Vector<Real> param, w;
    QuadElemTestAccess<Real>::LogSingularQuad1D(param, w, v0, order);

    // Basic structural sanity: matching sizes, nodes inside (0,1), weights sum to 1
    // (the rule must integrate f == 1 exactly to high accuracy).
    SCTL_ASSERT(param.Dim() == w.Dim());
    SCTL_ASSERT(param.Dim() > 0);
    Real wsum = 0;
    for (Long i = 0; i < param.Dim(); i++) {
        SCTL_ASSERT(param[i] > (Real)0 && param[i] < (Real)1);
        wsum += w[i];
    }
    SCTL_ASSERT(fabs(wsum - (Real)1) < (Real)1e-12);

    auto quad = [&](auto f) {
        Real I = 0;
        for (Long i = 0; i < param.Dim(); i++) I += w[i] * f(param[i]);
        return I;
    };

    const Real a = v0;
    const Real la = log<Real>(a), lb = log<Real>(1 - a);

    // (a) f = log|v - v0|
    {
        const Real I = quad([&](Real v) { return log<Real>(fabs(v - v0)); });
        const Real I_exact = a * la + (1 - a) * lb - 1;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=log|v-v0|        I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }

    // (b) f = v * log|v - v0|
    {
        const Real I = quad([&](Real v) { return v * log<Real>(fabs(v - v0)); });
        const Real I_exact = ((1 - a * a) / 2) * lb + (a * a / 2) * la - (Real)0.25 - a / 2;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=v*log|v-v0|      I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }

    // (c) f = (1 + v^2) * log|v - v0| + cos(3 v)
    //   \int_0^1 v^2 log|v - a| dv assembled from the antiderivative
    //   F(x) = ((x^3 - a^3)/3) log|x - a| - x^3/9 - a x^2/6 - a^2 x/3, F continuous
    //   through x=a, so I = F(1) - F(0).
    {
        const Real I = quad([&](Real v) {
            return (1 + v * v) * log<Real>(fabs(v - v0)) + cos<Real>(3 * v);
        });
        const Real I0 = a * la + (1 - a) * lb - 1;                                   // \int v^0 log
        const Real F1 = ((1 - a * a * a) / 3) * lb - (Real)1 / 9 - a / 6 - a * a / 3; // F(1)
        const Real F0 = (-a * a * a / 3) * la;                                        // F(0)
        const Real I2 = F1 - F0;                                                      // \int v^2 log
        const Real Icos = sin<Real>((Real)3) / 3;                                     // \int_0^1 cos(3v)
        const Real I_exact = I0 + I2 + Icos;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=(1+v^2)log+cos   I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-9);
    }

    // (d) purely smooth integrand (no singularity); the rule must still be high order.
    {
        const Real I = quad([&](Real v) { return cos<Real>(3 * v); });
        const Real I_exact = sin<Real>((Real)3) / 3;
        const Real err = fabs(I - I_exact);
        std::cout << "  test_LogSingularQuad1D: f=cos(3v)          I=" << I
                  << " exact=" << I_exact << " err=" << err << "\n";
        SCTL_ASSERT(err < (Real)1e-10);
    }
}

// Check the interpolation error incurred when the self-interaction quadrature
// samples a field at the Alpert log-singular quadrature nodes.
//
// IntegrateBlock never evaluates the true density/geometry at its quadrature
// nodes: it evaluates the order-`order` tensor-product Lagrange interpolant built
// from the patch's order x order GL node values (the Minterp operator in
// quad_element.cpp), then samples that interpolant at the quadrature nodes -- in
// the v-direction these are exactly the Alpert nodes from LogSingularQuad1D. For
// a NON-POLYNOMIAL field on a NON-FLAT surface this interpolation is not exact,
// so it sets a floor on the achievable self-interaction accuracy. Here we build
// the interpolant of a non-polynomial function sampled on the curved testsurf
// patch (z = u*v), evaluate it at the Alpert nodes in BOTH parameter directions,
// and confirm the interpolation error sits at the expected (spectral) level.
template <class Real> void test_QuadNodeInterp() {
    const Integer order = 8;
    const Long elem_idx = 0;

    // Non-flat patch z = u*v over [0,1]^2; its order-12 Lagrange interpolant is the
    // representation the quadrature actually integrates.
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);

    // Non-polynomial scalar field, sampled in physical (x,y,z) space. Composed with
    // the surface map X(u,v) = (u, v, u*v) it is a non-polynomial function of (u,v).
    auto g = [](const Real* X) {
        return exp<Real>((Real)0.5 * X[0]) * cos<Real>(X[1]) + sin<Real>(X[2]);
    };

    // Field values at the order x order patch nodes (the interpolation data).
    Vector<Real> Xpatch;
    qel.GetGeom(&Xpatch, nullptr, nullptr, nullptr, nullptr, nds, nds, elem_idx);
    Vector<Real> f_patch(order * order);
    for (Long p = 0; p < order * order; p++) f_patch[p] = g(&Xpatch[p * 3]);

    // Alpert log-singular quadrature nodes (LogSingularQuad1D) in u and v; together
    // they form the tensor-product target grid (node (a,b) at flat index a*Nv + b).
    Vector<Real> u_param, v_param, wu, wv;
    QuadElemTestAccess<Real>::LogSingularQuad1D(u_param, wu, (Real)0.3, order);
    QuadElemTestAccess<Real>::LogSingularQuad1D(v_param, wv, (Real)0.6, order);
    const Long Nu = u_param.Dim(), Nv = v_param.Dim();

    // Lagrange weights from patch nodes to the Alpert nodes (as in IntegrateBlock):
    // Mu[i*Nu + a] = L_i(u_param[a]).
    Vector<Real> Mu(order * Nu), Mv(order * Nv);
    LagrangeInterp<Real>::Interpolate(Mu, nds, u_param);
    LagrangeInterp<Real>::Interpolate(Mv, nds, v_param);

    // Exact field at the Alpert nodes, via the surface geometry there.
    Vector<Real> Xquad;
    qel.GetGeom(&Xquad, nullptr, nullptr, nullptr, nullptr, u_param, v_param, elem_idx);

    // Compare the tensor-product Lagrange interpolant against the exact field.
    Real max_err = 0, max_f = 0;
    for (Long a = 0; a < Nu; a++) {
        for (Long b = 0; b < Nv; b++) {
            Real f_interp = 0;
            for (Integer i = 0; i < order; i++) {
                for (Integer j = 0; j < order; j++) {
                    f_interp += f_patch[i * order + j] * Mu[i * Nu + a] * Mv[j * Nv + b];
                }
            }
            const Real f_exact = g(&Xquad[(a * Nv + b) * 3]);
            max_err = std::max<Real>(max_err, fabs(f_interp - f_exact));
            max_f   = std::max<Real>(max_f, fabs(f_exact));
        }
    }
    const Real rel_err = max_err / max_f;
    std::cout << "  test_QuadNodeInterp: order=" << order << " Nu=" << Nu << " Nv=" << Nv
              << " max_abs_err=" << max_err << " rel_err=" << rel_err << "\n";
    const Real rel_tol = 1e-6;
    SCTL_ASSERT(rel_err < rel_tol);
}

int main(int argc, char** argv) {

    using Real = double;

    test_ParamGrid<Real>();
    std::cout << "test_ParamGrid: PASSED\n";
    test_GetClosestNode_plane<Real>();
    std::cout << "test_GetClosestNode_plane: PASSED\n";
    test_GetClosestNode_curved<Real>();
    std::cout << "test_GetClosestNode_curved: PASSED\n";
    test_LogSingularQuad1D<Real>();
    std::cout << "test_LogSingularQuad1D: PASSED\n";
    test_QuadNodeInterp<Real>();
    std::cout << "test_QuadNodeInterp: PASSED\n";

    // // NearInterac: adaptive scheme vs. globally upsampled direct quadrature, over
    // // {Stokes single-/double-layer} x {flat plane, curved testsurf}.
    const Stokes3D_FxU ker_FxU;
    const Stokes3D_DxU ker_DxU;
    test_NearInterac<Real>(ker_FxU, false, "Stokes3D_FxU / plane");
    std::cout << "test_NearInterac (Stokes3D_FxU / plane): PASSED\n";

    test_NearInterac<Real>(ker_FxU, true,  "Stokes3D_FxU / testsurf");
    std::cout << "test_NearInterac (Stokes3D_FxU / testsurf): PASSED\n";

    test_NearInterac<Real>(ker_DxU, false, "Stokes3D_DxU / plane");
    std::cout << "test_NearInterac (Stokes3D_DxU / plane): PASSED\n";

    test_NearInterac<Real>(ker_DxU, true,  "Stokes3D_DxU / testsurf");
    std::cout << "test_NearInterac (Stokes3D_DxU / testsurf): PASSED\n";

    // SelfInterac: 1D-reduction singular scheme vs. the closed-form Laplace
    // single-layer potential of a uniform unit square (flat plane).
    test_SelfInterac<Real>();
    std::cout << "test_SelfInterac (Laplace3D_FxU / plane): PASSED\n";
    
    return 0;
}