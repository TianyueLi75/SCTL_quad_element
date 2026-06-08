// TODO: unit testing framework

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

// template <class Real> void test_ParamGrid() {
//     // Test tensor grid generation directly on ParamGrid (not via get_testsurf,
//     // whose z = x*y belongs to the test surface and not to ParamGrid).
//     const Long order = 4;
//     const Long nelem_perside = 2;
//     const Long N_per_side = order * nelem_perside; // 8 nodes per side
//     const Long N_total = N_per_side * N_per_side;  // 64 tensor-grid points

//     Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, nelem_perside);
//     assert(coord0.Dim() == N_total * 3);

//     // Explicit expected 1-D nodes: order-4 Gauss-Legendre nodes on [-1,1] are
//     // +-0.339981043584856 and +-0.861136311594053. Mapped to [0,1] via (x+1)/2
//     // and split into 2 equal panels [0,0.5] and [0.5,1], the 8 per-side
//     // coordinates are:
//     const Real x_param_exp[8] = {
//         0.034715922101486804, // panel 0
//         0.165004739103786020,
//         0.334995260896213980,
//         0.465284077898513196,
//         0.534715922101486804, // panel 1
//         0.665004739103786020,
//         0.834995260896213980,
//         0.965284077898513196
//     };

//     // The grid is the tensor product x_param_exp (x) x_param_exp, AoS order with
//     // u (xind) the slow index and v (yind) the fast index, with z left at 0.
//     const Real tol = 1e-12;
//     for (Long xind = 0; xind < N_per_side; xind++) {
//         for (Long yind = 0; yind < N_per_side; yind++) {
//             const Long idx = (xind * N_per_side + yind) * 3;
//             assert(fabs(coord0[idx + 0] - x_param_exp[xind]) < tol);
//             assert(fabs(coord0[idx + 1] - x_param_exp[yind]) < tol);
//             assert(fabs(coord0[idx + 2] - (Real)0) < tol);
//         }
//     }
// }

// template <class Real> void test_Upsample() {
//     // Upsample subdivides each input patch (order x order GL nodes) into
//     // nsub_perside x nsub_perside sub-panels of order x order GL nodes,
//     // interpolating the geometry (Xin) and field (Fin) and returning the surface
//     // quadrature weights (Wout).
//     const Long order = 8;
//     const Long nsub_perside = 2; // subdivide the patch into 2x2 sub-panels
//     const Long nnode = order * order;

//     // Single patch on the reference square [0,1]^2 carrying the polynomial graph
//     // surface (x,y,z) = (u, v, u*v). Since x, y and z are polynomials of degree
//     // <= order-1, the order-`order` interpolant reproduces them exactly, so the
//     // upsampled nodes equal the surface evaluated at the refined parameters.
//     Vector<Real> coord0 = get_testsurf<Real>(order, 1);
//     QuadElemList<Real> qel = QuadElemList<Real>(order, coord0);

//     Vector<Real> Xin;
//     qel.GetNodeCoord(&Xin, nullptr, nullptr);

//     // Scalar polynomial field F(u,v) = u + 2v + 3uv defined at the patch nodes.
//     // Node (i,j) (stored at p = i*order + j) sits at (u,v) = (nds[i], nds[j]).
//     const auto F = [](const Real u, const Real v) { return u + 2 * v + 3 * u * v; };
//     const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
//     Vector<Real> Fin(nnode);
//     for (Long i = 0; i < order; i++) {
//         for (Long j = 0; j < order; j++) {
//             Fin[i * order + j] = F(nds[i], nds[j]);
//         }
//     }

//     Vector<Real> Xout, Xnout, Fout, Wout;
//     qel.Upsample(Xout, Xnout, Fout, Wout, Xin, Fin, nsub_perside);

//     // Expected sizes: one input patch -> nsub_perside^2 sub-panel elements.
//     const Long nelem_out = nsub_perside * nsub_perside;
//     assert(Xout.Dim() == nelem_out * nnode * 3);
//     assert(Fout.Dim() == nelem_out * nnode);
//     assert(Wout.Dim() == nelem_out * nnode);

//     // Compare against the analytic surface / field / quadrature weights. For the
//     // graph z = u*v we have X_u = (1,0,v), X_v = (0,1,u), so the area element is
//     // |X_u x X_v| = sqrt(u^2 + v^2 + 1). Each sub-panel covers 1/nsub_perside of
//     // the patch per direction, contributing the (1/nsub_perside)^2 Jacobian.
//     const Vector<Real>& wts = LegQuadRule<Real>::wts(order);
//     const Real inv_nps = (Real)1 / nsub_perside;
//     const Real tol = 1e-10;

//     for (Long pi = 0; pi < nsub_perside; pi++) {
//         for (Long pj = 0; pj < nsub_perside; pj++) {
//             const Long out_elem = pi * nsub_perside + pj;
//             for (Long a = 0; a < order; a++) {
//                 for (Long b = 0; b < order; b++) {
//                     const Real u = (nds[a] + pi) * inv_nps;
//                     const Real v = (nds[b] + pj) * inv_nps;
//                     const Long out_node = out_elem * nnode + (a * order + b);

//                     // Coordinates (u, v, u*v).
//                     assert(fabs(Xout[out_node * 3 + 0] - u) < tol);
//                     assert(fabs(Xout[out_node * 3 + 1] - v) < tol);
//                     assert(fabs(Xout[out_node * 3 + 2] - u * v) < tol);

//                     // Interpolated field.
//                     assert(fabs(Fout[out_node] - F(u, v)) < tol);

//                     // Quadrature weight.
//                     const Real area = sqrt<Real>(u * u + v * v + 1);
//                     const Real w_exp = area * wts[a] * wts[b] * inv_nps * inv_nps;
//                     assert(fabs(Wout[out_node] - w_exp) < tol);
//                 }
//             }
//         }
//     }
// }

template <class Real> void test_GetClosestPoint_plane() {
    // Flat patch z = 0 over [0,1]^2 (x = u, y = v); ParamGrid leaves z = 0. Both
    // x and y are degree-1 in (u,v), so the order-`order` interpolant is exact.
    const Long order = 8;
    Vector<Real> coord0 = QuadElemList<Real>::ParamGrid(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // For a target above the plane, the closest point is its (x,y) projection,
    // the distance is |z|, and the surface normal (dX/du x dX/dv) is +z.
    const Vector<Real> Xtrg = {0.3, 0.7, 0.5};
    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, &Xstar, &Nstar, 0, Xtrg);

    const Real tol = 1e-9;
    assert(fabs(ustar - 0.3) < tol);
    assert(fabs(vstar - 0.7) < tol);
    assert(fabs(dist  - 0.5) < tol);
    assert(fabs(Xstar[0] - 0.3) < tol);
    assert(fabs(Xstar[1] - 0.7) < tol);
    assert(fabs(Xstar[2] - 0.0) < tol);
    assert(fabs(Nstar[0]) < tol);
    assert(fabs(Nstar[1]) < tol);
    assert(fabs(fabs(Nstar[2]) - 1.0) < tol);
}

template <class Real> void test_GetClosestPoint_curved() {
    // Curved patch z = u*v over [0,1]^2 (the fsurf from get_testsurf); z is
    // tensor-degree (1,1), so the order-`order` interpolant is exact.
    const Long order = 8;
    Vector<Real> coord0 = get_testsurf<Real>(order, 1);
    QuadElemList<Real> qel(order, coord0);

    // Offset the target a distance d along the unit surface normal at an interior
    // surface point Y(u0,v0). Then (u0,v0) is a stationary point of the distance
    // and d is well below the radius of curvature, so Y(u0,v0) is the unique
    // closest point at distance d. For z = u*v: X_u=(1,0,v0), X_v=(0,1,u0), so
    // the (un-normalized) normal is n = X_u x X_v = (-v0, -u0, 1).
    const Real u0 = 0.4, v0 = 0.6, d = 0.1;
    const Real Yx = u0, Yy = v0, Yz = u0*v0;
    Real nx = -v0, ny = -u0, nz = 1;
    const Real nrm = sqrt<Real>(nx*nx + ny*ny + nz*nz);
    nx /= nrm; ny /= nrm; nz /= nrm;

    const Vector<Real> Xtrg = {Yx + d*nx, Yy + d*ny, Yz + d*nz};
    Real ustar, vstar;
    Vector<Real> Xstar, Nstar;
    const Real dist = qel.GetClosestPoint(ustar, vstar, &Xstar, &Nstar, 0, Xtrg);

    const Real tol = 1e-8;
    assert(fabs(ustar - u0) < tol);
    assert(fabs(vstar - v0) < tol);
    assert(fabs(dist  - d ) < tol);
    assert(fabs(Xstar[0] - Yx) < tol);
    assert(fabs(Xstar[1] - Yy) < tol);
    assert(fabs(Xstar[2] - Yz) < tol);
    assert(fabs(Nstar[0] - nx) < tol);
    assert(fabs(Nstar[1] - ny) < tol);
    assert(fabs(Nstar[2] - nz) < tol);
}




int main(int argc, char** argv) {

    using Real = double;

    // test_ParamGrid<Real>();
    // test_Upsample<Real>();
    test_GetClosestPoint_plane<Real>();
    test_GetClosestPoint_curved<Real>();

    // std::cout << "test_ParamGrid: PASSED\n";
    // std::cout << "test_Upsample: PASSED\n";
    std::cout << "test_GetClosestPoint_plane: PASSED\n";
    std::cout << "test_GetClosestPoint_curved: PASSED\n";
    return 0;
}