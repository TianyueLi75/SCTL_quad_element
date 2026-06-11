#ifndef _SCTL_QUAD_ELEMENT_CPP_
#define _SCTL_QUAD_ELEMENT_CPP_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sctl.hpp>
#include "sctl/experimental/quad_element.hpp"
#include "sctl/experimental/alpert_quadr.cpp"

namespace sctl {

  template <class Real> template <class ValueType> QuadElemList<Real>::QuadElemList(Integer order0, const Vector<ValueType>& coord0) {
    Init(order0, coord0);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Init(Integer order0, const Vector<ValueType>& coord0) {
    order = order0;
    SCTL_ASSERT(order > 0);

    const Long nnode_per_elem = (Long)order * order;
    SCTL_ASSERT(coord0.Dim() % (nnode_per_elem * COORD_DIM) == 0);
    nelem = coord0.Dim() / (nnode_per_elem * COORD_DIM);

    coord.ReInit(nelem * COORD_DIM * nnode_per_elem);
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      const Long base = elem_idx * COORD_DIM * nnode_per_elem;
      for (Integer k = 0; k < COORD_DIM; k++) {
        for (Long p = 0; p < nnode_per_elem; p++) {
          coord[base + k * nnode_per_elem + p] = (Real)coord0[(elem_idx * nnode_per_elem + p) * COORD_DIM + k];
        }
      }
    }

    BuildDerivativeCache();
  }

  template <class Real> void QuadElemList<Real>::BuildDerivativeCache() {
    dcoord_du.ReInit(coord.Dim());
    dcoord_dv.ReInit(coord.Dim());

    const Long nnode_per_elem = (Long)order * order;
    const auto& nodes = ParamNodes(order);

    Vector<Real> line_in(order), line_out(order);
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Long comp_base = (elem_idx * COORD_DIM + k) * nnode_per_elem;

        for (Integer j = 0; j < order; j++) {
          for (Integer i = 0; i < order; i++) {
            line_in[i] = coord[comp_base + i * order + j];
          }
          LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
          for (Integer i = 0; i < order; i++) {
            dcoord_du[comp_base + i * order + j] = line_out[i];
          }
        }

        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            line_in[j] = coord[comp_base + i * order + j];
          }
          LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
          for (Integer j = 0; j < order; j++) {
            dcoord_dv[comp_base + i * order + j] = line_out[j];
          }
        }
      }
    }
  }

  template <class Real> Long QuadElemList<Real>::Size() const {
    return nelem;
  }

  template <class Real> Integer QuadElemList<Real>::Order() const {
    return order;
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::EvalTensorProduct(Vector<ValueType>& out, const Vector<ValueType>& in, const Matrix<ValueType>& MuT, const Matrix<ValueType>& Mv) {
    const Integer Nu = MuT.Dim(0);
    const Integer Nv = Mv.Dim(1);
    const Integer order = MuT.Dim(1);
    SCTL_ASSERT(Mv.Dim(0) == order);
    const Long ncomp = in.Dim() / (order * order);
    SCTL_ASSERT(in.Dim() == ncomp * order * order);

    const Long Nout = (Long)Nu * Nv;
    if (out.Dim() != ncomp * Nout) out.ReInit(ncomp * Nout);

    constexpr Integer Nbuff = 1024;
    StaticArray<ValueType,Nbuff> tmp_buf;
    Matrix<ValueType> tmp(order, Nv, (order * Nv > Nbuff ? NullIterator<ValueType>() : tmp_buf), order * Nv > Nbuff);

    for (Long k = 0; k < ncomp; k++) {
      const Matrix<ValueType> in_(order, order, (Iterator<ValueType>)in.begin() + k * order * order, false);
      Matrix<ValueType> out_(Nu, Nv, out.begin() + k * Nout, false);
      Matrix<ValueType>::GEMM(tmp, in_, Mv);
      Matrix<ValueType>::GEMM(out_, MuT, tmp);
    }
  }

  template <class Real> void QuadElemList<Real>::GetGeom(Vector<Real>* X, Vector<Real>* Xn, Vector<Real>* Xa, Vector<Real>* dX_du, Vector<Real>* dX_dv, const Vector<Real>& u_param, const Vector<Real>& v_param, const Long elem_idx) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nu = u_param.Dim();
    const Long Nv = v_param.Dim();
    const Long N = Nu * Nv;

    if (X && X->Dim() != N * COORD_DIM) X->ReInit(N * COORD_DIM);
    if (Xn && Xn->Dim() != N * COORD_DIM) Xn->ReInit(N * COORD_DIM);
    if (Xa && Xa->Dim() != N) Xa->ReInit(N);
    if (dX_du && dX_du->Dim() != N * COORD_DIM) dX_du->ReInit(N * COORD_DIM);
    if (dX_dv && dX_dv->Dim() != N * COORD_DIM) dX_dv->ReInit(N * COORD_DIM);

    Matrix<Real> MuT(order, Nu), Mv(order, Nv);
    Vector<Real> Mu_(order * Nu, MuT.begin(), false);
    Vector<Real> Mv_(order * Nv, Mv.begin(), false);
    LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_param);
    LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_param);
    MuT = MuT.Transpose();

    SCTL_ASSERT(elem_idx >= 0 && elem_idx < nelem);
    const Long base = elem_idx * nnode_per_elem * COORD_DIM;
    const Vector<Real> coord_(COORD_DIM * nnode_per_elem, (Iterator<Real>)coord.begin() + base, false);
    const Vector<Real> dcoord_du_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_du.begin() + base, false);
    const Vector<Real> dcoord_dv_(COORD_DIM * nnode_per_elem, (Iterator<Real>)dcoord_dv.begin() + base, false);

    if (X) {
      Vector<Real> X_soa;
      EvalTensorProduct(X_soa, coord_, MuT, Mv);
      for (Long i = 0; i < N; i++) {
        (*X)[i * COORD_DIM + 0] = X_soa[0 * N + i];
        (*X)[i * COORD_DIM + 1] = X_soa[1 * N + i];
        (*X)[i * COORD_DIM + 2] = X_soa[2 * N + i];
      }
    }
    if (Xn || Xa || dX_du || dX_dv) {
      Vector<Real> dXdu_soa, dXdv_soa;
      EvalTensorProduct(dXdu_soa, dcoord_du_, MuT, Mv);
      EvalTensorProduct(dXdv_soa, dcoord_dv_, MuT, Mv);
      for (Long i = 0; i < N; i++) {
        const Real du0 = dXdu_soa[0 * N + i];
        const Real du1 = dXdu_soa[1 * N + i];
        const Real du2 = dXdu_soa[2 * N + i];
        const Real dv0 = dXdv_soa[0 * N + i];
        const Real dv1 = dXdv_soa[1 * N + i];
        const Real dv2 = dXdv_soa[2 * N + i];

        const Real n0 = du1 * dv2 - du2 * dv1;
        const Real n1 = du2 * dv0 - du0 * dv2;
        const Real n2 = du0 * dv1 - du1 * dv0;
        const Real area = sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
        const Real inv_area = (area > 0 ? 1 / area : 0);

        if (Xn) {
          (*Xn)[i * COORD_DIM + 0] = n0 * inv_area;
          (*Xn)[i * COORD_DIM + 1] = n1 * inv_area;
          (*Xn)[i * COORD_DIM + 2] = n2 * inv_area;
        }
        if (Xa) {
          (*Xa)[i] = area;
        }
        if (dX_du) {
          (*dX_du)[i * COORD_DIM + 0] = du0;
          (*dX_du)[i * COORD_DIM + 1] = du1;
          (*dX_du)[i * COORD_DIM + 2] = du2;
        }
        if (dX_dv) {
          (*dX_dv)[i * COORD_DIM + 0] = dv0;
          (*dX_dv)[i * COORD_DIM + 1] = dv1;
          (*dX_dv)[i * COORD_DIM + 2] = dv2;
        }
      }
    }
  }

  template <class Real> void QuadElemList<Real>::GetNodeCoord(Vector<Real>* X, Vector<Real>* Xn, Vector<Long>* element_wise_node_cnt) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nnode = nelem * nnode_per_elem;

    if (X && X->Dim() != Nnode * COORD_DIM) X->ReInit(Nnode * COORD_DIM);
    if (Xn && Xn->Dim() != Nnode * COORD_DIM) Xn->ReInit(Nnode * COORD_DIM);
    if (element_wise_node_cnt) {
      if (element_wise_node_cnt->Dim() != nelem) element_wise_node_cnt->ReInit(nelem);
      (*element_wise_node_cnt) = nnode_per_elem;
    }

    const auto& nodes = ParamNodes(order);
    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      Vector<Real> X_, Xn_;
      if (X) X_.ReInit(nnode_per_elem * COORD_DIM, X->begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      if (Xn) Xn_.ReInit(nnode_per_elem * COORD_DIM, Xn->begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      GetGeom((X ? &X_ : nullptr), (Xn ? &Xn_ : nullptr), nullptr, nullptr, nullptr, nodes, nodes, elem_idx);
    }
  }

  template <class Real> void QuadElemList<Real>::GetFarFieldNodes(Vector<Real>& X, Vector<Real>& Xn, Vector<Real>& wts, Vector<Real>& dist_far, Vector<Long>& element_wise_node_cnt, const Real tol) const {
    const Long nnode_per_elem = (Long)order * order;
    const Long Nnode = nelem * nnode_per_elem;

    if (X.Dim() != Nnode * COORD_DIM) X.ReInit(Nnode * COORD_DIM);
    if (Xn.Dim() != Nnode * COORD_DIM) Xn.ReInit(Nnode * COORD_DIM);
    if (wts.Dim() != Nnode) wts.ReInit(Nnode);
    if (dist_far.Dim() != Nnode) dist_far.ReInit(Nnode);
    if (element_wise_node_cnt.Dim() != nelem) element_wise_node_cnt.ReInit(nelem);
    element_wise_node_cnt = nnode_per_elem;

    const auto& nodes = ParamNodes(order);
    const auto& node_wts = LegQuadRule<Real>::wts(order);

    // Compute dist_nodes[i]: the minimum distance in parameter space from node i
    // to the boundary of the Bernstein ellipse for [0,1].
    //
    // GL quadrature of order n achieves accuracy ~rho^{-2n} for a function analytic
    // inside the Bernstein ellipse with parameter rho (foci at 0 and 1). Choosing
    // rho so that rho^{2n} = 64/(15*tol) guarantees the far-field quadrature error
    // is below tol. The ellipse has real semi-axis b = (rho+1/rho)/4 and imaginary
    // semi-axis a = (rho-1/rho)/4, centered at 0.5.
    //
    // For a node x on the real axis, the closest point on the ellipse satisfies
    // cos(theta) = 4*b*(x-0.5) (from the normal condition, using b^2-a^2 = 1/4).
    // When |cos(theta)| <= 1 the closest point is on the curved part of the ellipse:
    //   dist = a * sqrt(1 + (a^2/b^2 - 1) * cos^2(theta))
    // Otherwise the closest point is the vertex and dist = b - |x - 0.5|.
    Vector<Real> dist_nodes(order);
    {
      const Integer n = order;
      const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
      const Real rho = pow<Real>((64 / (15 * tol_)), 1 / (Real)(2 * n));
      const Real a = (rho - 1 / rho) / 4;
      const Real b = (rho + 1 / rho) / 4;
      for (Integer i = 0; i < n; i++) {
        dist_nodes[i] = b - fabs(nodes[i] - (Real)0.5);  // distance to vertex (fallback)
        const Real cos_t = 4 * b * (nodes[i] - (Real)0.5);
        if (fabs(cos_t) <= 1) {
          dist_nodes[i] = a * sqrt<Real>(1 + ((a * a) / (b * b) - 1) * cos_t * cos_t);
        }
      }
    }

    #pragma omp parallel for schedule(static)
    for (Long elem_idx = 0; elem_idx < nelem; elem_idx++) {
      Vector<Real> X_(nnode_per_elem * COORD_DIM, X.begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      Vector<Real> Xn_(nnode_per_elem * COORD_DIM, Xn.begin() + elem_idx * nnode_per_elem * COORD_DIM, false);
      Vector<Real> wts_(nnode_per_elem, wts.begin() + elem_idx * nnode_per_elem, false);
      Vector<Real> dist_far_(nnode_per_elem, dist_far.begin() + elem_idx * nnode_per_elem, false);

      Vector<Real> Xa, dXdu, dXdv;
      GetGeom(&X_, &Xn_, &Xa, &dXdu, &dXdv, nodes, nodes, elem_idx);

      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Long p = i * order + j;
          const Real wu = node_wts[i];
          const Real wv = node_wts[j];
          wts_[p] = Xa[p] * wu * wv;

          // Scale parameter-space distances to physical space by the arc-length
          // of the element in each direction; take the max over both directions.
          const Real du = sqrt<Real>(dXdu[p * COORD_DIM + 0] * dXdu[p * COORD_DIM + 0] +
                                     dXdu[p * COORD_DIM + 1] * dXdu[p * COORD_DIM + 1] +
                                     dXdu[p * COORD_DIM + 2] * dXdu[p * COORD_DIM + 2]);
          const Real dv = sqrt<Real>(dXdv[p * COORD_DIM + 0] * dXdv[p * COORD_DIM + 0] +
                                     dXdv[p * COORD_DIM + 1] * dXdv[p * COORD_DIM + 1] +
                                     dXdv[p * COORD_DIM + 2] * dXdv[p * COORD_DIM + 2]);
          dist_far_[p] = std::max(dist_nodes[i] * du, dist_nodes[j] * dv);
        }
      }
    }
  }
  
  template <class Real> void QuadElemList<Real>::QuadParams(const Real tol, Real& b_ellipse, Integer& QuadOrder) {
    // Fix the Bernstein ellipse parameter rho and derive the per-panel
    // Gauss-Legendre order from rho and tol (cf. SlenderElemList::NearInteracHelper).
    // A panel of physical extent L is admissible (resolved to tol by an order
    // QuadOrder GL rule) when the target is at distance >= b_ellipse*L.
    const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
    const double rho = 2.5;
    b_ellipse = (Real)((rho + 1/rho) / 4);
    QuadOrder = std::max<Integer>(1, (Integer)std::ceil(-std::log(((15.0*(rho*rho-1))/64.0)*(double)tol_)/std::log(rho)*0.5 + 1));
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::IntegrateBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Vector<Real>& u_param, const Vector<Real>& wu, const Vector<Real>& v_param, const Vector<Real>& wv, const Kernel& ker) {
    // Accumulate into M_acc the contribution of the tensor-product quadrature
    // (u_param x v_param), with parameter-space weights wu (x) wv, on element
    // elem_idx against the single target Xtrg. Shared by the near (per-leaf) and
    // self (whole-block) schemes. The tensor grid is u-slow/v-fast, so quadrature
    // node (a,b) has flat index q = a*Nv + b (cf. GetGeom).
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Integer order = qel.order;
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Nu = u_param.Dim();
    const Long Nv = v_param.Dim();
    const Long nq = Nu * Nv;
    if (!nq) return;

    const Vector<Real>& pnds = ParamNodes(order);

    // Geometry (position, normal, area element) at the tensor grid.
    Vector<Real> Xl, Xnl, Xal;
    qel.GetGeom(&Xl, &Xnl, &Xal, nullptr, nullptr, u_param, v_param, elem_idx);

    // 1D Lagrange weights from patch nodes to quad nodes: Mu[i*Nu+a] = L_i(u_param[a]).
    Vector<Real> Mu(order*Nu), Mv(order*Nv);
    LagrangeInterp<Real>::Interpolate(Mu, pnds, u_param);
    LagrangeInterp<Real>::Interpolate(Mv, pnds, v_param);

    // Shifted source coords (translation-invariant kernel improves conditioning
    // when the target is close), normals, and surface-quadrature weights.
    StaticArray<Real,COORD_DIM> Xt0{0, 0, 0};
    const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
    Vector<Real> Xsrc(nq*COORD_DIM), Xnsrc(nq*COORD_DIM), wq(nq);
    for (Long a = 0; a < Nu; a++) {
      for (Long b = 0; b < Nv; b++) {
        const Long q = a*Nv + b;
        for (Integer k = 0; k < COORD_DIM; k++) {
          Xsrc[q*COORD_DIM+k] = Xl[q*COORD_DIM+k] - Xtrg[k];
          Xnsrc[q*COORD_DIM+k] = Xnl[q*COORD_DIM+k];
        }
        wq[q] = Xal[q]*wu[a]*wv[b];
      }
    }

    // Kernel matrix from sources to the (shifted) single target;
    Matrix<Real> Mker;
    ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)

    // Weighted kernel KW (nq x KDIM0*KDIM1_out); for the dot-product case, contract
    // the innermost COORD_DIM with the target normal.
    Matrix<Real> KW(nq, KDIM0*KDIM1_out);
    for (Long q = 0; q < nq; q++) {
      for (Integer k0 = 0; k0 < KDIM0; k0++) {
        for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
          Real val;
          if (trg_dot_prod) {
            val = 0;
            for (Integer l = 0; l < COORD_DIM; l++) {
              val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[l];
            }
          } else {
            val = Mker[q*KDIM0+k0][k1];
          }
          KW[q][k0*KDIM1_out+k1] = val*wq[q];
        }
      }
    }

    // Minterp (patch nodal basis -> quad nodes), tensor product of Mu, Mv.
    Matrix<Real> Minterp(nnode, nq);
    for (Integer i = 0; i < order; i++) {
      for (Integer j = 0; j < order; j++) {
        for (Long a = 0; a < Nu; a++) {
          for (Long b = 0; b < Nv; b++) {
            Minterp[i*order+j][a*Nv+b] = Mu[i*Nu+a]*Mv[j*Nv+b];
          }
        }
      }
    }

    Matrix<Real> tmp(nnode, KDIM0*KDIM1_out);
    Matrix<Real>::GEMM(tmp, Minterp, KW);
    for (Long r = 0; r < nnode; r++) {
      for (Long c = 0; c < KDIM0*KDIM1_out; c++) M_acc[r][c] += tmp[r][c];
    }
  }

  template <class Real> void QuadElemList<Real>::BuildGraded1D(Vector<Real>& param, Vector<Real>& w, const QuadElemList<Real>& qel, const Long elem_idx, const Real center, const Real cross, const Integer dir, const Real b_ellipse, const Vector<Real>& Xtrg, const Vector<Real>& qnds, const Vector<Real>& qwts, const Integer max_depth) {
    // 1D analogue of the near-interaction quadtree: split [0,1] in direction `dir`,
    // grading toward `center`, until each segment is admissible for the target.
    const Integer QuadOrder = qnds.Dim();
    constexpr Long MaxLeaves = 4096;

    struct Seg { Real a0, a1; Integer depth; };
    std::vector<Seg> stack, leaves;
    stack.push_back({0, 1, 0});

    Vector<Real> ps(1), cs(1), Xc, dXdu, dXdv;
    cs[0] = cross;
    while (!stack.empty()) {
      const Seg s = stack.back(); stack.pop_back();

      // Distance from the target to the segment: clamp `center` into [a0,a1].
      ps[0] = std::min<Real>(s.a1, std::max<Real>(s.a0, center));
      if (dir == 0) qel.GetGeom(&Xc, nullptr, nullptr, nullptr, nullptr, ps, cs, elem_idx);
      else          qel.GetGeom(&Xc, nullptr, nullptr, nullptr, nullptr, cs, ps, elem_idx);
      Real dist2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { const Real d = Xc[k]-Xtrg[k]; dist2 += d*d; }
      const Real dist = sqrt<Real>(dist2);

      // Physical segment extent from the surface speed at the segment center.
      ps[0] = (s.a0+s.a1)/2;
      if (dir == 0) qel.GetGeom(nullptr, nullptr, nullptr, &dXdu, &dXdv, ps, cs, elem_idx);
      else          qel.GetGeom(nullptr, nullptr, nullptr, &dXdu, &dXdv, cs, ps, elem_idx);
      const Vector<Real>& dX = (dir == 0 ? dXdu : dXdv);
      Real sp2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) sp2 += dX[k]*dX[k];
      const Real L = sqrt<Real>(sp2)*(s.a1-s.a0);

      // Admissible (leaf) when the target is outside the Bernstein neighborhood; a
      // non-finite dist forces a leaf so a degenerate metric cannot loop to depth.
      if (!(dist < b_ellipse*L) || s.depth >= max_depth) {
        leaves.push_back(s);
        SCTL_ASSERT((Long)leaves.size() <= MaxLeaves);
      } else {
        const Real am = (s.a0+s.a1)/2;
        stack.push_back({s.a0, am, s.depth+1});
        stack.push_back({am, s.a1, s.depth+1});
      }
    }

    const Long N = (Long)leaves.size() * QuadOrder;
    param.ReInit(N);
    w.ReInit(N);
    Long idx = 0;
    for (const Seg& s : leaves) {
      const Real len = s.a1 - s.a0;
      for (Integer a = 0; a < QuadOrder; a++) {
        param[idx] = s.a0 + len*qnds[a];
        w[idx] = qwts[a]*len;
        idx++;
      }
    }
  }

  template <class Real> void QuadElemList<Real>::LogSingularQuad1D(Vector<Real>& param, Vector<Real>& w, const Real v0, const Integer order) {
    // Alpert hybrid Gauss-trapezoidal quadrature on [0,1] for an integrand with a
    // log singularity at the interior point v0. The interval is split at v0 into
    // [0,v0] and [v0,1]; on each sub-interval a uniform trapezoidal grid is corrected
    // with Alpert's log endpoint rule at the v0 side and a smooth endpoint rule at
    // the outer side (0 or 1). Endpoint-correction tables come from alpert_quadr.cpp.

    // Snap to a supported Alpert log order (cf. QuadLogExtraPtNodes).
    // static const int log_orders[] = {2, 3, 4, 5, 6, 8, 10, 12, 14, 16};
    // int ord = log_orders[0];
    // for (int o : log_orders) { ord = o; if (o >= order) break; }
    const int ord = 16;
    // TODO: carry tolerance into this and correspond to log_orders; 
    // TODO: add upsampling before and downsample after for more accurate GL-Alpert interpolation?

    std::vector<double> px, pw;
    // TODO: in distributed memory, these should be preallocated and access limited

    // Assemble the corrected rule on [a,b]; corr == 2 -> log endpoint, else smooth.
    auto add_interval = [&](double a, double b, int corra, int corrb) {
      const ExtraPtResult L = (corra == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const ExtraPtResult R = (corrb == 2 ? QuadLogExtraPtNodes((double)ord) : QuadSmoothExtraPtNodes((double)ord));
      const int skipL = L.NodesToSkip, skipR = R.NodesToSkip;

      // Uniform grid with ~2*ord intervals, but enough to host both corrections.
      const int N = std::max(skipL + skipR + 2, 2 * ord);
      const int N1 = N - 1;
      const double h = (b - a) / N1;

      // Regular trapezoidal nodes, dropping the skipped nodes nearest each endpoint
      // (both endpoints are corrected here, so every kept node has full weight h).
      for (int i = skipL; i <= N1 - skipR; ++i) {
        px.push_back(a + i * h);
        pw.push_back(h);
      }
      // Left endpoint correction nodes (measured from a).
      for (size_t i = 0; i < L.ExtraNodes.size(); ++i) {
        px.push_back(a + L.ExtraNodes[i] * h);
        pw.push_back(L.ExtraWeights[i] * h);
      }
      // Right endpoint correction nodes (measured from b).
      for (size_t i = 0; i < R.ExtraNodes.size(); ++i) {
        px.push_back(b - R.ExtraNodes[i] * h);
        pw.push_back(R.ExtraWeights[i] * h);
      }
    };

    add_interval(0.0, (double)v0, /*smooth*/ 1, /*log*/ 2);
    add_interval((double)v0, 1.0, /*log*/ 2, /*smooth*/ 1);

    const Long N = (Long)px.size();
    param.ReInit(N);
    w.ReInit(N);
    for (Long i = 0; i < N; ++i) { param[i] = (Real)px[i]; w[i] = (Real)pw[i]; }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Real tol) {
    // Adaptive 2D quadtree for an off-surface target: refine [0,1]^2 into a graded
    // set of leaf panels, then integrate each leaf with a QuadOrder GL rule via
    // IntegrateBlock. M_acc (nnode x KDIM0*KDIM1_out) is sized/zeroed here.

    // TODO: only binary split right now, see CSBQ for changing step-size.

    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Integer order = qel.order;
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    Real b_ellipse; Integer QuadOrder;
    QuadParams(tol, b_ellipse, QuadOrder);
    const Vector<Real>& qnds = ParamNodes(QuadOrder);
    const Vector<Real>& qwts = LegQuadRule<Real>::wts(QuadOrder);

    constexpr Integer max_depth = 20;
    constexpr Long MaxLeaves = 4096;

    // Closest point (u*,v*) on the element seeds the grading.
    Real ustar, vstar;
    qel.GetClosestNode(ustar, vstar, nullptr, nullptr, elem_idx, Xtrg);

    struct Panel { Real u0, u1, v0, v1; Integer depth; };
    std::vector<Panel> stack, leaves;
    stack.push_back({0, 1, 0, 1, 0});
    Vector<Real> us(1), vs(1), Xc, dXdu, dXdv;
    while (!stack.empty()) {
      const Panel p = stack.back(); stack.pop_back();

      // Distance from target to the panel: clamp (u*,v*) into the rectangle.
      us[0] = std::min<Real>(p.u1, std::max<Real>(p.u0, ustar));
      vs[0] = std::min<Real>(p.v1, std::max<Real>(p.v0, vstar));
      qel.GetGeom(&Xc, nullptr, nullptr, nullptr, nullptr, us, vs, elem_idx);
      Real dist2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { const Real d = Xc[k]-Xtrg[k]; dist2 += d*d; }
      const Real dist = sqrt<Real>(dist2);

      // Physical panel extents from the surface speeds at the panel center.
      us[0] = (p.u0+p.u1)/2; vs[0] = (p.v0+p.v1)/2;
      qel.GetGeom(nullptr, nullptr, nullptr, &dXdu, &dXdv, us, vs, elem_idx);
      Real su2 = 0, sv2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) { su2 += dXdu[k]*dXdu[k]; sv2 += dXdv[k]*dXdv[k]; }
      const Real Lu = sqrt<Real>(su2)*(p.u1-p.u0);
      const Real Lv = sqrt<Real>(sv2)*(p.v1-p.v0);

      // Admissible (leaf) when the target is outside the Bernstein neighborhood; a
      // non-finite dist forces a leaf so a degenerate metric cannot loop to depth.
      if (!(dist < b_ellipse*std::max<Real>(Lu, Lv)) || p.depth >= max_depth) {
        leaves.push_back(p);
        SCTL_ASSERT((Long)leaves.size() <= MaxLeaves);
      } else {
        const Real um = (p.u0+p.u1)/2, vm = (p.v0+p.v1)/2;
        stack.push_back({p.u0, um, p.v0, vm, p.depth+1});
        stack.push_back({um, p.u1, p.v0, vm, p.depth+1});
        stack.push_back({p.u0, um, vm, p.v1, p.depth+1});
        stack.push_back({um, p.u1, vm, p.v1, p.depth+1});
      }
    }

    // TODO: if distributive memory, will need to grab only relevant entries of M_acc.
    if (M_acc.Dim(0) != nnode || M_acc.Dim(1) != KDIM0*KDIM1_out) {
      M_acc.ReInit(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();
    }

    Vector<Real> u_param(QuadOrder), v_param(QuadOrder), wu(QuadOrder), wv(QuadOrder);
    for (const Panel& p : leaves) {
      const Real du = p.u1-p.u0, dv = p.v1-p.v0;
      for (Integer a = 0; a < QuadOrder; a++) { u_param[a] = p.u0 + du*qnds[a]; wu[a] = qwts[a]*du; }
      for (Integer b = 0; b < QuadOrder; b++) { v_param[b] = p.v0 + dv*qnds[b]; wv[b] = qwts[b]*dv; }
      IntegrateBlock(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker);
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInteracBlock(Matrix<Real>& M_acc, const QuadElemList<Real>& qel, const Long elem_idx, const Real u0, const Real v0, const Vector<Real>& Xtrg, const Vector<Real>& normal_trg, const Kernel& ker, const Real tol) {
    // Singular self-interaction for an on-surface target at parameter (u0,v0).
    // 1D reduction: adaptive panel refinement in direction A (u, shared across all
    // B nodes) toward u0 + a 1D log-singular quadrature in direction B (v) toward
    // v0. The tensor product of the two 1D rules is integrated by IntegrateBlock.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();
    const Integer order = qel.order;
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    Real b_ellipse; Integer QuadOrder;
    QuadParams(tol, b_ellipse, QuadOrder);
    const Vector<Real>& qnds = ParamNodes(QuadOrder);
    const Vector<Real>& qwts = LegQuadRule<Real>::wts(QuadOrder);
    constexpr Integer max_depth = 20;

    // Direction A (u): one shared graded refinement toward u0 along the line v=v0.
    Vector<Real> u_param, wu;
    BuildGraded1D(u_param, wu, qel, elem_idx, u0, v0, /*dir=*/0, b_ellipse, Xtrg, qnds, qwts, max_depth);

    // Direction B (v): Alpert 1D log-singular rule at v0, with correction order
    // matched to the per-panel GL quadrature order.
    Vector<Real> v_param, wv;
    LogSingularQuad1D(v_param, wv, v0, QuadOrder);

    M_acc.ReInit(nnode, KDIM0*KDIM1_out);
    M_acc.SetZero();
    IntegrateBlock(M_acc, qel, elem_idx, Xtrg, normal_trg, u_param, wu, v_param, wv, ker);
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // On-surface (singular) self-interaction. For each element, every surface
    // discretization node is a target lying exactly on the element; the singular
    // block (1D-reduction scheme) builds its contribution. The resulting matrix
    // M_lst[e] has shape (nnode*KDIM0) x (nnode*KDIM1_out) and is applied as
    // U = F * M_lst[e] by the boundary-integral driver.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();

    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    const Integer order = qel.order;
    const Long nnode = (Long)order * order;
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;
    const Vector<Real>& nds = ParamNodes(order);

    SCTL_ASSERT((Long)M_lst.Dim() == qel.nelem);

    for (Long elem_idx = 0; elem_idx < qel.nelem; elem_idx++) {
      // Surface nodes (targets) and their normals on this element.
      Vector<Real> Xnodes, Xnnodes;
      qel.GetGeom(&Xnodes, (trg_dot_prod ? &Xnnodes : nullptr), nullptr, nullptr, nullptr, nds, nds, elem_idx);

      Matrix<Real>& M = M_lst[elem_idx];
      if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != nnode*KDIM1_out) M.ReInit(nnode*KDIM0, nnode*KDIM1_out);
      M.SetZero();

      for (Integer ti = 0; ti < order; ti++) {
        for (Integer tj = 0; tj < order; tj++) {
          const Long t = ti*order + tj; // target node index = column block
          const Real u0 = nds[ti], v0 = nds[tj];

          Vector<Real> Xtrg(COORD_DIM, Xnodes.begin() + t*COORD_DIM, false);
          Vector<Real> ntrg;
          if (trg_dot_prod) ntrg.ReInit(COORD_DIM, Xnnodes.begin() + t*COORD_DIM, false);

          Matrix<Real> M_acc;
          SelfInteracBlock(M_acc, qel, elem_idx, u0, v0, Xtrg, ntrg, ker, tol);

          // Scatter into column block t of M: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1].
          for (Integer i = 0; i < order; i++) {
            for (Integer j = 0; j < order; j++) {
              const Long pnode = i*order + j;
              for (Integer k0 = 0; k0 < KDIM0; k0++) {
                for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
                  M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
                }
              }
            }
          }
        }
      }
    }
  }

  template <class Real> Real QuadElemList<Real>::GetClosestNode(Real& ustar, Real& vstar, Vector<Real>* Xstar, Vector<Real>* Nstar, const Long elem_idx, const Vector<Real>& Xtrg) const {
    const auto& nds = ParamNodes(order);
    const Long nnode = (Long)order * order;

    // Brute-force seed over the order x order nodal grid.
    // TODO: binary search among each direction.
    Vector<Real> Xnodes, Xnnodes;
    GetGeom(&Xnodes, &Xnnodes, nullptr, nullptr, nullptr, nds, nds, elem_idx);
    Long seed = 0;
    Real best = -1;
    StaticArray<Real,COORD_DIM> bbox_min, bbox_max;
    for (Integer k = 0; k < COORD_DIM; k++) { bbox_min[k] = Xnodes[k]; bbox_max[k] = Xnodes[k]; }
    for (Long p = 0; p < nnode; p++) {
      Real r2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real x = Xnodes[p*COORD_DIM+k];
        const Real d = x - Xtrg[k];
        r2 += d*d;
        bbox_min[k] = std::min<Real>(bbox_min[k], x);
        bbox_max[k] = std::max<Real>(bbox_max[k], x);
      }
      if (best < 0 || r2 < best) { best = r2; seed = p; }
    }
    ustar = nds[seed/order];
    vstar = nds[seed%order];

    if (Xstar) { Xstar->ReInit(COORD_DIM); for (Integer k = 0; k < COORD_DIM; k++) (*Xstar)[k] = Xnodes[seed*COORD_DIM+k]; }
    if (Nstar) { Nstar->ReInit(COORD_DIM); for (Integer k = 0; k < COORD_DIM; k++) (*Nstar)[k] = Xnnodes[seed*COORD_DIM+k]; }
    return sqrt<Real>(best);

  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    // Per-target near-singular interaction. An off-surface target is integrated by
    // the adaptive 2D quadtree (NearInteracBlock). A target that lands on the
    // source element (a shared node/edge, dist ~ 0) is routed to the singular self
    // block (SelfInteracBlock) so the kernel is never evaluated at r=0; on-surface
    // singular self interactions proper are built by SelfInterac.
    static constexpr Integer KDIM0 = Kernel::SrcDim();
    static constexpr Integer KDIM1full = Kernel::TrgDim();

    const QuadElemList<Real>& qel = *static_cast<const QuadElemList<Real>*>(self);
    const Integer order = qel.order;
    const Long nnode = (Long)order * order;
    const bool trg_dot_prod = (normal_trg.Dim() > 0);
    const Integer KDIM1_out = trg_dot_prod ? KDIM1full / COORD_DIM : KDIM1full;

    const Long Ntrg = Xt.Dim() / COORD_DIM;
    if (M.Dim(0) != nnode*KDIM0 || M.Dim(1) != Ntrg*KDIM1_out) {
      M.ReInit(nnode*KDIM0, Ntrg*KDIM1_out);
    }
    M.SetZero();
    if (!Ntrg) return;

    for (Long t = 0; t < Ntrg; t++) {
      Vector<Real> Xtrg(COORD_DIM, (Iterator<Real>)Xt.begin() + t*COORD_DIM, false);
      Vector<Real> ntrg;
      if (trg_dot_prod) ntrg.ReInit(COORD_DIM, (Iterator<Real>)normal_trg.begin() + t*COORD_DIM, false);

      // Closest point detects on-surface targets and routes them to the singular block.
      // Real ustar, vstar;
      // const Real dist = qel.GetClosestNode(ustar, vstar, nullptr, nullptr, elem_idx, Xtrg);

      Matrix<Real> M_acc;
      NearInteracBlock(M_acc, qel, elem_idx, Xtrg, ntrg, ker, tol);

      // Scatter into M for target t: M[(i*order+j)*KDIM0+k0][t*KDIM1_out+k1].
      for (Integer i = 0; i < order; i++) {
        for (Integer j = 0; j < order; j++) {
          const Long pnode = i*order + j;
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              M[pnode*KDIM0+k0][t*KDIM1_out+k1] = M_acc[pnode][k0*KDIM1_out+k1];
            }
          }
        }
      }
    }
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamNodes(const Integer Order) {
    return LegQuadRule<Real>::nds(Order);
  }

  template <class Real> const Vector<Real>& QuadElemList<Real>::ParamGrid(const Integer Order, const Integer Nelem_perside) {
    const Vector<Real> nodes = ParamNodes(Order);

    Vector<Real> x_param(Order * Nelem_perside);
    for (int pind=0; pind < Nelem_perside; pind ++) {
        for (int nind=0; nind < Order; nind ++) {
            x_param[pind * Order + nind] = (nodes[nind] + pind) / Nelem_perside; // TODO check
        }
    }
    static Vector<Real> coord0(x_param.Dim() * x_param.Dim() * COORD_DIM);
    for (int xind=0; xind < x_param.Dim(); xind ++) {
        for (int yind=0; yind < x_param.Dim(); yind ++) {
            const Long idx = xind * x_param.Dim() * COORD_DIM + yind * COORD_DIM;
            coord0[idx + 0] = x_param[xind];
            coord0[idx + 1] = x_param[yind];
            coord0[idx + 2] = 0.; 
        }
    }
    return coord0;
  }

  template <class Real> void QuadElemList<Real>::Write(const std::string& fname, const Comm& comm) const {
    auto allgather = [&comm](Vector<Real>& v_out, const Vector<Real>& v_in) {
      const Long Nproc = comm.Size();
      StaticArray<Long,1> len{v_in.Dim()};
      Vector<Long> cnt(Nproc), dsp(Nproc);
      comm.Allgather(len + 0, 1, cnt.begin(), 1);
      dsp = 0;
      omp_par::scan(cnt.begin(), dsp.begin(), Nproc);

      v_out.ReInit(dsp[Nproc-1] + cnt[Nproc-1]);
      comm.Allgatherv(v_in.begin(), v_in.Dim(), v_out.begin(), cnt.begin(), dsp.begin());
    };

    Vector<Real> coord_;
    allgather(coord_, coord);

    const Long nnode_per_elem = (Long)order * order;
    const Long Nelem_total = coord_.Dim() / (COORD_DIM * nnode_per_elem);
    SCTL_ASSERT(coord_.Dim() == Nelem_total * COORD_DIM * nnode_per_elem);

    if (comm.Rank()) return;

    const Integer precision = (Integer)std::ceil(-std::log((double)machine_eps<Real>()) / std::log(10.0));
    const Integer width = precision + 8;
    std::ofstream file(fname, std::ofstream::out | std::ofstream::trunc);
    SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for writing: ") + fname);

    file << "#";
    file << std::setw(width - 1) << "X";
    file << std::setw(width) << "Y";
    file << std::setw(width) << "Z";
    file << std::setw(width) << "ElemOrder";
    file << '\n';

    file << std::scientific << std::setprecision(precision);
    for (Long elem_idx = 0; elem_idx < Nelem_total; elem_idx++) {
      const Long base = elem_idx * COORD_DIM * nnode_per_elem;
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Integer k = 0; k < COORD_DIM; k++) {
          file << std::setw(width) << coord_[base + k * nnode_per_elem + p];
        }
        if (!p) file << std::setw(width) << order;
        file << '\n';
      }
    }
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Read(const std::string& fname, const Comm& comm) {
    std::ifstream file(fname, std::ifstream::in);
    SCTL_ASSERT_MSG(file.good(), std::string("Unable to open file for reading: ") + fname);

    std::string line;
    Vector<ValueType> coord_;
    Vector<Long> order_markers;
    while (std::getline(file, line)) {
      const size_t first_char_pos = line.find_first_not_of(' ');
      if (first_char_pos == std::string::npos || line[first_char_pos] == '#') continue;

      std::istringstream iss(line);
      for (Integer k = 0; k < COORD_DIM; k++) {
        ValueType a;
        iss >> a;
        SCTL_ASSERT(!iss.fail());
        coord_.PushBack(a);
      }

      Integer order_;
      if (iss >> order_) {
        order_markers.PushBack(order_);
      } else {
        order_markers.PushBack(-1);
      }
    }
    file.close();

    // Determine order from the first element marker and verify uniformity.
    SCTL_ASSERT(order_markers.Dim() > 0);
    const Integer file_order = order_markers[0];
    SCTL_ASSERT(file_order > 0);
    const Long nnode_per_elem = (Long)file_order * file_order;

    SCTL_ASSERT(order_markers.Dim() % nnode_per_elem == 0);
    const Long Nelem_total = order_markers.Dim() / nnode_per_elem;
    for (Long elem = 0; elem < Nelem_total; elem++) {
      const Long offset = elem * nnode_per_elem;
      SCTL_ASSERT(order_markers[offset] == file_order);
      for (Long j = 1; j < nnode_per_elem; j++) {
        SCTL_ASSERT(order_markers[offset + j] == file_order || order_markers[offset + j] == -1);
      }
    }

    {
      const Long Np = comm.Size();
      const Long pid = comm.Rank();

      const Long i0 = Nelem_total * (pid + 0) / Np;
      const Long i1 = Nelem_total * (pid + 1) / Np;

      const Long j0 = i0 * nnode_per_elem;
      const Long j1 = i1 * nnode_per_elem;

      Vector<ValueType> coord_local;
      coord_local.ReInit((j1 - j0) * COORD_DIM, coord_.begin() + j0 * COORD_DIM, false);
      Init<ValueType>(file_order, coord_local);
    }
  }

  template <class Real> void QuadElemList<Real>::GetVTUData(VTUData& vtu_data, const Vector<Real>& F, const Long elem_idx) const {
    if (elem_idx == -1) {
      const Long nnode_per_elem = (Long)order * order;
      Long dof = 0;
      Long offset = 0;
      if (F.Dim()) {
        const Long Nnode = nelem * nnode_per_elem;
        dof = (Nnode ? F.Dim() / Nnode : 0);
        SCTL_ASSERT(F.Dim() == Nnode * dof);
      }
      for (Long i = 0; i < nelem; i++) {
        const Vector<Real> F_(nnode_per_elem * dof, (Iterator<Real>)F.begin() + offset, false);
        GetVTUData(vtu_data, F_, i);
        offset += F_.Dim();
      }
      return;
    }

    Vector<Real> u_nodes(order + 2), v_nodes(order + 2);
    u_nodes[0] = 0;
    v_nodes[0] = 0;
    u_nodes[order + 1] = 1;
    v_nodes[order + 1] = 1;
    Vector<Real>(order, u_nodes.begin() + 1, false) = ParamNodes(order);
    Vector<Real>(order, v_nodes.begin() + 1, false) = ParamNodes(order);

    Vector<Real> X;
    GetGeom(&X, nullptr, nullptr, nullptr, nullptr, u_nodes, v_nodes, elem_idx);

    const Long Nu = u_nodes.Dim();
    const Long Nv = v_nodes.Dim();
    Vector<Real> Fgrid;
    if (F.Dim()) {
      const Long nnode_per_elem = (Long)order * order;
      const Long dof = F.Dim() / nnode_per_elem;
      SCTL_ASSERT(F.Dim() == nnode_per_elem * dof);

      Vector<Real> F_soa(dof * nnode_per_elem);
      for (Long p = 0; p < nnode_per_elem; p++) {
        for (Long k = 0; k < dof; k++) {
          F_soa[k * nnode_per_elem + p] = F[p * dof + k];
        }
      }

      Matrix<Real> MuT(order, Nu), Mv(order, Nv);
      Vector<Real> Mu_(order * Nu, MuT.begin(), false);
      Vector<Real> Mv_(order * Nv, Mv.begin(), false);
      LagrangeInterp<Real>::Interpolate(Mu_, ParamNodes(order), u_nodes);
      LagrangeInterp<Real>::Interpolate(Mv_, ParamNodes(order), v_nodes);
      MuT = MuT.Transpose();

      Vector<Real> F_soa_eval;
      EvalTensorProduct(F_soa_eval, F_soa, MuT, Mv);

      Fgrid.ReInit(Nu * Nv * dof);
      for (Long p = 0; p < Nu * Nv; p++) {
        for (Long k = 0; k < dof; k++) {
          Fgrid[p * dof + k] = F_soa_eval[k * (Nu * Nv) + p];
        }
      }
    }

    const Long point_offset = vtu_data.coord.Dim() / COORD_DIM;
    for (const auto& x : X) vtu_data.coord.PushBack((VTUData::VTKReal)x);
    for (const auto& f : Fgrid) vtu_data.value.PushBack((VTUData::VTKReal)f);

    for (Long i = 0; i < Nu - 1; i++) {
      for (Long j = 0; j < Nv - 1; j++) {
        const Long idx = point_offset + i * Nv + j;
        vtu_data.connect.PushBack(idx);
        vtu_data.connect.PushBack(idx + 1);
        vtu_data.connect.PushBack(idx + Nv + 1);
        vtu_data.connect.PushBack(idx + Nv);
        vtu_data.offset.PushBack(vtu_data.connect.Dim());
        vtu_data.types.PushBack(9);
      }
    }
  }

  template <class Real> void QuadElemList<Real>::WriteVTK(const std::string& fname, const Vector<Real>& F, const Comm& comm) const {
    VTUData vtu_data;
    GetVTUData(vtu_data, F);
    vtu_data.WriteVTK(fname, comm);
  }

  template <class Real> template <class ValueType> void QuadElemList<Real>::Copy(QuadElemList<ValueType>& elem_lst) const {
    elem_lst.nelem = nelem;
    elem_lst.order = order;

    elem_lst.coord.ReInit(coord.Dim());
    elem_lst.dcoord_du.ReInit(dcoord_du.Dim());
    elem_lst.dcoord_dv.ReInit(dcoord_dv.Dim());
    for (Long i = 0; i < coord.Dim(); i++) elem_lst.coord[i] = (ValueType)coord[i];
    for (Long i = 0; i < dcoord_du.Dim(); i++) elem_lst.dcoord_du[i] = (ValueType)dcoord_du[i];
    for (Long i = 0; i < dcoord_dv.Dim(); i++) elem_lst.dcoord_dv[i] = (ValueType)dcoord_dv[i];
  }

}

#endif // _SCTL_QUAD_ELEMENT_CPP_
