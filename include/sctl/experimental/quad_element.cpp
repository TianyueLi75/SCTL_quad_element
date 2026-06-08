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

  // template <class Real> void QuadElemList<Real>::Upsample(Vector<Real>& Xout, Vector<Real>& Xnout, Vector<Real>& Fout, Vector<Real>& Wout, const Vector<Real>& Xin, const Vector<Real>& Fin, const Long& nsub_perside) {
  //   // Given Xin (the order x order Gauss-Legendre nodal coordinates of one or
  //   // more patches), subdivide each patch into nsub_perside x
  //   // nsub_perside sub-panels, each carrying order x order GL nodes. The
  //   // patch geometry (Xin) and field (Fin) are interpolated onto the refined
  //   // nodes; Wout returns the surface-quadrature weights at those nodes.
  //   //
  //   // Layout: Xin/Xout/Fin/Fout are AoS, ordered patch-by-patch, with the
  //   // order x order nodes of each patch in (u,v) tensor order (u slow, v fast).
  //   // Output sub-panels for input patch e are ordered (pi,pj) with pi slow.
  //   SCTL_ASSERT(nsub_perside > 0);

  //   const Long nnode_per_elem = (Long)order * order;
  //   SCTL_ASSERT(Xin.Dim() % (nnode_per_elem * COORD_DIM) == 0);
  //   const Long nelem_in = Xin.Dim() / (nnode_per_elem * COORD_DIM);

  //   Long dof = 0;
  //   if (Fin.Dim()) {
  //     SCTL_ASSERT(Fin.Dim() % (nelem_in * nnode_per_elem) == 0);
  //     dof = Fin.Dim() / (nelem_in * nnode_per_elem);
  //   }

  //   const Long nsub = (Long)nsub_perside * nsub_perside;
  //   const Long nelem_out = nelem_in * nsub;
  //   const Long Nnode_out = nelem_out * nnode_per_elem;

  //   if (Xout.Dim() != Nnode_out * COORD_DIM) Xout.ReInit(Nnode_out * COORD_DIM);
  //   if (Wout.Dim() != Nnode_out) Wout.ReInit(Nnode_out);
  //   if (Fout.Dim() != Nnode_out * dof) Fout.ReInit(Nnode_out * dof);

  //   const auto& nodes = ParamNodes(order);
  //   const auto& node_wts = LegQuadRule<Real>::wts(order);

  //   // Parameter values (in the patch reference square [0,1]^2) of the refined
  //   // tensor grid, laid out panel-by-panel: sub_param[pind*order + a] places GL
  //   // node a inside sub-panel pind, which spans [pind, pind+1]/nsub_perside.
  //   const Long M = (Long)nsub_perside * order;
  //   Vector<Real> sub_param(M);
  //   for (Integer pind = 0; pind < nsub_perside; pind++) {
  //     for (Integer a = 0; a < order; a++) {
  //       sub_param[pind * order + a] = (nodes[a] + pind) / nsub_perside;
  //     }
  //   }

  //   // Interpolation matrices from the patch nodes to the refined tensor grid
  //   // (same convention as GetGeom: MuT is (M x order), Mv is (order x M)).
  //   Matrix<Real> MuT(order, M), Mv(order, M);
  //   {
  //     Vector<Real> Mu_(order * M, MuT.begin(), false);
  //     Vector<Real> Mv_(order * M, Mv.begin(), false);
  //     LagrangeInterp<Real>::Interpolate(Mu_, nodes, sub_param);
  //     LagrangeInterp<Real>::Interpolate(Mv_, nodes, sub_param);
  //     MuT = MuT.Transpose();
  //   }

  //   const Real inv_nps = (Real)1 / (Real)nsub_perside;
  //   const Long Mgrid = M * M;

  //   for (Long e = 0; e < nelem_in; e++) {
  //     // Gather this patch's coordinates in SoA layout for EvalTensorProduct.
  //     Vector<Real> coord_soa(COORD_DIM * nnode_per_elem);
  //     for (Integer k = 0; k < COORD_DIM; k++) {
  //       for (Long p = 0; p < nnode_per_elem; p++) {
  //         coord_soa[k * nnode_per_elem + p] = Xin[(e * nnode_per_elem + p) * COORD_DIM + k];
  //       }
  //     }

  //     // Nodal parameter-derivatives of the patch geometry (cf. BuildDerivativeCache).
  //     Vector<Real> dcoord_du_soa(COORD_DIM * nnode_per_elem);
  //     Vector<Real> dcoord_dv_soa(COORD_DIM * nnode_per_elem);
  //     {
  //       Vector<Real> line_in(order), line_out(order);
  //       for (Integer k = 0; k < COORD_DIM; k++) {
  //         const Long cb = k * nnode_per_elem;
  //         for (Integer j = 0; j < order; j++) {
  //           for (Integer i = 0; i < order; i++) line_in[i] = coord_soa[cb + i * order + j];
  //           LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
  //           for (Integer i = 0; i < order; i++) dcoord_du_soa[cb + i * order + j] = line_out[i];
  //         }
  //         for (Integer i = 0; i < order; i++) {
  //           for (Integer j = 0; j < order; j++) line_in[j] = coord_soa[cb + i * order + j];
  //           LagrangeInterp<Real>::Derivative(line_out, line_in, nodes);
  //           for (Integer j = 0; j < order; j++) dcoord_dv_soa[cb + i * order + j] = line_out[j];
  //         }
  //       }
  //     }

  //     // Interpolate coordinates and parameter-derivatives onto the refined grid.
  //     Vector<Real> X_soa, dXdu_soa, dXdv_soa;
  //     EvalTensorProduct(X_soa, coord_soa, MuT, Mv);
  //     EvalTensorProduct(dXdu_soa, dcoord_du_soa, MuT, Mv);
  //     EvalTensorProduct(dXdv_soa, dcoord_dv_soa, MuT, Mv);

  //     // Interpolate the field onto the refined grid.
  //     Vector<Real> F_soa_eval;
  //     if (dof) {
  //       Vector<Real> F_soa(dof * nnode_per_elem);
  //       for (Long p = 0; p < nnode_per_elem; p++) {
  //         for (Long c = 0; c < dof; c++) {
  //           F_soa[c * nnode_per_elem + p] = Fin[(e * nnode_per_elem + p) * dof + c];
  //         }
  //       }
  //       EvalTensorProduct(F_soa_eval, F_soa, MuT, Mv);
  //     }

  //     // Scatter the refined grid into the per-sub-panel output elements.
  //     for (Integer pi = 0; pi < nsub_perside; pi++) {
  //       for (Integer pj = 0; pj < nsub_perside; pj++) {
  //         const Long out_elem = e * nsub + (pi * nsub_perside + pj);
  //         for (Integer a = 0; a < order; a++) {
  //           for (Integer b = 0; b < order; b++) {
  //             const Long iu = (Long)pi * order + a;
  //             const Long iv = (Long)pj * order + b;
  //             const Long q = iu * M + iv;              // index into refined SoA grid
  //             const Long p = (Long)a * order + b;      // node index within sub-panel
  //             const Long out_node = out_elem * nnode_per_elem + p;

  //             for (Integer k = 0; k < COORD_DIM; k++) {
  //               Xout[out_node * COORD_DIM + k] = X_soa[k * Mgrid + q];
  //             }
  //             for (Long c = 0; c < dof; c++) {
  //               Fout[out_node * dof + c] = F_soa_eval[c * Mgrid + q];
  //             }

  //             // Surface element |dX/du x dX/dv| at the refined node.
  //             const Real du0 = dXdu_soa[0 * Mgrid + q];
  //             const Real du1 = dXdu_soa[1 * Mgrid + q];
  //             const Real du2 = dXdu_soa[2 * Mgrid + q];
  //             const Real dv0 = dXdv_soa[0 * Mgrid + q];
  //             const Real dv1 = dXdv_soa[1 * Mgrid + q];
  //             const Real dv2 = dXdv_soa[2 * Mgrid + q];
  //             const Real n0 = du1 * dv2 - du2 * dv1;
  //             const Real n1 = du2 * dv0 - du0 * dv2;
  //             const Real n2 = du0 * dv1 - du1 * dv0;
  //             const Real area = sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
  //             const Real inv_area = (area > 0 ? 1 / area : 0);

  //             Xnout[out_node * COORD_DIM + 0] = n0 * inv_area;
  //             Xnout[out_node * COORD_DIM + 1] = n1 * inv_area;
  //             Xnout[out_node * COORD_DIM + 2] = n2 * inv_area;
          

  //             // GL weights are for [0,1]; each sub-panel spans 1/nsub_perside
  //             // of the patch in each direction, hence the inv_nps^2 Jacobian.
  //             Wout[out_node] = area * node_wts[a] * node_wts[b] * inv_nps * inv_nps;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  
  template <class Real> template <class Kernel> void QuadElemList<Real>::SelfInterac(Vector<Matrix<Real>>& M_lst, const Kernel& ker, Real tol, bool trg_dot_prod, const ElementListBase<Real>* self) {
    // TODO: implement singular self-interaction quadrature for QuadElemList
    SCTL_ASSERT(false);
  }

  template <class Real> Real QuadElemList<Real>::GetClosestPoint(Real& ustar, Real& vstar, Vector<Real>* Xstar, Vector<Real>* Nstar, const Long elem_idx, const Vector<Real>& Xtrg) const {
    const auto& nds = ParamNodes(order);
    const Long nnode = (Long)order * order;

    // Brute-force seed over the order x order nodal grid.
    Vector<Real> Xnodes;
    GetGeom(&Xnodes, nullptr, nullptr, nullptr, nullptr, nds, nds, elem_idx);
    Long seed = 0;
    Real best = -1;
    for (Long p = 0; p < nnode; p++) {
      Real r2 = 0;
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real d = Xnodes[p*COORD_DIM+k] - Xtrg[k];
        r2 += d*d;
      }
      if (best < 0 || r2 < best) { best = r2; seed = p; }
    }
    ustar = nds[seed/order];
    vstar = nds[seed%order];

    // Gauss-Newton closest-point iteration: minimize |X(u,v) - Xtrg|^2.
    Vector<Real> up(1), vp(1), X, dXdu, dXdv;
    for (Integer iter = 0; iter < 30; iter++) {
      up[0] = ustar; vp[0] = vstar;
      GetGeom(&X, nullptr, nullptr, &dXdu, &dXdv, up, vp, elem_idx);
      Real a = 0, b = 0, c = 0, g0 = 0, g1 = 0; // 2x2 normal equations
      for (Integer k = 0; k < COORD_DIM; k++) {
        const Real r = Xtrg[k] - X[k];
        a += dXdu[k]*dXdu[k];
        b += dXdu[k]*dXdv[k];
        c += dXdv[k]*dXdv[k];
        g0 += dXdu[k]*r;
        g1 += dXdv[k]*r;
      }
      const Real det = a*c - b*b;
      if (fabs(det) < machine_eps<Real>()) break;
      const Real du = ( c*g0 - b*g1) / det;
      const Real dv = (-b*g0 + a*g1) / det;
      const Real un = std::min<Real>(1, std::max<Real>(0, ustar + du));
      const Real vn = std::min<Real>(1, std::max<Real>(0, vstar + dv));
      const Real step = fabs(un - ustar) + fabs(vn - vstar);
      ustar = un; vstar = vn;
      if (step < machine_eps<Real>()*8) break;
    }

    // Final evaluation for the distance and (optional) point/normal.
    up[0] = ustar; vp[0] = vstar;
    Vector<Real> Xf, Nf;
    GetGeom(&Xf, (Nstar ? &Nf : nullptr), nullptr, nullptr, nullptr, up, vp, elem_idx);
    Real d2 = 0;
    for (Integer k = 0; k < COORD_DIM; k++) {
      const Real d = Xf[k] - Xtrg[k];
      d2 += d*d;
    }
    if (Xstar) { Xstar->ReInit(COORD_DIM); for (Integer k = 0; k < COORD_DIM; k++) (*Xstar)[k] = Xf[k]; }
    if (Nstar) { Nstar->ReInit(COORD_DIM); for (Integer k = 0; k < COORD_DIM; k++) (*Nstar)[k] = Nf[k]; }
    return sqrt<Real>(d2);
  }

  template <class Real> void QuadElemList<Real>::GetNearestNode(Vector<Real>& Ystar, Vector<Real>& Ynstar, Long& Y_elem_idx, const Vector<Real>& Xtrg) {
    // Finds the closest surface point Ystar (over all elements) to the target
    // Xtrg, the element Y_elem_idx it lies on, and the unit normal Ynstar there.
    Ystar.ReInit(COORD_DIM);
    Ynstar.ReInit(COORD_DIM);
    Y_elem_idx = -1;
    Real best = -1;
    for (Long e = 0; e < nelem; e++) {
      Real u, v;
      Vector<Real> Xs, Ns;
      const Real d = GetClosestPoint(u, v, &Xs, &Ns, e, Xtrg);
      if (best < 0 || d < best) {
        best = d;
        Y_elem_idx = e;
        Ystar = Xs;
        Ynstar = Ns;
      }
    }
  }

  template <class Real> template <class Kernel> void QuadElemList<Real>::NearInterac(Matrix<Real>& M, const Vector<Real>& Xt, const Vector<Real>& normal_trg, const Kernel& ker, Real tol, const Long elem_idx, const ElementListBase<Real>* self) {
    // Direct adaptive-quadrature near-singular interaction. Per target, the
    // parameter square [0,1]^2 of the element is refined into a graded quadtree
    // of leaf panels: a panel is a leaf once the target lies outside its
    // Bernstein-ellipse neighborhood (so an order x order Gauss-Legendre rule
    // resolves the kernel to tolerance), else it is split into 4 children. Each
    // leaf is integrated with an order x order GL rule and accumulated into the
    // source nodal basis. This is the 2D analogue of the 1D forward-marching
    // adaptive scheme in SlenderElemList::NearInteracHelper.
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

    // Fix the Bernstein ellipse parameter rho and derive the per-panel
    // Gauss-Legendre order from rho and tol (cf. SlenderElemList::NearInteracHelper).
    // A panel of physical extent L is admissible (resolved to tol by an order
    // QuadOrder GL rule) when the target is at distance >= b_ellipse*L.
    const Real tol_ = std::max<Real>(tol, machine_eps<Real>());
    const double rho = 2.5;
    const Real b_ellipse = (Real)((rho + 1/rho) / 4);
    const Integer QuadOrder = std::max<Integer>(1, (Integer)std::ceil(-std::log(((15.0*(rho*rho-1))/64.0)*(double)tol_)/std::log(rho)*0.5 + 1));

    const Vector<Real>& pnds = ParamNodes(order);          // patch nodal-basis nodes
    const Vector<Real>& qnds = ParamNodes(QuadOrder);      // per-panel quadrature nodes
    const Vector<Real>& qwts = LegQuadRule<Real>::wts(QuadOrder);

    constexpr Integer max_depth = 20;
    constexpr Long MaxLeaves = 4096;

    struct Panel { Real u0, u1, v0, v1; Integer depth; };

    for (Long t = 0; t < Ntrg; t++) {
      StaticArray<Real,COORD_DIM> Xtrg;
      for (Integer k = 0; k < COORD_DIM; k++) Xtrg[k] = Xt[t*COORD_DIM+k];
      const Vector<Real> Xtrg_v(COORD_DIM, Xtrg, false);

      // Closest point (u*,v*) on the element; used to seed the grading.
      Real ustar, vstar;
      qel.GetClosestPoint(ustar, vstar, nullptr, nullptr, elem_idx, Xtrg_v);

      // Build the graded quadtree of leaf panels.
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

        if (dist >= b_ellipse*std::max<Real>(Lu, Lv) || p.depth >= max_depth) {
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

      // Integrate each leaf, accumulating into the source nodal basis.
      Matrix<Real> M_acc(nnode, KDIM0*KDIM1_out);
      M_acc.SetZero();

      const Long nq = (Long)QuadOrder * QuadOrder; // quadrature nodes per leaf panel
      Vector<Real> u_param(QuadOrder), v_param(QuadOrder);
      Vector<Real> Xl, Xnl, Xal, Mu(order*QuadOrder), Mv(order*QuadOrder);
      StaticArray<Real,COORD_DIM> Xt0{0, 0, 0};
      const Vector<Real> Xt0_v(COORD_DIM, Xt0, false);
      for (const Panel& p : leaves) {
        for (Integer a = 0; a < QuadOrder; a++) u_param[a] = p.u0 + (p.u1-p.u0)*qnds[a];
        for (Integer b = 0; b < QuadOrder; b++) v_param[b] = p.v0 + (p.v1-p.v0)*qnds[b];
        qel.GetGeom(&Xl, &Xnl, &Xal, nullptr, nullptr, u_param, v_param, elem_idx);

        // 1D Lagrange weights from patch nodes to panel quad nodes:
        // Mu[i*QuadOrder+a] = L_i(u_param[a]).
        LagrangeInterp<Real>::Interpolate(Mu, pnds, u_param);
        LagrangeInterp<Real>::Interpolate(Mv, pnds, v_param);

        const Real Jac = (p.u1-p.u0)*(p.v1-p.v0);

        // Shifted source coords (translation-invariant kernel: improves
        // conditioning when the target is close), normals, and GL weights.
        Vector<Real> Xsrc(nq*COORD_DIM), Xnsrc(nq*COORD_DIM), wq(nq);
        for (Integer a = 0; a < QuadOrder; a++) {
          for (Integer b = 0; b < QuadOrder; b++) {
            const Long q = a*QuadOrder + b;
            for (Integer k = 0; k < COORD_DIM; k++) {
              Xsrc[q*COORD_DIM+k] = Xl[q*COORD_DIM+k] - Xtrg[k];
              Xnsrc[q*COORD_DIM+k] = Xnl[q*COORD_DIM+k];
            }
            wq[q] = Xal[q]*qwts[a]*qwts[b]*Jac;
          }
        }

        // Kernel matrix from leaf sources to the (shifted) single target;
        // KernelMatrix already applies uKerScaleFactor, so it is not reapplied.
        Matrix<Real> Mker;
        ker.template KernelMatrix<Real,false>(Mker, Xt0_v, Xsrc, Xnsrc); // (nq*KDIM0 x KDIM1full)

        // Weighted kernel reshaped to KW (nq x KDIM0*KDIM1_out); for the
        // dot-product case, contract the innermost COORD_DIM with the target normal.
        Matrix<Real> KW(nq, KDIM0*KDIM1_out);
        for (Long q = 0; q < nq; q++) {
          for (Integer k0 = 0; k0 < KDIM0; k0++) {
            for (Integer k1 = 0; k1 < KDIM1_out; k1++) {
              Real val;
              if (trg_dot_prod) {
                val = 0;
                for (Integer l = 0; l < COORD_DIM; l++) {
                  val += Mker[q*KDIM0+k0][k1*COORD_DIM+l] * normal_trg[t*COORD_DIM+l];
                }
              } else {
                val = Mker[q*KDIM0+k0][k1];
              }
              KW[q][k0*KDIM1_out+k1] = val*wq[q];
            }
          }
        }

        // Minterp (patch nodal basis -> leaf quad nodes), tensor product of Mu, Mv.
        Matrix<Real> Minterp(nnode, nq);
        for (Integer i = 0; i < order; i++) {
          for (Integer j = 0; j < order; j++) {
            for (Integer a = 0; a < QuadOrder; a++) {
              for (Integer b = 0; b < QuadOrder; b++) {
                Minterp[i*order+j][a*QuadOrder+b] = Mu[i*QuadOrder+a]*Mv[j*QuadOrder+b];
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
