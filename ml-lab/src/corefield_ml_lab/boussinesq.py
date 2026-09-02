"""Steady Boussinesq natural convection, 2-D Cartesian and axisymmetric.

WHY THIS IS WRITTEN WITH A BENCHMARK ATTACHED
---------------------------------------------
This project has already lost a branch to a hand-rolled finite-element solver.
`WITHDRAWN.md` records the failure: the reference solver produced a hot spot of
161, then 513, then 1960 degC as the mesh was refined, because element area
cancelled in the stiffness assembly. It never converged and nobody checked.

So the rule here is that no transformer result is reported until the solver has
reproduced a published benchmark and shown mesh convergence. The benchmark is
the de Vahl Davis differentially heated square cavity, which exists precisely to
catch this class of error.

FORMULATION
-----------
Non-dimensional, steady, Boussinesq:

    div(u) = 0
    (u.grad)u = -grad(p) + Pr*lap(u) + Ra*Pr*T*e_z
    (u.grad)T = lap(T)

Taylor-Hood elements: P2 velocity, P1 pressure, P2 temperature, which satisfies
the inf-sup condition. The velocity-pressure system is solved as a saddle point
problem; the nonlinearity is handled by Picard iteration, which is slower than
Newton but has a much wider basin of attraction and does not need a Jacobian.

AXISYMMETRIC FORM
-----------------
With `axisymmetric=True` every integral carries the radial weight r, and the
radial momentum equation gains the hoop term u_r / r**2 that the Cartesian
Laplacian does not have. Dropping that term is a standard and silent error, so
it is written explicitly below.

**(b)** Laminar and steady. Real transformer oil convection reaches Rayleigh
numbers where the flow is neither, and the CFD literature uses turbulence models
or transient solves there. Results here are valid in the laminar regime and the
limitation is reported with them rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from skfem import (Basis, BilinearForm, ElementTriP1, ElementTriP2,
                   ElementVector, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, dot, grad


@dataclass
class Solution:
    velocity: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    vel_basis: Basis
    pre_basis: Basis
    tem_basis: Basis
    iterations: int
    residual: float
    converged: bool


def _bases(mesh: MeshTri, intorder: int = 4):
    return (Basis(mesh, ElementVector(ElementTriP2()), intorder=intorder),
            Basis(mesh, ElementTriP1(), intorder=intorder),
            Basis(mesh, ElementTriP2(), intorder=intorder))


def solve_boussinesq(
    mesh: MeshTri,
    *,
    rayleigh: float,
    prandtl: float,
    hot_wall,
    cold_wall,
    temperature_source=None,
    axisymmetric: bool = False,
    max_iterations: int = 60,
    tolerance: float = 1e-8,
    relaxation: float = 0.6,
    intorder: int = 4,
) -> Solution:
    """Solve steady Boussinesq convection on `mesh`.

    Parameters
    ----------
    hot_wall, cold_wall : predicates on coordinates selecting Dirichlet
        temperature boundaries at T = 1 and T = 0. Either may be None
    temperature_source : optional f(x) volumetric heat source, which is how a
        winding is represented rather than a hot wall
    axisymmetric : weight every integral by r and add the hoop term
    relaxation : Picard under-relaxation. Natural convection stalls or diverges
        without it once Ra is large
    """
    vel, pre, tem = _bases(mesh, intorder)

    def rw(w):
        """Radial weight: r in axisymmetric coordinates, 1 in Cartesian."""
        return w.x[0] if axisymmetric else 1.0 + 0.0 * w.x[0]

    @BilinearForm
    def a_visc(u, v, w):
        base = ddot(grad(u), grad(v))
        if axisymmetric:
            # Hoop term from the axisymmetric vector Laplacian. Omitting it is
            # a silent error that leaves the solver looking convergent.
            r = w.x[0]
            safe = np.where(r > 1e-12, r, 1e-12)
            base = base + u[0] * v[0] / safe**2
        return prandtl * base * rw(w)

    @BilinearForm
    def a_conv(u, v, w):
        return dot(np.einsum('i...,ij...->j...', w['wind'], grad(u)), v) * rw(w)

    @BilinearForm
    def b_div(u, q, w):
        d = grad(u)[0][0] + grad(u)[1][1]
        if axisymmetric:
            r = w.x[0]
            safe = np.where(r > 1e-12, r, 1e-12)
            d = d + u[0] / safe
        return -d * q * rw(w)

    @LinearForm
    def f_buoy(v, w):
        return rayleigh * prandtl * w['temp'] * v[1] * rw(w)

    @BilinearForm
    def k_diff(t, s, w):
        return dot(grad(t), grad(s)) * rw(w)

    @BilinearForm
    def k_conv(t, s, w):
        return dot(w['wind'], grad(t)) * s * rw(w)

    @LinearForm
    def f_src(s, w):
        return w['q'] * s * rw(w)

    # Dirichlet sets. Velocity is no-slip on the whole boundary.
    vel_D = vel.get_dofs().all()
    tem_D, tem_val = [], tem.zeros()
    if hot_wall is not None:
        d = tem.get_dofs(hot_wall).all()
        tem_D += list(d)
        tem_val[d] = 1.0
    if cold_wall is not None:
        d = tem.get_dofs(cold_wall).all()
        tem_D += list(d)
        tem_val[d] = 0.0
    tem_D = np.unique(np.array(tem_D, dtype=int)) if tem_D else np.array([], dtype=int)

    K = asm(k_diff, tem)
    B = asm(b_div, vel, pre)
    nv, npr = vel.N, pre.N

    u = vel.zeros()
    t = tem_val.copy()
    if temperature_source is not None:
        qh = tem.project(temperature_source)
        src = asm(f_src, tem, q=tem.interpolate(qh))
    else:
        src = tem.zeros()

    residual, it, converged = np.inf, 0, False
    for it in range(1, max_iterations + 1):
        wind = vel.interpolate(u)

        # Energy with the current wind.
        Kt = K + asm(k_conv, tem, wind=wind)
        t_new = solve(*condense(Kt, src, x=tem_val, D=tem_D)) if tem_D.size \
            else spl.spsolve(Kt.tocsc(), src)

        # Momentum and continuity with the new temperature.
        temp_at_vel = tem.interpolate(t_new)
        A = asm(a_visc, vel) + asm(a_conv, vel, wind=wind)
        f = asm(f_buoy, vel, temp=temp_at_vel)
        S = sp.bmat([[A, B.T], [B, None]], format='csr')
        rhs = np.concatenate([f, np.zeros(npr)])
        # Pin one pressure dof: otherwise the system is singular up to a
        # constant and the direct solver reports success on a wrong answer.
        D = np.concatenate([vel_D, np.array([nv])])
        x = np.zeros(nv + npr)
        sol = solve(*condense(S, rhs, x=x, D=D))
        u_new = sol[:nv]

        du = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-30)
        dt = np.linalg.norm(t_new - t) / max(np.linalg.norm(t_new), 1e-30)
        residual = max(du, dt)
        u = relaxation * u_new + (1 - relaxation) * u
        t = relaxation * t_new + (1 - relaxation) * t
        if residual < tolerance:
            converged = True
            break

    return Solution(u, sol[nv:], t, vel, pre, tem, it, float(residual), converged)


def wall_nusselt(sol: Solution, wall, *, axisymmetric: bool = False) -> float:
    """Average Nusselt number on a Dirichlet wall, from the temperature gradient.

    Computed as the boundary integral of -dT/dn, which is the definition the
    de Vahl Davis benchmark tabulates.
    """
    fb = sol.tem_basis.boundary(wall)

    @LinearForm
    def flux(s, w):
        n = w.n
        return -dot(grad(w['t']), n) * s * (w.x[0] if axisymmetric else 1.0)

    @LinearForm
    def area(s, w):
        return s * (w.x[0] if axisymmetric else 1.0)

    th = fb.interpolate(sol.temperature)
    num = asm(flux, fb, t=th).sum()
    den = asm(area, fb).sum()
    return float(num / den)
