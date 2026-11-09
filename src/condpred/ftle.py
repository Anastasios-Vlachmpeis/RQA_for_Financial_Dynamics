"""Finite-time Lyapunov exponents along trajectories."""
from __future__ import annotations

import numpy as np


def ftle_map(jacobian_fn, states, h: int) -> np.ndarray:
    """Largest singular-value growth per step, averaged over windows of length ``h``."""
    states = np.asarray(states, float)
    n = len(states)
    out = np.full(n, np.nan)
    for t in range(n - h):
        m = np.eye(states.shape[1] if states.ndim > 1 else 1)
        for k in range(h):
            idx = t + k
            jac = np.atleast_2d(jacobian_fn(states[idx] if states.ndim == 1 else states[idx]))
            m = jac @ m
        s = np.linalg.svd(m, compute_uv=False)
        out[t] = np.log(s[0] + 1e-300) / h
    return out


def logistic_jacobian(r, x):
    return np.array([[r * (1 - 2 * x)]])


def ftle_logistic(states, r=4.0, h=10):
    return ftle_map(lambda x: logistic_jacobian(r, float(x)), states, h)


def _rk4_step_flow(f, jac, s, phi, hstep):
    """One RK4 step on state ``s`` and variational matrix ``phi`` (dPhi/dt = J Phi)."""
    j1 = jac(s)
    k1 = f(s)
    k1_phi = j1 @ phi

    s2 = s + 0.5 * hstep * k1
    j2 = jac(s2)
    k2 = f(s2)
    k2_phi = j2 @ (phi + 0.5 * hstep * k1_phi)

    s3 = s + 0.5 * hstep * k2
    j3 = jac(s3)
    k3 = f(s3)
    k3_phi = j3 @ (phi + 0.5 * hstep * k2_phi)

    s4 = s + hstep * k3
    j4 = jac(s4)
    k4 = f(s4)
    k4_phi = j4 @ (phi + hstep * k3_phi)

    s_new = s + (hstep / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    phi_new = phi + (hstep / 6.0) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi)
    return s_new, phi_new


def ftle_flow(f, jac, state0, dt, sample_every, n, h, sub=5, burn=0):
    """Integrate trajectory; FTLE via variational RK4 + QR re-orthonormalisation each sample."""
    dim = len(state0)
    s = np.asarray(state0, float).copy()
    phi = np.eye(dim)
    dt_micro = dt / sub
    dt_sample = dt * sample_every

    total_samples = n + burn
    traj = np.empty((total_samples, dim))
    log_stretch = np.zeros(total_samples)

    for i in range(total_samples):
        for _ in range(sample_every):
            for _ in range(sub):
                s, phi = _rk4_step_flow(f, jac, s, phi, dt_micro)
        q, r = np.linalg.qr(phi)
        phi = q
        log_stretch[i] = np.log(np.abs(r[0, 0]) + 1e-300)
        traj[i] = s

    traj = traj[burn:]
    log_stretch = log_stretch[burn:]
    n_out = len(traj)
    expon = np.full(n_out, np.nan)
    for t in range(n_out - h):
        expon[t] = np.sum(log_stretch[t : t + h]) / (h * dt_sample)
    return expon, traj


def lorenz_jacobian(s, sigma=10, rho=28, beta=8 / 3):
    x, y, z = s
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta],
    ])


def lorenz_rhs(s, sigma=10, rho=28, beta=8 / 3):
    return np.array([sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]])
