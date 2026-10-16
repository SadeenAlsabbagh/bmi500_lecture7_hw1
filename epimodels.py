"""
epimodels.py -- compartmental epidemic models and hand-written ODE solvers.

BMI 500, Lecture 7, HW 1.

Design:
  * A model is a right-hand-side function  rhs(t, y, params) -> dy/dt.
  * A solver step (Euler, RK4) knows nothing about epidemics; it only calls rhs.
  * simulate() marches a step function from t = 0 to t_end.
So the SEIR model (Part D) only needed one new rhs function.

No built-in ODE solver is used here. scipy.integrate.solve_ivp is used only in
the notebooks, to verify these solvers.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model: SIR (counts, not fractions)
# ---------------------------------------------------------------------------
def sir_rhs(t, y, params):
    """Right-hand side of the SIR model.

    dS/dt = -beta*S*I
    dI/dt =  beta*S*I - gamma*I
    dR/dt =  gamma*I

    y      : array [S, I, R] (numbers of people)
    params : dict with 'beta' (per person per day) and 'gamma' (per day)
    t      : time in days (unused: the model is autonomous, but solvers pass it)
    """
    S, I, R = y
    beta, gamma = params["beta"], params["gamma"]
    new_infections = beta * S * I      # people moving S -> I per day
    new_recoveries = gamma * I         # people moving I -> R per day
    return np.array([-new_infections,
                     new_infections - new_recoveries,
                     new_recoveries])


# ---------------------------------------------------------------------------
# Model: SEIR with births and deaths (counts)
# ---------------------------------------------------------------------------
def seir_rhs(t, y, params):
    """Right-hand side of the SEIR model with equal birth and death rates.

    dS/dt = mu*N - beta*S*I - mu*S
    dE/dt = beta*S*I - (sigma + mu)*E
    dI/dt = sigma*E - (gamma + mu)*I
    dR/dt = gamma*I - mu*R

    y      : array [S, E, I, R] (numbers of people)
    params : dict with 'beta' (per person per day), 'sigma', 'gamma', 'mu' (per day)
             and 'N' (population size that sets the number of births, mu*N per day)
    Summing the equations gives d(S+E+I+R)/dt = mu*(N - (S+E+I+R)),
    so the total stays at N if it starts at N.
    """
    S, E, I, R = y
    beta, sigma, gamma, mu, N = (params[k] for k in ("beta", "sigma", "gamma", "mu", "N"))
    births         = mu * N            # newborns enter S
    new_infections = beta * S * I      # S -> E (infected, not yet infectious)
    new_infectious = sigma * E         # E -> I (end of latent period, mean 1/sigma)
    new_recoveries = gamma * I         # I -> R
    return np.array([births - new_infections - mu * S,
                     new_infections - new_infectious - mu * E,
                     new_infectious - new_recoveries - mu * I,
                     new_recoveries - mu * R])


# ---------------------------------------------------------------------------
# One-step solvers (model-independent)
# ---------------------------------------------------------------------------
def euler_step(rhs, t, y, h, params):
    """Forward Euler: assume the slope at t_n holds over the whole step.

    y_{n+1} = y_n + h * f(t_n, y_n)          local error O(h^2), global O(h)
    """
    return y + h * rhs(t, y, params)


def rk4_step(rhs, t, y, h, params):
    """Classical 4th-order Runge-Kutta: average four slope samples in the step.

    k1 at the start, k2 and k3 at the midpoint, k4 at the end;
    y_{n+1} = y_n + h/6 (k1 + 2k2 + 2k3 + k4)  local error O(h^5), global O(h^4)
    """
    k1 = rhs(t, y, params)
    k2 = rhs(t + h / 2, y + h / 2 * k1, params)
    k3 = rhs(t + h / 2, y + h / 2 * k2, params)
    k4 = rhs(t + h, y + h * k3, params)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def simulate(rhs, y0, t_end, h, params, method="rk4"):
    """Integrate dy/dt = rhs(t, y, params) from t = 0 to t_end with fixed step h.

    Returns
      t : (n+1,) array of times 0, h, 2h, ..., t_end
      Y : (n+1, len(y0)) array; row k is the state at t[k]
    """
    step = STEPPERS[method]
    n_steps = int(round(t_end / h))
    if not np.isclose(n_steps * h, t_end):
        raise ValueError("t_end must be a whole multiple of h")

    t = np.linspace(0.0, t_end, n_steps + 1)
    Y = np.zeros((n_steps + 1, len(y0)))
    Y[0] = y0
    for n in range(n_steps):
        Y[n + 1] = step(rhs, t[n], Y[n], h, params)
    return t, Y
