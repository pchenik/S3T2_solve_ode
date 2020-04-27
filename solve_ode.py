import enum
import numpy as np

from one_step_methods import OneStepMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, func, y_start, ts):
    """
    performs fix-step integration using one-step method
    ts: array of timestamps
    return: list of t's, list of y's
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(func, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def runge_error(y1, y2, p):
    r1 = np.linalg.norm(y2 - y1) / (1 - 2 ** (-p))
    r2 = np.linalg.norm(y2 - y1) / (2 ** p - 1)
    return max(r1, r2)

def adaptive_step_integration(method: OneStepMethod, func, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    performs adaptive-step integration using one-step method
    t_span: (t0, t1)
    adapt_type: Runge or Embedded
    tolerances control the error:
        err <= atol
        err <= |y| * rtol
    return: list of t's, list of y's
    """
    y = y_start
    t, t_end = t_span

    ys = np.array([y], dtype='float64')
    ts = np.array([t], dtype='float64')

    eps = 1e-15
    h = 0.1
    p = method.p
    err = 0

    while abs(ts[-1] - t_end) > eps:
        if ts[-1] + h > t_end:
            h = t_end - ts[-1]

        if adapt_type != AdaptType.EMBEDDED:
            ts1, y1 = fix_step_integration(method, func, ys[-1], np.linspace(ts[-1], ts[-1] + h, 2))
            ts2, y2 = fix_step_integration(method, func, ys[-1], np.linspace(ts[-1], ts[-1] + h, 3))
            err = runge_error(y1[-1], y2[-1], p)
            yy = y2[-1]
        else:
            y2, err = method.embedded_step(func, ts[-1], ys[-1], h)
            yy = y2

        #tol = np.linalg.norm(yy) * rtol + atol

        if err > atol: #err > atol:
            h = h * 0.9 * (atol / err) ** (1 / (p + 1))
            continue
        else:
            ts = np.insert(ts, len(ts), ts[-1] + h, axis=0)
            ys = np.insert(ys, len(ys), yy, axis=0)
            h = h * (atol / err) ** (1 / (p + 1))

    return ts, ys
    raise NotImplementedError