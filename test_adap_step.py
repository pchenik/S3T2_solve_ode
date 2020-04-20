import pytest
import numpy as np
import matplotlib.pyplot as plt

from ode_collection import Harmonic, HarmExp, Arenstorf
from utils import get_log_error
from solve_ode import adaptive_step_integration, AdaptType
from one_step_methods import (
    ExplicitEulerMethod,
    RungeKuttaMethod,
    EmbeddedRungeKuttaMethod,
)
import coeffs_collection as collection


@pytest.mark.parametrize('f,y0', (
        (Harmonic(np.array([1., 1.]), 1, 1), np.array([1., 1.])),
        (HarmExp(), np.exp([1, 0])),
))
def test_adaptive(f, y0):
    """
    test adaptive step algorithms
    """
    t0, t1 = 0, 4*np.pi

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        (ExplicitEulerMethod(), AdaptType.EULER),
        (RungeKuttaMethod(coeffs=collection.rk4_coeffs),            AdaptType.RUNGE),
        #(EmbeddedRungeKuttaMethod(coeffs=collection.dopri_coeffs),  AdaptType.EMBEDDED),
    )

    for method, adapt_type in methods:
        f.clear_call_counter()
        ts, ys = adaptive_step_integration(method=method,
                                           func=f, y_start=y0, t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        print(f'{method.name} took {f.get_call_counter()} function calls')

        tss.append(np.array(ts))
        yss.append(ys)

    ts = np.array(sorted([t for ts in tss for t in ts]))
    exact = f[ts].T
    y0 = np.array([y[0] for y in exact])

    # plots
    plt.figure('y(t)'), plt.suptitle('test_adaptive: y(t)'), plt.xlabel('t'), plt.ylabel('y')
    plt.plot(ts, y0, 'ko-', label='exact')

    plt.figure('dt(t)'), plt.suptitle('test_adaptive: step sizes'), plt.xlabel('t'), plt.ylabel('dt')
    plt.figure('dy(t)'), plt.suptitle('test_adaptive: accuracies'), plt.xlabel('t'), plt.ylabel('accuracy')

    for (m, _), ts, ys in zip(methods, tss, yss):
        plt.figure('y(t)'), plt.plot(ts, [y[0] for y in ys], '.', label=m.name)
        plt.figure('dt(t)'), plt.plot(ts[:-1], ts[1:] - ts[:-1], '.-', label=m.name)
        plt.figure('dy(t)'), plt.plot(ts, get_log_error(f[ts].T, ys), '.-', label=m.name)

    plt.figure('y(t)'), plt.legend()
    plt.figure('dt(t)'), plt.legend()
    plt.figure('dy(t)'), plt.legend()

    plt.show()


def test_adaptive_order():
    """
    test adaptive algorithms convergence
    """
    t0, t1 = 0, 2*np.pi
    y0 = np.array([1., 1.])
    f = Harmonic(y0, 1, 1)

    methods = (
        (ExplicitEulerMethod(),                                     AdaptType.EULER),
        (RungeKuttaMethod(coeffs=collection.rk4_coeffs),            AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs=collection.dopri_coeffs),  AdaptType.EMBEDDED),
    )
    tols = 10. ** -np.arange(3, 9)

    plt.figure()
    for i, (method, adapt_type) in enumerate(methods):
        print(method.name)
        fcs = []
        errs = []
        for tol in tols:
            f.clear_call_counter()
            ts, ys = adaptive_step_integration(method=method,
                                               func=f, y_start=y0, t_span=(t0, t1),
                                               adapt_type=adapt_type,
                                               atol=tol, rtol=tol*1e3)
            err = np.linalg.norm(ys[-1] - f[t1])
            fc = f.get_call_counter()
            print(f'{method.name}: {fc} RHS calls, err = {err:.5f}')
            errs.append(err)
            fcs.append(fc)

        x = np.log10(fcs)
        y = -np.log10(errs)
        k, b = np.polyfit(x, y, 1)
        plt.plot(x, k*x+b, 'k:')
        plt.plot(x, y, 'p', label=f'{method.name} ({k:.2f})')
    plt.suptitle('test_adaptive_order: check RHS evals')
    plt.xlabel('log10(function_calls)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def test_arenstorf():
    """
    https://en.wikipedia.org/wiki/Richard_Arenstorf#The_Arenstorf_Orbit
    https://commons.wikimedia.org/wiki/File:Arenstorf_Orbit.gif
    Q: which parts of the orbit are fastest?
    """
    problem = Arenstorf()
    t0, t1 = 0, 1 * problem.t_period
    y0 = problem.y0

    atol = 1e-6
    rtol = 1e-3

    tss = []
    yss = []

    methods = (
        # (ExplicitEulerMethod(),                                     AdaptType.EULER),
        (RungeKuttaMethod(coeffs=collection.rk4_coeffs),            AdaptType.RUNGE),
        (EmbeddedRungeKuttaMethod(coeffs=collection.dopri_coeffs),  AdaptType.EMBEDDED),
    )

    plt.figure('traj'), plt.suptitle('Arenstorf orbit: trajectory'), plt.xlabel('x1'), plt.ylabel('x2')
    plt.figure('dt(t)'), plt.suptitle('Arenstorf orbit: step sizes'), plt.xlabel('t'), plt.ylabel('dt')

    for method, adapt_type in methods:
        ts, ys = adaptive_step_integration(method=method,
                                           func=problem, y_start=y0, t_span=(t0, t1),
                                           adapt_type=adapt_type,
                                           atol=atol, rtol=rtol)
        tss.append(np.array(ts))
        yss.append(ys)

    for (m, _), ts, ys in zip(methods, tss, yss):
        plt.figure('traj'), plt.plot([y[0] for y in ys],
                                     [y[1] for y in ys],
                                     ':', label=m.name)
        plt.figure('dt(t)'), plt.plot(ts[:-1], ts[1:] - ts[:-1], '.-', label=m.name)

    plt.figure('traj')
    plt.plot(0, 0, 'bo', label='Earth')
    plt.plot(1, 0, '.', color='grey', label='Moon')
    plt.legend()
    plt.figure('dt(t)'), plt.legend()
    plt.show()


#test_adaptive(Harmonic(np.array([1., 1.]), 1, 1), np.array([1., 1.]))