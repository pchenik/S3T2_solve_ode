import pytest
import numpy as np
import matplotlib.pyplot as plt

from ode_collection import VanDerPol
from solve_ode import adaptive_step_integration, AdaptType
from one_step_methods import (
    ExplicitEulerMethod,
    ImplicitEulerMethod,
    EmbeddedRosenbrockMethod,
)
import coeffs_collection as collection


def test_stiff():
    """
    test explicit vs implicit methods on a stiff problem
    """
    t0 = 0
    t1 = 800*np.pi

    mu = 1000
    y0 = np.array([2., 0.])
    f = VanDerPol(y0, mu)

    colors = 'rgbcmyk'
    for i, (method, adapt_type) in enumerate(
            [
                (ExplicitEulerMethod(),                                    AdaptType.RUNGE),
                (ImplicitEulerMethod(),                                    AdaptType.RUNGE),
                (EmbeddedRosenbrockMethod(collection.rosenbrock23_coeffs), AdaptType.EMBEDDED),
            ]
    ):
        f.clear_call_counter()
        ts, ys = adaptive_step_integration(method, f, y0, (t0, t1), adapt_type=adapt_type, atol=1e-6, rtol=1e-3)
        print(f'{method.name}: {len(ts)-1} steps, {f.get_call_counter()} RHS calls')

        plt.figure(1)
        plt.plot([y[0] for y in ys],
                 [y[1] for y in ys],
                 f'{colors[i]}.--', label=method.name)
        plt.figure(2)
        plt.subplot(1,2,1), plt.plot(ts, [y[0] for y in ys], f'{colors[i]}.--', label=method.name)
        plt.subplot(1,2,2), plt.plot(ts, [y[1] for y in ys], f'{colors[i]}.--', label=method.name)
        plt.figure(3)
        plt.plot(ts[:-1],
                 np.array(ts[1:]) - np.array(ts[:-1]),
                 f'{colors[i]}.--', label=method.name)

    plt.figure(1)
    plt.xlabel('x'), plt.ylabel('y'), plt.legend()
    plt.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, y(x)')

    plt.figure(2)
    plt.subplot(1,2,1), plt.xlabel('t'), plt.ylabel('x'), plt.legend()
    plt.subplot(1,2,2), plt.xlabel('t'), plt.ylabel('y'), plt.legend()
    plt.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, x(t)')

    plt.figure(3)
    plt.xlabel('t'), plt.ylabel('dt'), plt.legend()
    plt.suptitle(f'test_stiff: Van der Pol, mu={mu:.2f}, dt(t)')

    plt.show()
