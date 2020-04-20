import numpy as np
from one_step_methods import OneStepMethod
from ode_collection import Harmonic
from utils import get_log_error
from solve_ode import fix_step_integration
from one_step_methods import (
    ExplicitEulerMethod,
    RungeKuttaMethod,
)
import matplotlib.pyplot as plt
import coeffs_collection as collection

#  coefficients for Adams methods
adams_coeffs = {
    1: [1],
    2: [-1 / 2, 3 / 2],
    3: [5 / 12, -4 / 3, 23 / 12],
    4: [-3 / 8, 37 / 24, -59 / 24, 55 / 24],
    5: [251 / 720, -637 / 360, 109 / 30, -1387 / 360, 1901 / 720]
}


def adams(func, y_start, T, coeffs, one_step_method: OneStepMethod):
    Yn = np.array([y_start])
    h = T[1] - T[0]
    for i in range(len(coeffs) - 1):
        Yn = np.insert(Yn, len(Yn), one_step_method.step(func, T[i], Yn[-1], h), axis=0)

    for i in range(len(coeffs), len(T)):
        series = np.array([0., 0.])
        for j in range(len(coeffs)):
            series += coeffs[j] * func(T[i - len(coeffs) + j], Yn[i - len(coeffs) + j])
        y = Yn[-1] + h * series
        Yn = np.insert(Yn, len(Yn), y, axis=0)

    return T, Yn
    raise NotImplementedError
