import numpy as np
from copy import copy
from scipy.integrate import RK45, solve_ivp
from scipy.optimize import fsolve

import coeffs_collection as collection
from ode_collection import ODE


class OneStepMethod:
    def __init__(self, **kwargs):
        self.name = 'default_method'
        self.p = None  # order
        self.__dict__.update(**kwargs)

    def step(self, func: ODE, t, y, dt):
        """
        make a step: t => t+dt
        """
        raise NotImplementedError


class ExplicitEulerMethod(OneStepMethod):
    """
    Explicit Euler method (no need to modify)
    order is 1
    """
    def __init__(self):
        super().__init__(name='Euler (explicit)', p=1)

    def step(self, func: ODE, t, y, dt):
        return y + dt * func(t, y)


class ImplicitEulerMethod(OneStepMethod):
    """
    Implicit Euler method
    order is 1
    """
    def __init__(self):
        super().__init__(name='Euler (implicit)', p=1)

    def step(self, func: ODE, t, y, dt):
        raise NotImplementedError


class RungeKuttaMethod(OneStepMethod):
    """
    Explicit Runge-Kutta method with (A, b) coefficients
    Rewrite step() method without usage of built-in RK45()
    """
    def __init__(self, coeffs: collection.RKScheme):
        super().__init__(**coeffs.__dict__)

    def step(self, func: ODE, t, y, dt):
        A, b = self.A, self.b
        # rk = RK45(func, t, y, t + dt)
        # rk.h_abs = dt
        # rk.step()
        # return rk.y
        y1 = y
        s = len(b)
        phi = 0
        K = [0 for i in range(s)]
        for i in range(s):
            left_arg, right_arg = t + dt * sum([A[i][j] for j in range(i)]), y + dt * sum([A[i][j] * K[j] for j in range(i)])
            K[i] = func(left_arg, right_arg)
            phi += b[i] * K[i]


        return y + dt * phi

class EmbeddedRungeKuttaMethod(RungeKuttaMethod):
    """
    Embedded Runge-Kutta method with (A, b, e) coefficients:
    y1 = RK(func, A, b)
    y2 = RK(func, A, d), where d = b+e
    embedded_step() method should return approximation (y1) AND approximations difference (dy = y2-y1)
    """
    def __init__(self, coeffs: collection.EmbeddedRKScheme):
        super().__init__(coeffs=coeffs)

    def embedded_step(self, func: ODE, t, y, dt):
        A, b, e = self.A, self.b, self.e
        c = np.sum(A, axis=0)
        y1 = RK45(func, A, b)
        y2 = RK45(func, A, b + e)
        return y1, y2-y1

        raise NotImplementedError

class EmbeddedRosenbrockMethod(OneStepMethod):
    """
    Embedded Rosenbrock method with (A, G, gamma, b, e) coefficients:
    y1 = Rosenbrock(func, A, G, gamma, b)
    y2 = Rosenbrock(func, A, G, gamma, d), where d = b+e
    embedded_step() method should return approximation (y1) AND approximations difference (dy = y2-y1)
    """
    def __init__(self, coeffs: collection.EmbeddedRosenbrockScheme):
        super().__init__(**coeffs.__dict__)

    def embedded_step(self, func: ODE, t, y, dt):
        A, G, g, b, e = self.A, self.G, self.gamma, self.b, self.e
        c = np.sum(A, axis=0)
        raise NotImplementedError
        return y1, dy
