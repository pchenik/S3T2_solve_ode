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

    def embedded_step(self, func: ODE, t, y, dt):
        """
        it hasn't been placed here at first
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
        f = lambda y_new: y_new - y - dt * func(t + dt, y_new)
        y_next = fsolve(f, y)
        return y + dt * func(t + dt, y_next)

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
        # rk = RK45(func, t, y, t + dt, first_step=dt)
        # rk.h_abs = dt
        # rk.step()
        # return rk.y

        # y1 = y
        # s = len(b)
        # phi = 0
        # K = [0 for i in range(s)]
        # c = np.array([np.sum(A[i][0:i]) for i in range(len(A))])
        # for i in range(s):
        #     left_arg, right_arg = t + dt * c[i], y + dt * sum([A[i][j] * K[j] for j in range(i)])
        #     K[i] = func(left_arg, right_arg)
        #     phi += b[i] * K[i]
        # return y + dt * phi

        c = np.array([np.sum(A[i][0:i]) for i in range(len(A))])
        K = np.zeros((len(c), len(y)))
        K[0] = func(t, y)
        for s, (a, c) in enumerate(zip(A[1:], c[1:]), start=1):
            dy = np.dot(K[:s].T, a[:s]) * dt
            K[s] = func(t + c * dt, y + dy)

        y_new = y + dt * np.dot(K.T, b)

        return y_new


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
        y1 = super().step(func, t, y, dt)
        self.b += e
        y2 = super().step(func, t, y, dt)
        self.b -= e
        return y1, np.linalg.norm(y2 - y1)

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

    def Rosenbrock(self, func: ODE, t, y, dt):
        A, G, g, b, e = self.A, self.G, self.gamma, self.b, self.e
        s = len(b)
        k = np.zeros((s, len(y)))
        for i in range(s):
            left_ = dt * func(t, y + np.sum([A[i][j] * k[j] for j in range(i)], axis=0))
            right_mult = np.zeros(len(y))
            if i > 0:
                right_mult = np.sum([G[i][j] * k[j] for j in range(i)], axis=0)
            right_ = dt * func.jacobian(t, y).dot(right_mult)
            INV = np.linalg.inv(np.eye(len(y)) - dt * func.jacobian(t, y) * g)
            k[i] = INV.dot(left_ + right_)

        y_new = y + np.sum([b[j] * k[j] for j in range(s)])
        return y_new

    def embedded_step(self, func: ODE, t, y, dt):
        A, G, g, b, e = self.A, self.G, self.gamma, self.b, self.e
        y1 = self.Rosenbrock(func, t, y, dt)
        self.b += e
        y2 = self.Rosenbrock(func, t, y, dt)
        self.b -= e
        dy = np.linalg.norm(y2 - y1)
        return y1, dy
        raise NotImplementedError
