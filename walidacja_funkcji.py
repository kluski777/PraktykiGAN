import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from numba import jit
import numpy as np

"""
    Zwykly greisen
    Ne(t)= 0.31 / sqrt(ln(E0/Ec)) *  exp[t( 1 - 3/2 ln(s))]

    t - distance the shower has developed [radiation length]
    E0 - primary energy (of a particle?)
    Ec - critical energy (of material?)
    Ne(t) - number of particles in electromagnetic shower
    s - shower age, s = 3t / (t + 2 * ln(E0/Ec))
"""

def greisen(t: np.ndarray | float, E0: float = 1e3, Ec: float = 21.8):
    """
        E0 - energia poczatkowa czastki [tutaj w MeV]
        Ec - energia typowa dla materialu [MeV]
    """
    beta0 = np.log(E0 / Ec)
    if isinstance(t, float):
        if t < 0.:
            return 0.0
        else:
            s = 3*t /(t + 2*beta0)
            return 0.31 / np.sqrt(beta0) * np.exp(t * (1 - 1.5 * np.log(s)))
    else:
        valid = t > 0
        result = np.zeros_like(t, dtype=float) # jak t nie ma sensu fizycznego to 0

        if np.any(valid):
            t_valid = t[valid]
            s = 3*t_valid / (t_valid + 2*beta0)
            result[valid] = 0.31 / np.sqrt(beta0) * np.exp(t_valid * (1 - 1.5 * np.log(s)))

        return result


# @jit(nopython=True)
def NKG(r: float, r_m: float = 79, s: float = 1, r_min: float = 1):
    """
    r - distance from the main axis
    r_m - Molier radius: for earth 79 meters at the sea level or 91 meters above the ground
    s - shower age allegedly 0 < s < 2 but the maximum is for s = 1
    r_min - a point where equation stops working
    r_max - where the equation stops working
    """
    r_max = 2 * r_m # raczej znikome szanse ze wyjdzie poza 2 r_m, model sie dosc slabo uczy dla r > 2r_m

    if isinstance(r, (float, int)):
        if r < 0.0:
            return 0.0
        if r < r_min:
            r = r_min
        if r > r_max:
            r = r_max
        r_ratio = r / r_m
        return (r_ratio) ** (s - 2) * (1 + r_ratio) ** (s - 4.5)

    elif isinstance(r, np.ndarray):
        r = r.copy()  # unikamy modyfikacji oryginalnej tablicy
        r[r < r_min] = r_min
        r[r > r_max] = r_max
        r_ratio = r / r_m
        return np.where(r_ratio >= 0.0, (r_ratio) ** (s - 2) * (1 + r_ratio) ** (s - 4.5), 0.0)


def jakis_rozklad(x):
    return np.exp(-(3 * x**4 - 2 * x**2))


def metropolis_hastings_probing(distribution: callable, length: int):
    x_actual = 0.1
    xs = np.zeros(length)
    ys = np.zeros(length)
    added = 0

    while added < length:
        x_proposed = x_actual + np.random.normal(loc=0, scale=1)
        ys[added] = distribution(x_proposed)
        prob_adding = ys[added] / distribution(x_actual) # te rozklady nie zaleza od polozenia poprzedniego punktu

        if np.random.uniform(low=0., high=1.) <= prob_adding: # w teorii powinno byc min(1, prob_adding) ale to nic nie zmienia w praktyce
            x_actual = x_proposed
        else:
            ys[added] = ys[added-1]
        xs[added] = x_actual
        added += 1

    return xs, ys

class rs_prober_NKG:
    def __init__(self):
        minimized = minimize_scalar(lambda x: -NKG(x), method='brent')
        self.max_cache = -minimized.fun
        self.bounds_cache = {}

    def rejection_sampling(self, length: int, epsilon: float = 0.1, 
                            looking_x_left: float = -1, looking_x_right: float = 1, 
                            from_x: float = None):
        if from_x is None:
            from_x = fsolve(lambda x: NKG(x) - epsilon, looking_x_left)[0]
        to_x = fsolve(lambda x: NKG(x) - epsilon, looking_x_right)[0]

        xs = []
        batch_size = length * 2  # Oversampling

        while len(xs) < length:
            X = np.random.uniform(from_x, to_x, size=batch_size)
            u = np.random.uniform(0, self.max_cache, size=batch_size)
            samples = X[u <= NKG(X)]
            xs.extend(samples[:length - len(xs)])

        xs = np.array(xs[:length])
        return xs
