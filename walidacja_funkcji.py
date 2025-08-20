import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
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


def rejection_sampling_probing(distribution: callable, length: int, epsilon: int, looking_x_left: int = -1, looking_x_right: int = 1):
    """
        epsilon - zwroci takie x dla ktorych f(x) > epsilon
        X0 - charakterystyczna wartosc dla danego materialu - dlugosc radiacji
    """
    xs = []

    minimized_function = minimize_scalar(lambda x: -distribution(x), method='brent') # moze wylazic na liczby urojone to trzeba by sprawdzic
    max_value = -minimized_function.fun # trzeba znalezc maksymalna wartosc dystrubucji
    from_x = fsolve(lambda x: distribution(x) - epsilon, looking_x_left)
    to_x = fsolve(lambda x: distribution(x) - epsilon, looking_x_right)

    while len(xs) < length:
        X = np.random.uniform(from_x, to_x, size=length)
        u = np.random.uniform(0, max_value, size=length)
        samples = X[u <= distribution(X)]
        if len(xs) + len(samples) >= length:
            xs.extend(samples[:length - len(xs)])
        else:
            xs.extend(samples)

    xs = np.array(xs)
    return xs, distribution(xs)
