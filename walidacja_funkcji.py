from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
import numpy as np
from numba import cuda

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
    r_max - where the equation stops working, 2 * r_m
    """
    r_max = 2 * r_m

    if r < 0.0:
        return 0.0
    if r < r_min:
        r = r_min
    if r > r_max:
        r = r_max
    r_ratio = r / r_m
    return (r_ratio) ** (s - 2) * (1 + r_ratio) ** (s - 4.5)


def NKG_np(r: float, r_m: float = 79, s: float = 1, r_min: float = 1):
    r_max = 2 * r_m
    r_ratio = np.clip(r, r_min, r_max) / r_m
    return np.power(r_ratio, s - 2) * np.power(1 + r_ratio, s - 4.5)


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

# rzuc to kiedy an gpu
class rs_prober_NKG:
    def __init__(self, epsilon: float = 0.1, looking_x_left: float = -1, looking_x_right: float = 1, from_x: float = None):
        minimized = minimize_scalar(lambda x: -NKG_np(x), method='brent')
        self.max_cache = -minimized.fun
        self.from_x = from_x
        if self.from_x is None:
            self.from_x = fsolve(lambda x: NKG_np(x) - epsilon, looking_x_left)[0]
        self.to_x = fsolve(lambda x: NKG_np(x) - epsilon, looking_x_right)[0]

    def rejection_sampling(self, length: int):
        xs = np.empty(length)  # Pre-allocate
        filled = 0

        while filled < length:
            X = np.random.uniform(self.from_x, self.to_x, size=length)
            u = np.random.uniform(0, self.max_cache, size=length)
            samples = X[u <= NKG_np(X)] # zgaduje ze to jest super kosztowne

            n_samples = min(len(samples), length - filled)
            xs[filled:filled + n_samples] = samples[:n_samples]
            filled += n_samples

        return xs


@cuda.jit(device=True)
def NKG_cuda(r):
    r_m = 79.0
    s = 1.0
    r_min = 1.0
    r_max = 2 * r_m
    r_ratio = max(min(r, r_max), r_min) / r_m
    return (r_ratio ** (s - 2)) * ((1 + r_ratio) ** (s - 4.5))

def NKG_cpu(r):
    r_m = 79.0
    s = 1.0
    r_min = 1.0
    r_max = 2 * r_m
    r_ratio = np.clip(r, r_min, r_max) / r_m
    return np.power(r_ratio, s - 2) * np.power(1 + r_ratio, s - 4.5)

@cuda.jit
def rejection_kernel(rng_states, from_x, to_x, max_val, output, count):
    idx = cuda.grid(1)
    if idx >= rng_states.size:
        return
    for _ in range(100):  # Increased for more samples
        x = xoroshiro128p_uniform_float32(rng_states, idx) * (to_x - from_x) + from_x
        u = xoroshiro128p_uniform_float32(rng_states, idx) * max_val
        if u <= NKG_cuda(x):
            pos = cuda.atomic.add(count, 0, 1)
            if pos < output.size:
                output[pos] = x

class rs_prober_NKG_GPU:
    def __init__(self, epsilon=0.1):
        self.max_cache = -minimize_scalar(lambda x: -NKG_cpu(x), bounds=(1, 158), method='bounded').fun
        self.from_x = 0  # Fixed to domain min
        self.to_x = fsolve(lambda x: NKG_cpu(x) - epsilon, 79)[0]  # Better initial for convergence

    def rejection_sampling(self, length):
        threads_per_block = 256
        blocks = 128  # Increased for more parallelism
        total_threads = blocks * threads_per_block
        rng = create_xoroshiro128p_states(total_threads, seed=42)
        d_out = cuda.device_array(length * 2, dtype=np.float32)
        d_count = cuda.device_array(1, dtype=np.int32)
        d_count[0] = 0

        rejection_kernel[blocks, threads_per_block](rng, self.from_x, self.to_x, self.max_cache, d_out, d_count)

        n = d_count.copy_to_host()[0]
        return d_out.copy_to_host()[:min(length, n)]
