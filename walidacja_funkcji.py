import torch
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

"""
    s = 3/(1 + 2ln(E0/eps0)/t)
    ale t - vertical depth - funkcja 3 katow tak naprawde
    eps0 = 84 MeV - incident energy
    E0 - energia krytyczna zaloze sobie 1GeV
    Zakladajac najczestsza wartosc s = 1 => 2ln(E0/eps0)/t = 2
    A zatem wliczajac w to katy
    s = 3/(1 + 2cos(theta) * (1 - tan(a) tan(theta) cos(phi)))
"""

def NKG_torch(r:torch.Tensor, s:torch.Tensor=1, r_m:torch.Tensor=79, r_min:torch.Tensor=1):
    r_max = 2 * r_m
    r_ratio = torch.clamp(r, r_min, r_max) / r_m
    return torch.pow(r_ratio, s - 2) * torch.pow(1 + r_ratio, s - 4.5)


def NKG_np(r: float, s: float = 1, r_m: float = 79, r_min: float = 1):
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

class rs_prober_NKG:
    def __init__(self, epsilon=0.1, looking_x_left=-1, looking_x_right=1, from_x=0, theta=None, alpha=None, phi=None):
        device = 'cuda'
        self.s = torch.tensor(1.0, device=device, dtype=torch.float16)
        self.r_m = torch.tensor(79.0, device=device, dtype=torch.float16)

        if None not in (theta, alpha, phi):
            angle_part = np.cos(theta) * (1 - np.tan(alpha) * np.tan(theta) * np.cos(phi)) 
            self.s = torch.tensor(3 / (1 + 2 * angle_part), device=device, dtype=torch.float16)
            self.r_m = torch.tensor(angle_part * 79, device=device, dtype=torch.float16)

        minimized = minimize_scalar(lambda x: -NKG_np(x, self.s.cpu().item(), self.r_m.cpu().item()), method='brent')
        self.max_cache = torch.tensor(-minimized.fun, device=device, dtype=torch.float16)

        self.from_x = torch.tensor(
            fsolve(lambda x: NKG_np(x, self.s.cpu().item(), self.r_m.cpu().item()) - epsilon, looking_x_left)[0],
            device=device,
            dtype=torch.float16
        )
        self.to_x = torch.tensor(
            fsolve(lambda x: NKG_np(x, self.s.cpu().item(), self.r_m.cpu().item()) - epsilon, looking_x_right)[0],
            device=device, 
            dtype=torch.float16
        )

    def rejection_sampling(self, length: int):
        with torch.no_grad():
            xs = torch.empty(length) # preallocated tensor
            filled = 0
            x_dist = torch.distributions.Uniform(low=self.from_x, high=self.to_x)
            u_dist = torch.distributions.Uniform(0, self.max_cache)

            while filled < length:
                X = x_dist.sample((length,)).cuda()
                u = u_dist.sample((length,)).cuda()
                samples = X[u <= NKG_torch(X, self.s, self.r_m)] # zgaduje ze to jest super kosztowne

                n_samples = min(len(samples), length - filled)
                xs[filled:filled + n_samples] = samples[:n_samples]
                filled += n_samples

            return xs
