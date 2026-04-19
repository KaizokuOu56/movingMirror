# ============================================================
# generate_dataset.py
# ============================================================

import numpy as np
from scipy.special import loggamma
import mpmath as mp

# -----------------------------
#? Config
#? Range of t: [0, 10], Number of samples = 128
#? Range of omega & omega_p: [0.01, 2.0], Number of samples = 30
#TODO Try with different number of time samples later
# -----------------------------
N_t = 128
N_w = 30

t = np.linspace(0.0, 10.0, N_t)
omega = np.linspace(0.01, 2.0, N_w)
omega_p = np.linspace(0.01, 2.0, N_w)

# -----------------------------
# Davies trajectory
# -----------------------------
def z_davies(t, A, B, kappa):
    return -t - A*np.exp(-2*kappa*t) + B

# -----------------------------
# Good trajectory (solve implicit)
# -----------------------------
def solve_x(t, kappa, g=1.0):
    x = -t
    for _ in range(20):
        f = g*(t + x) + np.sinh(2*kappa*x)
        df = g + 2*kappa*np.cosh(2*kappa*x)
        x -= f/df
    return x

def z_good(t, kappa):
    return np.array([solve_x(tt, kappa) for tt in t])

# -----------------------------
# Davies beta
# -----------------------------
def beta_davies(omega, omega_p, A, B, kappa):
    D = (1/kappa)*np.log(A) - B

    omega_ = omega[:,None]
    omega_p_ = omega_p[None,:]

    log_pref = (
        -np.log(2*np.pi)
        -0.5*(np.log(omega_) + np.log(omega_p_))
        -np.pi*omega_p_/(2*kappa)
    )

    phase = (
        -1j*omega_p_*D
        +1j*omega_*B
        -1j*(omega_p_/kappa)*np.log(omega_)
    )

    lgamma = loggamma(1 + 1j*omega_p_/kappa)
    log_beta = log_pref + phase + lgamma

    beta = -1j*np.exp(np.clip(log_beta.real,-700,700)) * np.exp(1j*log_beta.imag)
    return beta

# -----------------------------
# Good beta (mpmath)
# -----------------------------
def beta_good(omega, omega_p, kappa):
    beta = np.zeros((len(omega), len(omega_p)), dtype=np.complex128)

    for i,w in enumerate(omega):
        for j,wp in enumerate(omega_p):
            pref = -np.sqrt(w*wp)/(np.pi*kappa*wp)
            thermal = np.exp(-np.pi*w/(2*kappa))
            K = mp.besselk(1j*w/kappa, wp/kappa)
            beta[i,j] = pref * thermal * complex(K)

    return beta

# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(n_samples):

    X, Y = [], []

    for i in range(n_samples):

        if np.random.rand() < 0.5:
            # Davies
            #? A in [0.5, 1.5]
            #? B in [0.0, 1.0]
            #? kappa in [0.5, 1.5]

            A = np.random.uniform(0.5, 1.5)
            B = np.random.uniform(0.0, 1.0)
            kappa = np.random.uniform(0.5, 1.5)

            z = z_davies(t, A, B, kappa)
            beta = beta_davies(omega, omega_p, A, B, kappa)

        else:
            # Good
            #? kappa in [0.5, 1.5]
            
            kappa = np.random.uniform(0.5, 1.5)

            z = z_good(t, kappa)
            beta = beta_good(omega, omega_p, kappa)

        # build input channels
        channels = []
        for val in z:
            channels.append(np.full((N_w, N_w), val))

        omega_grid = np.tile(omega[:,None], (1,N_w))
        omega_p_grid = np.tile(omega_p[None,:], (N_w,1))

        channels.append(omega_grid)
        channels.append(omega_p_grid)

        X.append(np.stack(channels, axis=0))
        Y.append(np.stack([beta.real, beta.imag], axis=0))

        if i % 50 == 0:
            print(f"Generated {i}/{n_samples}")

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    #TODO   Increase samples later
    n_samples = 500

    X, Y = generate_dataset(n_samples)

    np.savez("dataset.npz", X=X, Y=Y)

    print("Dataset saved to dataset.npz")