import numpy as np
import scipy
import matplotlib.pyplot as plt

def beta(p, omega, A, B, kappa):
    D = 1/kappa * np.log(A) - B

    pref = 1j / 2*np.pi / (p * omega)
    exp = np.exp(-np.pi*p / (2*kappa))
    phase = np.exp(-1j * p * D) * np.exp(-1j * omega * B) * (omega**(-1j * p / kappa))
    gam = scipy.special.gamma(1 + 1j * p / kappa)

    return pref * exp * phase * gam

def alpha(p, omega, A, B, kappa):
    D = 1/kappa * np.log(A) - B

    pref = -1j / 2*np.pi / (p * omega)
    exp = np.exp(np.pi*p / (2*kappa))
    phase = np.exp(-1j * p * D) * np.exp(1j * omega * B) * (omega**(-1j * p / kappa))
    gam = scipy.special.gamma(1 + 1j * p / kappa)

    return pref * exp * phase * gam

A = 1
B = 0
kappa = 20

'''for i in range(1, 50):
    for j in range(1, 5):
        print(beta(i, j, A, B, kappa))
    print(sep="\n")
'''
modes = [(1,1), (5,5), (10,10), (100,100), (15,100), (3,4), (13,7)]
for p, w in modes:
    K = np.logspace(-4, 3, 100)
    B = beta(p, w, A, B, K)
    A = alpha(p, w, A, B, K)

    plt.plot(K, np.abs(B))
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"|$\beta$|")
    plt.title(fr"$\beta$ for $p={p} $, $\omega={w} $ vs $\kappa$")
    plt.show()
    plt.savefig(f"images/beta_vs_kappa (p={p}, w={w}).png")

    plt.close()

    plt.plot(K, np.abs(A))
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"|$\alpha$|")
    plt.title(fr"$\alpha$ for $p={p} $, $\omega={w} $ vs $\kappa$")
    plt.show()
    plt.savefig(f"images/alpha_vs_kappa (p={p}, w={w}).png")
    plt.close()