import numpy as np 
import numpy.random as npr
import scipy.stats as stats
from scipy.integrate import quad
np.random.seed(6789)

###################### Algorithme Principal #######################
def coupling_gaussian_distributions(mu, eta):
    """  Cas d'application 1 du papier
    On a supposé que eta > mu.
    """
    if eta <= mu: 
        print("Dans le papier nous supposons que eta > mu")
        return None
    else: 
        U = npr.uniform()  #Draw u ∼ U(0, 1)
        alpha_eta = alpha(eta)
        alpha_mu = alpha(mu)
        p_hat = lambda k: echantillonneur_robert(k, mu, alpha_mu)
        q_hat = lambda k: echantillonneur_robert(k, eta, alpha_eta)
        min_p_q_hat = lambda x: min(p_hat(x), q_hat(x))
        integral, _ = quad(min_p_q_hat, 0, np.inf)

        if U <= integral:  #If u ≤ \int(min(ˆp(x), qˆ(x))) dx
            U = npr.uniform()
            X = inverse_c(U, mu, eta)
            Y = X
        else: 


        return None

###################### Fonctions aide #######################
def translated_exp_dist(x, m, l):
    """  Exponentielle translatée 
    """
    if x >= m:
        return l * np.exp(- l * (x - m))
    else:
        return 0
    
def alpha(z):
    """  Fonction Alpha définie pour la fonction 
        exponentielle translatée
    """
    return (z + np.sqrt(z**2 + 4) / 2)

def gamma(mu, eta):
    """  Gamma comme définie dans le papier
    """
    alpha_mu = alpha(mu)
    alpha_eta = alpha(eta)
    return (np.log(alpha_eta) - np.log(alpha_mu) + eta * alpha_eta - mu * alpha_mu) / (alpha_eta - alpha_mu)


def echantillonneur_robert(mu, alpha):
    while True:
        u1, u2 = np.random.uniform(size=(2,))
        x = mu - np.log(1 - u1) / alpha
        accepte = u2 <= np.exp(-0.5 * (x - alpha) ** 2)
        if accepte:
            return x
        


