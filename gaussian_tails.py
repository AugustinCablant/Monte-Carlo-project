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
        gamma = get_gamma(mu,eta)
        def p_hat(x):
            return translated_exp_dist(x, mu, alpha(mu))
        def q_hat(x):
            return translated_exp_dist(x, eta, alpha(eta))
        def min_p_q_hat(x):
            return min(p_hat(x), q_hat(x))
        integral, _ = quad(min_p_q_hat, 0, np.inf)

        if U <= integral:  #If u ≤ \int(min(ˆp(x), qˆ(x))) dx
            u = npr.uniform()
            X = inverse_cas_1(mu, eta, u)
            Y = X
        else: 
            u, v = npr.uniform(), npr.uniform()
            X = inverse_cas_2(mu, eta, u, p=True)
            Y = inverse_cas_2(mu, eta, v, p=False)


        return None

###################### Fonctions aide #######################
def get_gamma(mu, eta):
    """  Gamma comme définie dans le papier
    """
    alpha_mu = alpha(mu)
    alpha_eta = alpha(eta)
    return (np.log(alpha_eta) - np.log(alpha_mu) + eta * alpha_eta - mu * alpha_mu) / (alpha_eta - alpha_mu)

def inverse_cas_1(mu, eta, u):
    """ Cas 1)a équation 18 page 10
    """
    alpha_mu = alpha(mu)
    alpha_eta = alpha(eta)
    gamma = get_gamma(mu, eta)
    Z = np.exp(-alpha_mu * (eta -mu)) - np.exp(-alpha_mu * (gamma - mu)) + np.exp(-alpha_eta * (gamma - eta))
    Z1 = np.exp(-alpha_mu * (eta -mu)) - np.exp(-alpha_mu * (gamma - mu))
    Z2 = np.exp(-alpha_eta * (gamma - eta))
    if u <= Z1 / Z :
        return mu - np.log(np.exp(-alpha_mu * (eta - mu)) - u * 
                     (np.exp(-alpha_mu * (eta - mu)) - np.exp(-alpha_mu * (gamma -mu)))) / alpha_mu
    else:
        return gamma - (np.log(1-u)) / alpha_eta
    
def inverse_cas_2(mu, eta, u, p=True):
    """ 1)b page 28 du papier
    """
    alpha_mu = alpha(mu)
    alpha_eta = alpha(eta)
    gamma = get_gamma(mu, eta)
    if p:
        U = npr.uniform()
        Z = 1 - np.exp(-alpha_mu * (eta - mu)) + np.exp(-alpha_mu * (gamma - mu)) - np.exp(-alpha_eta * (gamma -eta))
        Z1 = np.exp(-alpha_mu * (gamma - mu)) - np.exp(-alpha_eta * (gamma -eta))
        Z2 = 1 - np.exp(-alpha_mu * (eta - mu))
        if U <= Z1 / Z:
            def resolve_chandrupatla(k):
                return (np.exp(-alpha_mu * (k - mu)) - alpha_eta * np.exp(-alpha_eta * (k - eta))) / Z1
            def inverse_func(y):
                return chandrupatla(resolve_chandrupatla(y), -10, 10)
            return inverse_func(U)
        else:
            return mu - np.log(1 - u * (1 - np.exp(-alpha_mu * (eta - mu)))) / alpha_mu
    else:
        U = npr.uniform()
        Z = 1 - np.exp(-alpha_mu * (eta - mu) + np.exp(-alpha_mu * (gamma - mu))- np.exp(-alpha_eta * (gamma-eta)))
        def resolve_c2(k):
            return (1 - np.exp(- alpha_eta * (k -eta)) - np.exp(-alpha_mu * (eta - mu)) + np.exp(-alpha_mu * (x -mu))) / Z
        def inverse_func2(y):
            return chandrupatla(resolve_c2(y), -10, 10) 
        return inverse_func2(U)
        

def chandrupatla(func, a, b, tol=1e-6, max_iter=10000):
    """
    Méthode de Chandrupatla pour trouver la racine de la fonction 'func' dans l'intervalle [a, b].
    """
    # Initialisation
    iter_count = 0
    while iter_count < max_iter:
        # Calcul du nouveau point
        x = (a * func(b) - b * func(a)) / (func(b) - func(a))
        
        # Test de convergence
        if abs(func(x)) < tol:
            return x
        
        # Mise à jour de l'intervalle
        if func(x) * func(a) < 0:
            b = x
        else:
            a = x
        
        iter_count += 1
    raise ValueError("La méthode de Chandrupatla n'a pas convergé après {} itérations.".format(max_iter))

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

