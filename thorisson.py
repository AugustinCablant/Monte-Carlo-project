import numpy as np

def thorisson_fonction(p, q, log_p, log_q, C=1., n_samples=10000):
    """Algorithme de couplage de Thorisson
    """
    C = np.clip(C, 0., 1.)

    log_w = lambda x: log_q(x) - log_p(x)

    def log_phi(x):
        return np.minimum(log_w(x), np.log(C))

    X = p.rvs()
    log_u = np.log(np.random.uniform(size=n_samples))

    # P(accepte) = phi(X)
    accepte_X_init = log_u < log_phi(X)

    Y = q.rvs(size=n_samples)
    log_v = np.log(np.random.uniform(size=n_samples))

    # P(accepte) = 1 - phi(Y)/w(Y)
    accepte = log_v > log_phi(Y) - log_w(Y)

    Z = np.where(accepte, Y, np.nan)
    est_couplé = accepte_X_init & (Z == X)

    n_essais = np.count_nonzero(~est_couplé)

    # Prendre le premier couplage trouvé
    Z_couplé = Z[est_couplé][0]

    return X, Z_couplé, est_couplé, n_essais
