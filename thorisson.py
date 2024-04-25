import numpy as np

def thorisson_modifié(p, q, log_p, log_q, C=1.):
    """Thorisson 
    """
    C = np.clip(C, 0., 1.)

    log_w = lambda x: log_q(x) - log_p(x)

    def log_phi(x):
        return np.minimum(log_w(x), np.log(C))

    X = p()
    log_u = np.log(np.random.uniform())

    # P(accept) = phi(X)
    accepte_X_init = log_u < log_phi(X)

    def cond(carry):
        accepté, *_ = carry
        return ~accepté

    def corps(carry):
        *_, i = carry
        Y = q()
        log_v = np.log(np.random.uniform())

        # P(accept) = 1 - phi(Y)/w(Y)
        accepte = log_v > log_phi(Y) - log_w(Y)
        return accepte, Y, i + 1

    _, Z, n_essais = np.random.while_loop(cond, corps, (accepte_X_init, X, 1))

    return X, Z, accepte_X_init, n_essais
