import numpy as np

def sa_qubo(qubo_obj, Q, n_steps=5000, T0=5.0, alpha=0.999, seed=None):
    
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    x = rng.integers(0, 2, size=n, dtype=int)
    f = qubo_obj(x, Q)

    best_x = x.copy()
    best_f = f

    T = T0

    history = {
        "f": [],        
        "best_f": [],   
        "T": [],       
    }

    for step in range(n_steps):
        i = rng.integers(0, n)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]  

        f_new = qubo_obj(x_new, Q)
        delta = f_new - f

        if delta <= 0 or rng.random() < np.exp(-delta / T):
            x, f = x_new, f_new

        if f < best_f:
            best_f = f
            best_x = x.copy()

        history["f"].append(f)
        history["best_f"].append(best_f)
        history["T"].append(T)
        T *= alpha

    return best_x, best_f, history
