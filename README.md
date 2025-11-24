# Tensor-Train Factorized Probability Models for Estimation of Distribution Algorithms
Estimation of Distribution Algorithms (EDAs) work by iteratively learning a probabilistic model that captures the distribution of the best-performing candidates. This idea closely mirrors the cross-entropy method (CEM), where an explicit probability model called the 'proposal distribution' is maintained over the search space (also referred to as design space). Each iteration draws new samples from this distribution, evaluates them (or some other form of signal, maybe a mini local optimization over the candidates etc.), and then updates the proposal model so that it better matches the top-ranked samples. Over time, this proposal distribution becomes increasingly concentrated around the global optimum.

This repository builds tensor-train/Matrix-Product state (TT/MPS) factorized probability models as scalable and expressive proposal distributions for EDAs. The representation enables efficient sampling, parameter updates via local sweeps, and tractable handling of high-dimensional discrete spaces.

## Features
- Lightweight NumPy backend implementing tensor-train probability models and optimization routines
- DMRG-style local updates
    - Single-site and two-site sweeps
    - Adaptive bond dimensions
    - Coordinate ascent/descent–style parameter updates
- Gradient-based optimization in the tangent space for smoother updates
- Multiple probability definitions, including non-negative TT models and Born-machine formulations
- Applications beyond black-box optimization, including portfolio optimization, QUBO-style problems, and more coming soon 

## Installation
```bash
# Install directly from GitHub
pip install "git+https://github.com/ZekiZeybek/Tensor-Train-EDA.git"
# Or clone for development
git clone https://github.com/ZekiZeybek/Tensor-Train-EDA.git
cd Tensor-Train-EDA
pip install -e .

```

## Usage 

```python
import numpy as np
from tt_eda.MPS_TN import MPS
from tt_eda.MPS_Opt import Opt_MPS


def alpine_obj(d, n, domain_a=-10., domain_b=10.):
    a = domain_a
    b = domain_b
    def func(I_vector):
        X = I_vector.astype(np.float64) / (n - 1) * (b - a) + a
        term = X * np.sin(X) + 0.1 * X
        y = np.sum(np.abs(term))
        return y
    return func, domain_a, domain_b


def make_random_qubo(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q = 0.5 * (A + A.T)  
    return Q

def make_qubo_obj(Q: np.ndarray):
    def f(x: np.ndarray) -> float:
        return float(x @ Q @ x)
    return f


if __name__ == "__main__":

    ## Function optimization example

    ## MPS/TT parameters (parameters of the optimization problem ...)
    num_args = 7
    n_grid_args= 11
    chi = 3
    ## TT-EDA parameters
    tot_func_eval =2000 
    num_samples = 30
    num_elites = 5
    learning_rate = 0.06 
    num_sweeps =5 
    seed = np.random.randint(1, 1000000)

    # Init the proposal distribution
    mps_proposal_dist = MPS.uniform_pdf_MPS(length=num_args, phys_leg_dim=n_grid_args, const_chi=chi)

    # Init the optimizer
    target_function, func_domain_a, func_domain_b = alpine_obj(num_args, n_grid_args)
    optimizer= Opt_MPS(
        bbox_func=target_function,
        MPS_instance=mps_proposal_dist,
        num_bbox_func_query=tot_func_eval,
        tot_num_samples=num_samples,
        num_top_samples=num_elites,
        learning_rate=learning_rate,
        num_sweeps=num_sweeps
    )

    x_min_final, y_min_final, history = optimizer.opt_engine_weighted_stag(base_seed=seed)
    print("Final optimal x in MPS indices:", x_min_final)
    print("Final optimal x (mapped to function domain):", x_min_final.astype(np.float64) / (n_grid_args - 1) * (func_domain_b - func_domain_a) + func_domain_a)
    print("Final optimal f(x):", y_min_final)


    ## QUBO optimization example
    num_args = 20
    n_grid_args= 2
    chi = 3
    Q = make_random_qubo(num_args, seed=seed)
    f = make_qubo_obj(Q)  

    # Init the proposal distribution
    mps_proposal_dist = MPS.uniform_pdf_MPS(length=num_args, phys_leg_dim=n_grid_args, const_chi=chi)

    # Init the optimizer
    optimizer= Opt_MPS(
        bbox_func=f,
        MPS_instance=mps_proposal_dist,
        num_bbox_func_query=tot_func_eval,
        tot_num_samples=num_samples,
        num_top_samples=num_elites,
        learning_rate=learning_rate,
        num_sweeps=num_sweeps
    )

    # Run the optimization
    x_min_final, y_min_final, history = x_min_final, y_min_final, history = optimizer.opt_engine_weighted_stag(base_seed=seed)
    print("Final optimal bitstring:", x_min_final)
    print("Final optimal xQx:", y_min_final)
```

### Notebooks
The following notebooks walk through:
- Building and initializing TT-factorized probability models
- Running the EDA loop with single-site or two-site updates
- Comparing different probability definitions (non-negative vs Born)
- Applying TT-EDA to QUBO problems and portfolio optimization

In particular,
See `notebooks/`:
- `tt_pmf.ipynb`: This notebook goes through forming a proposal distribution with respect to different probability models using tensor-train networks
- `func_opt.ipynb`: This notebook goes through the use case of optimizing test functions to probe the performance (lacks comparision with other methods, later will be added)
- `portfolio_opt.ipynb`: This notebook goes through the use case of portfolio optimization problems formulated in QUBO form 


## TODO
- General cleaning up needed
- Test the optimizer on probabilistic parametric function fitting
- Generalize the run() interface to support different gradient modes + probability model families
- Refactor for a cleaner OOP architecture and integrate JAX backends for heavy math (current single(two)-site NumPy version is already stable)
- Add QAOA-style use case + benchmarking
- Improve documentation + add thorough docstrings


## Note
- This repo. builds on and extends the ideas introduced in [PROTES: Probabilistic Optimization with Tensor Networks](https://arxiv.org/pdf/2301.12162).
- For the interested reader, must check out the book ['Algorithms for Optimization' (Mykel J. Kochenderfer, Tim A. Wheeler)](https://mitpress.mit.edu/9780262039420/algorithms-for-optimization/) for more theory on stochastic optimization techniques that strategically utilize randomness for doing cool useful stuff
- [Dimitrios Diplaris](https://github.com/didiplar) implemented the backend for the stock data gathering for the portflio optimization use case.

