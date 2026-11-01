# Kumaraswamy Estimation under Block Generalized Adaptive Progressive Hybrid Censoring

Python code accompanying the paper on **classical and Bayesian estimation for the Kumaraswamy distribution under block generalized adaptive progressive hybrid censoring (block GAPHC)**, including order-restricted inference across blocks.

The repository contains a worked-example notebook that implements every estimator on a small simulated dataset, and a Monte Carlo driver for the simulation study.

## Model in brief

There are $k$ independent blocks. Block $i$ holds Kumaraswamy lifetimes with a **common shape** $\alpha$ and a **block-specific parameter** $\lambda_i$:

$$f_i(x;\alpha,\lambda_i)=\alpha\lambda_i x^{\alpha-1}(1-x^\alpha)^{\lambda_i-1},\qquad S_i(x)=(1-x^\alpha)^{\lambda_i},\quad 0<x<1.$$

A larger $\lambda_i$ means shorter lifetimes. Where blocks are naturally ranked (for example by stress level), the order restriction $\lambda_1\le\dots\le\lambda_k$ is imposed.

Each block is observed under a GAPHC scheme with $n_i$ units, a target of $r_i$ failures and thresholds $T_{1i}<T_{2i}$. One of three cases occurs:

| Case | Condition | What happens |
|---|---|---|
| I | $X_{i,r_i}<T_{1i}$ | Stop at the $r_i$-th failure; planned removals are used as is. |
| II | $T_{1i}<X_{i,r_i}<T_{2i}$ | No removals after $T_{1i}$; at the $r_i$-th failure all remaining units are removed. |
| III | $X_{i,r_i}>T_{2i}$ | Stop at $T_{2i}$; surviving units are censored there. |

## Methods implemented

- **Maximum likelihood** — profile likelihood in $\alpha$ with closed-form $\hat\lambda_i(\alpha)=m_i/W_i(\alpha)$
- **Order-restricted MLE** — pool-adjacent-violators (PAVA) pooling of the $\hat\lambda_i(\alpha)$
- **Fisher-information confidence intervals** — normal-approximation and log-transformed
- **Pivotal-based exact confidence intervals for $\alpha$**, per block — a working assumption following Wu, Tseng & Kus (*J. Supercomputing*, 2025); available for Cases I and II
- **Bayesian estimation** — Gamma priors, Gibbs sampling for $\lambda_i\mid\alpha$ and a random-walk Metropolis step for $\alpha$; posterior means and HPD intervals, unrestricted and order-restricted (truncated-Gamma updates)

## Repository contents

| File | Description |
|---|---|
| `pivotal_based_classical_and_bayesian_estimate_kumaraswamy_block_gaphc.pdf` | The paper. |
| `kumaraswamy_block_gaphc_mathematical_model.ipynb` | Worked example on one simulated 3-block dataset: data generator and every estimator above, with saved outputs. Self-contained. |
| `sim.py` | Monte Carlo driver for the simulation study (paper §4.1). |
| `kum.py` | Library module imported by `sim.py` (`gen_block`, `mle`, `fisher_ci`, `alpha_ci`, `lam_ci_known_alpha`, `bayes_draws`, `hpd`). **Must be in the same directory as `sim.py`.** |
| `case0802.csv` | Insulating-fluid breakdown-time data used in the real-data application (paper §4.2). |

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy scipy jupyter
```

The code was tested with NumPy 2.4 and SciPy 1.17.

## Quick start

### Worked example

```bash
jupyter notebook kumaraswamy_block_gaphc_mathematical_model.ipynb
```

The notebook is organised in seven parts, each mapped to a section of the paper (the mapping is in its first cell). All random draws are seeded, so the results reproduce exactly.

### Simulation study

```bash
python sim.py <setting> <replications> <bayes>
```

| Argument | Values | Meaning |
|---|---|---|
| `setting` | `A`, `B` | Parameter setting (below) |
| `replications` | integer | Number of Monte Carlo replications |
| `bayes` | `0`, `1`, `all` | `0` = classical only; `1` = also Bayes under the Gamma(2,1) prior; `all` = Bayes under all five priors |

Example:

```bash
python sim.py A 1000 1      # writes sim_A_1.json
```

Replication $i$ uses seed `1000 + i`. Results are written to `sim_<setting>_<bayes>.json`.

**Design.** Two blocks, $n=60$, $r=40$, true $\alpha=2$. The planned scheme removes one unit at each of the first 10 failures and the remaining 10 units at the $r$-th failure.

| Setting | $(\lambda_1,\lambda_2)$ | $(T_{11},T_{12})$ | $(T_{21},T_{22})$ |
|---|---|---|---|
| A | (2.0, 4.0) | (0.20, 0.15) | (0.60, 0.50) |
| B | (2.5, 3.0) | (0.20, 0.20) | (0.60, 0.60) |

Priors for the Bayesian part: Gamma(2,1), Gamma(1,1), Gamma(0.1,0.1), Gamma(4,2), Gamma(8,2) (shape, rate), used for $\alpha$ and each $\lambda_i$.

**Output.** For each replication the JSON records the censoring case and number of failures per block; unrestricted and restricted MLEs; whether pooling occurred; coverage and length of the Fisher (normal and log) intervals; coverage of the pivotal interval for $\alpha$ and of the interval for $\lambda_i$ with $\alpha$ known; and the Bayes posterior means, the probability that the unrestricted posterior satisfies the order, and HPD coverage. The order-restricted Bayes estimates are obtained from the unrestricted draws that satisfy $\lambda_1\le\lambda_2$.

## Notes on the demo notebook

- The notebook's generator is deliberately simple: the planned scheme is Type-II-style and Case III is simulated by drawing a fresh sample censored at $T_{2i}$. The notes in Part 2 explain the consequences. The simulation study uses a non-trivial removal plan.
- The Bayesian chains in the notebook are short and intended for demonstration only.
