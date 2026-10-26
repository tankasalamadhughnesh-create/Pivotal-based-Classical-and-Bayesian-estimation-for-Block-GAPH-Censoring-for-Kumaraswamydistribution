# Statistical Inference for Kumaraswamy Distribution under Block GAPHC

This repository contains the computational and theoretical framework for parameter estimation of the **Kumaraswamy distribution** subjected to **Block Generalized Adaptive Progressive Hybrid Censoring (GAPHC)** with optional order restrictions on scale parameters.

---

## 📌 Project Overview

Progressive hybrid censoring schemes are widely used in reliability engineering, survival analysis, and life-testing experiments. This project implements:

1. **Classical Estimation**:
   - Unrestricted Maximum Likelihood Estimation (MLE) via profile log-likelihood numerical optimization.
   - Asymptotic variance-covariance matrix and standard errors derived from the **Observed Fisher Information Matrix**.
   - Order-restricted MLE ($\lambda_1 \le \lambda_2 \le \dots \le \lambda_k$) via the **Pool-Adjacent-Violators Algorithm (PAVA)**.

2. **Bayesian Estimation**:
   - Independent Gamma prior specifications.
   - **Metropolis-Hastings MCMC Sampler** for unrestricted parameter estimation.
   - Truncated MCMC Sampler for order-restricted Bayesian inference.

---

## 📁 Repository Structure
.
├── Kumaraswamy_Block_GAPHC_Final.ipynb  # Primary Jupyter Notebook with full code
├── README.md                            # Project overview and documentation

---

## 🛠 Model Formulation

### 1. Kumaraswamy Distribution
For $0 < x < 1$ and shape parameters $\alpha, \beta > 0$:

$$F(x; \alpha, \beta) = 1 - (1 - x^{\alpha})^{\beta}, \qquad f(x; \alpha, \beta) = \alpha \beta x^{\alpha-1} (1 - x^{\alpha})^{\beta-1}$$

### 2. Likelihood under Block GAPHC
For parameter vector $\boldsymbol{\theta} = (\alpha, \lambda_1, \dots, \lambda_k)^T$ and block observations $x_{ij}$:

$$L(\boldsymbol{\theta}) = \prod_{i=1}^k \left[ \prod_{j=1}^{m_i} f_i(x_{ij}) S_i(x_{ij})^{R_{ij}} \right] S_i(\tau_i)^{C_i}$$

Defining the transformed data functional $W_i(\alpha)$:

$$W_i(\alpha) = -\left[ \sum_{j=1}^{m_i} (1 + R_{ij}) \log(1 - x_{ij}^{\alpha}) + C_i \log(1 - \tau_i^{\alpha}) \right]$$

The simplified profile estimator for $\lambda_i$ given $\alpha$ is:

$$\widehat{\lambda}_i(\alpha) = \frac{m_i}{W_i(\alpha)}$$

---

## 🚀 Getting Started

### Prerequisites

Install required Python libraries:

```bash
pip install numpy scipy matplotlib