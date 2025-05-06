# Application of the Sliding Frank-Wolfe Algorithm on TV-Regularized Optimization Problems

This repository contains the implementation of the numerical experiments presented in the Master's thesis *"Application of the Sliding Frank-Wolfe Algorithm on TV-Regularized Optimization Problems"* by Lena Schmedt.

The goal is to reconstruct a ground truth — a linear combination of rectangular characteristic functions — from its truncated Fourier data.
To do so the classic Frank-Wolfe Algorithm as well as its sliding modification are used.

## Code Sources

Note, that the implementation of the Setup of a valid ground truth is based on the Code presented in https://github.com/hollerm/ani_tv_recovery, which was published in association with the paper 
- M. Holler and B. Wirth. Exact reconstruction and reconstruction from noisy data with anisotropic total variation. To appear in SIAM Journal on Mathematical Analysis, 2023. https://arxiv.org/abs/2207.04757

Moreover, the code regarding the coarse 
