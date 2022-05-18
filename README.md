# A fast iterative algorithm for near-diagonal eigenvalue problems

## Abstract

We introduce a novel eigenvalue algorithm for near-diagonal matrices inspired by Rayleigh-Schrödinger perturbation theory and termed Iterative Perturbative Theory (IPT). Contrary to standard eigenvalue algorithms, which are either 'direct' (to compute all eigenpairs) or 'iterative' (to compute just a few), IPT computes any number of eigenpairs with the same basic iterative procedure. Thanks to this perfect parallelism, IPT proves more efficient than classical methods (LAPACK or CUSOLVER for the full-spectrum problem, preconditioned Davidson solvers for extremal eigenvalues). We give sufficient conditions for linear convergence and demonstrate performance on dense and sparse test matrices, including one from quantum chemistry.

## Example

Diagonalization of a near-diagonal matrix of the matrix $M = \textrm{diag}(1,\cdots, n) + 10^{-2} \textrm{rand}(n, n)$:

![](/papers/SIMAX/resubmission/plots/timings.png)


## Reference
Kenmoe, Kriemann, Smerlak, Zadorin, _A fast iterative algorithm for near-diagonal eigenvalue problems_, [arXiv:2012.14702](https://arxiv.org/abs/2012.14702)