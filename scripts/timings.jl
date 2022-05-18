using DrWatson
include(srcdir("IterativePerturbationTheory.jl"))
include(srcdir("DavidsonMethods.jl"))


using LinearAlgebra, SparseArrays
using KrylovKit: eigsolve
using IterativeSolvers: lobpcg
using DFTK: LOBPCG
using CPUTime, BenchmarkTools
using Preconditioners

using JacobiDavidson: jdqr, jdqz

const TOL = 1e-10

using Plots; gr(dpi = 500)

M(N, η) = spdiagm(
    0 => 1:N,
    1 => fill(η, N - 1),
    2 => fill(η, N - 2),
    -1 => fill(η, N - 1),
    -2 => fill(η, N - 2)
)
S(N, η) = (M(N, η) + M(N, η)')/2
p = 1

s = S(1000, .1)

vals, vecs, info  = eigsolve(s, 1, :SR)
typeof(info) 


@time s = lobpcg(S, false, p; P = P, tol = TOL, log = true)

using LinearMaps

f = LinearMap(s)

e = zeros(1000); e[1] = 1
@time eigsolve(s, 1)

z = LOBPCG(s, Matrix{Float64}(I, size(s, 1), 2), I, TOL)

@time LOBPCG(S, Matrix{Float64}(I, N, p), I, P, TOL).λ

@time s = jdqr(S; pairs = p, verbose = false);
