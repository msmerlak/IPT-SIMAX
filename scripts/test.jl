using DrWatson
includet(srcdir("ipt.jl"))
includet(srcdir("davidson.jl"))
includet(srcdir("lobpcg.jl"))

using SparseArrays
using KrylovKit
using BenchmarkTools
using JacobiDavidson


using Plots; gr(dpi = 500)

N = 10_000
η = .5
M = spdiagm(
    0 => 1:N,
    1 => fill(η, N - 1),
    -1 => fill(η, N - 1)
)
p = 1


@btime ipt(M; acceleration = :acx, pairs = 1, tol = 1e-10).errors

@btime davidson_method(H; method = :davidson).value


@time LOBPCG(H, Matrix{Float64}(I, N, p); tol = 1e-10);
@time davidson_method(H; method = :davidson);
plot(davidson_method(H; method = :davidson).errors, yaxis = :log)
plot!(ipt(H; pairs = p, tol = 1e-10, save_residuals = true).errors) 
plot!(LOBPCG(H, Matrix{Float64}(I, N, 1); tol = 1e-10).residual_history |> vec)

eigs_ipt(H).errors
ipt(H; pairs = 5, tol = 1e-10, acceleration = true, save_residuals = true)


@time eigs_ipt(H);
@time ipt(H; pairs = 1, tol = 1e-10, acceleration = true, save_residuals = false);