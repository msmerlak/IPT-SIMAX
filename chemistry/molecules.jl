
using CPUTime, BenchmarkTools

include("fci.jl")

using LinearMaps, LinearAlgebra
using IterativePerturbationTheory
using KrylovKit, Arpack, JacobiDavidson, IterativeSolvers, DFTK

H₂0 = """
O 0 0 0; 
H 0.2774 0.8929 0.2544;
H 0.6068, -0.2383, -0.7169
"""

Be = "Be 0 0 0"

LiH = """
Li 0 0 0
H 0 0 1.596
"""

H, na, nb, hdiag, scf, fci_solver = py"hop"(
    LiH,
    "6-31g"
)
E₀ = scf.energy_nuc()

L = LinearMap(c -> H(Array(c)), na * nb; issymmetric=true)

X = Matrix{eltype(L)}(I, L.N, 1);

#@btime E = ipt(L, 1, X; diagonal=hdiag, acceleration=:none, trace = true).trace
@time E = ipt(L, 1, X; tol = 1e-8, diagonal=hdiag, acceleration=:acx).values[1]

@time E_ref = fci_solver.kernel(tol=1e-8, max_cycle=500, max_space=100)[1]

E + E₀ ≈ E_ref