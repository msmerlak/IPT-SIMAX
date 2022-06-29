using IterativePerturbationTheory
include("IPT-convergence/src/hamiltonians.jl")
H = anharmonic_oscillator(1., dim = 10, order = 4)

ipt(H, 1; acceleration = :none)