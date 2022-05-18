using DrWatson
@quickactivate

using IterativePerturbationTheory, LinearAlgebra
using DataFrames, Arrow
using BenchmarkTools
using CUDA

results = DataFrame(
    N = [],
    time_syevd = [],
    time_ipt = []
)

λ = .01

for n in 5:14

    N = 2^n
    @show N

    M = diagm(1:N) + λ*rand(N, N)
    cuS = CuArray{eltype(M)}((M + M')/2)

    cuS2 = copy(cuS)
    vals, vecs = CUDA.CUSOLVER.syevd!('V','U', cuS2)
    @show residual_syevd = norm(cuS * vecs - vecs * Diagonal(vals))

    cuS2 = copy(cuS)
    t_syevd = @belapsed CUDA.@sync CUDA.CUSOLVER.syevd!('V','U', $cuS2)


    Z = ipt(cuS, N, iterations = 7)
    @show residual_ipt = norm(cuS * Z.vectors - Z.vectors * Diagonal(Z.values))
    t_ipt = @belapsed CUDA.@sync ipt($cuS, $N, iterations = 7)

    result = [N, 
    t_syevd, 
    t_ipt
    ]

    @show result
    push!(results, result)

    CUDA.unsafe_free!(cuS)
    CUDA.unsafe_free!(cuS2)
    GC.gc()
    CUDA.reclaim()

    Arrow.write(datadir("GPU_T_vs_N"), results)
end