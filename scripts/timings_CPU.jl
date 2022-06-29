using DrWatson
@quickactivate

using IterativePerturbationTheory, LinearAlgebra
using DataFrames, Arrow
using BenchmarkTools


results = DataFrame(
    N = [],
    symm = [],
    t_matmul = [],
    time_eigen = [],
    time_ipt = [],
    time_ipt_acx = [],
)

for n in 5:13, symm in (true, false)

    N = 2^n
    λ = .01
    M = diagm(1:N) + λ*rand(N, N)
    if symm
        M = (M + M')/2 
    end

    A = copy(M)
    t_matmul = @belapsed mul!($A, $M, $M);

    t_eigen = @belapsed eigen($M);
    t_ipt = @belapsed ipt($M, $N; acceleration = :none);
    t_ipt_acx = @belapsed ipt($M, $N; acceleration = :acx);

    result = [N, symm, t_matmul, t_eigen, t_ipt, t_ipt_acx]

    @show result
    push!(results, result)

    Arrow.write(datadir("CPU_T_vs_N"), results)
end



