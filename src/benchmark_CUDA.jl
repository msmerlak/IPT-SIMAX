using DrWatson
@quickactivate

using IterativePerturbationTheory

using CUDA
using LinearAlgebra, SparseArrays

using BenchmarkTools: @belapsed

function benchmark_CUDA!(df, M)

    N = size(M, 1)
    T = eltype(M)

    q = 5

    X₀ = typeof(M)(I, size(M)...)

    t₀ = @belapsed $M*$X₀ 
    tol = 1e-5

    # IPT - unaccelerated

    z = ipt(M, N, X₀; acceleration = :none)
    t = @belapsed ipt($M, $N, $X₀; acceleration = :none)

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [N, :ipt, z.values[1:q], residual, t/t₀])
    else
        push!(df, [N, :ipt, missing, missing, missing])
    end

    # IPT - accelerated

    z = ipt(M, N, X₀)
    t = @belapsed ipt($M, $N, $X₀)

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [N, :ipt, z.values[1:q], residual, t/t₀])
    else
        push!(df, [N, :ipt, missing, missing, missing])
    end



    # SYEVD

    MM = copy(M)
    vals, vecs = CUSOLVER.syevd!('V','U', MM)
    residual = norm(M * vecs - vecs * Diagonal(vals[:, 1]))

    t = @belapsed CUSOLVER.syevd!('V','U', $MM)

    push!(df, [N, :syevd, vals[1:q], residual, t/t₀])

    # SYEVJ

    MM = copy(M)
    vals, vecs = CUSOLVER.syevjBatched!('V','U', MM)
    residual = norm(M * vecs - vecs * Diagonal(vals[:, 1]))

    t = @belapsed CUSOLVER.syevjBatched!('V','U', $MM)

    push!(df, [N, :syevj, vals[1:q], residual, t/t₀])

    return df

end
