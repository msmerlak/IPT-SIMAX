using DrWatson
@quickactivate

using PyCall
@pyimport primme as primme
@pyimport scipy.sparse as sp
@pyimport quimb as qu

include(srcdir("IterativePerturbationTheory.jl"))
#include(srcdir("DavidsonMethods.jl"))

using Preconditioners, SparseArrays

using KrylovKit
using GenericLinearAlgebra
using DFTK: LOBPCG

using BenchmarkTools
using DataFrames

function benchmark!(df, M, p; tol = 1e-10)

    N = size(M, 1)
    T = eltype(M)

    symmetric = issymmetric(M)
    density = issparse(M) ? nnz(M)/length(M) : 1.

    q = min(p, 5)

    X₀ = typeof(M)(I, N, p) + 1e-2rand(symmetric ? T : Complex{T}, N, p)
    t₀ = (@timed M*X₀).time 


    # IPT - unaccelerated

    Z = @timed ipt(M, p, X₀; tol = tol, acceleration = :none)
    z = Z.value
    t = Z.time

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [N, symmetric, density, p, :ipt, z.values[1:q], residual, z.matvecs, t/t₀])
    else
        push!(df, [N, symmetric, density, p, :ipt, missing, missing, missing, missing])
    end

    # IPT - accelerated
    Z = @timed ipt(M, p, X₀; tol = tol, acceleration = :acx)
    z = Z.value
    t = Z.time

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [N, symmetric, density, p, :ipt_acx, z.values[1:q], residual, z.matvecs, t/t₀])
    else
        push!(df, [N, symmetric, density, p, :ipt_acx, missing, missing, missing, missing])
    end


    if p ≤ round(N/10)

          #  KRYLOV-SCHUR
        krylovdim = max(KrylovDefaults.krylovdim, 4p)
        Z = @timed eigsolve(M, X₀[:, 1], p, :SR, tol = tol, issymmetric = symmetric, krylovdim = krylovdim)
        vals, vecs, info = Z.value
        t = Z.time

        vecs = reduce(hcat, vecs[1:p])

        residual = norm(M * vecs - vecs * Diagonal(vals[1:p]))
        push!(df, [N, symmetric, density, p, :krylov_schur, vals[1:q], residual, info.numops, t/t₀])

        if symmetric

            # LOBPCG - Diagonal preconditioner

            P = DiagonalPreconditioner(M)
            Z = @timed LOBPCG(M, X₀, I, P, tol)
            z = Z.value
            t = Z.time

            residual = norm(M * z.X - z.X * Diagonal(z.λ))
            push!(df, [N, symmetric, density, p, :lobpcg, z.λ[1:q], residual, z.n_matvec, t/t₀])
            
            # PRIMME - Diagonal preconditioner

            pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            pyP = sp.spdiags(1. ./diag(M), [0], N, N)

             Z = @timed primme.eigsh(pyM, p, which = :SA, v0 = X₀, OPinv = pyP, return_stats = true, tol = tol);
             vals, vecs, info = Z.value

             t = Z.time
            residual = norm(M * vecs - vecs * Diagonal(vals))

            push!(df, [N, symmetric, density, p, :primme, vals[1:q], residual, info["numMatvecs"], t/t₀])

            # SLEPC - GD

            pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            v₀ = X₀[:, 1]
            options = Dict("STType" => "precond", "KSPType" => "preonly", "PCType" => "jacobi")

            Z = @timed qu.linalg.slepc_linalg.eigs_slepc(pyM, p, which = :SR, v0 = v₀, tol = tol, EPSType = :gd, st_opts = options)
            
            vals, vecs = Z.value
            t = Z.time
            
            residual = norm(M * vecs - vecs * Diagonal(vals))

            push!(df, [N, symmetric, density, p, :slepc_gd, vals[1:q], residual, nothing, t/t₀])

            # SLEPC - JD

            pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            v₀ = X₀[:, 1]
            options = Dict("STType" => "precond", "KSPType" => "gmres", "PCType" => "jacobi")

            Z = @timed qu.linalg.slepc_linalg.eigs_slepc(pyM, p, which = :SR, v0 = v₀, tol = tol, EPSType = :jd, st_opts = options)
            vals, vecs = Z.value
            t = Z.time
            
            residual = norm(M * vecs - vecs * Diagonal(vals))

            push!(df, [N, symmetric, density, p, :slepc_jd, vals[1:q], residual, nothing, t/t₀])
            
        end

    else

        # Direct diagonalization
        
        Z = @timed eigen(Matrix(M))
        z = Z.value
        t = Z.time
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [N, symmetric, density, p, :direct, z.values[1:q], residual, missing, t/t₀])
    end

    return df

end

