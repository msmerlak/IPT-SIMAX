using DrWatson
@quickactivate

using PyCall
@pyimport primme as primme
@pyimport scipy.sparse as sp
@pyimport scipy.sparse.linalg as spla
#@pyimport quimb as qu

using Arpack
using IterativePerturbationTheory

using SparseArrays

#using KrylovKit
#using GenericLinearAlgebra
#using DFTK: LOBPCG

using BenchmarkTools
using DataFrames

function benchmark!(df, M, p; tol = 1e-10)

    N = size(M, 1)
    T = eltype(M)

    density = issparse(M) ? nnz(M)/length(M) : missing

    X₀ = Matrix{T}(I, N, p.nev) + 1e-2rand(p.symmetric ? T : Complex{T}, N, p.nev)

    # reference time
    Y = similar(X₀)
    t₀ = @belapsed mul!($Y, $M, $X₀) 


    # IPT - unaccelerated

    z = ipt(M, p.nev, X₀; tol = tol, acceleration = :none, trace = true)
    t = @belapsed ipt($M, $p.nev, $X₀; tol = $tol, acceleration = :none)

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :ipt, z.values, residual, length(z.trace), z.matvecs, t/t₀])
    else
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :ipt, missing, missing, missing, missing, missing])
    end

    # IPT - accelerated

    z = ipt(M, p.nev, X₀; tol = tol, trace = true)
    t = @belapsed ipt($M, $p.nev, $X₀; tol = $tol)

    if z !== :Failed
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :ipt_acx, z.values, residual, length(z.trace), z.matvecs, t/t₀])
    else
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :ipt_acx, missing, missing, missing, missing, missing])
    end

    if p.nev ≤ round(N/10)

        #ARPACK
        v0 = real.(X₀[:, 1])
        vals, vecs, nconv, niter, nmult, resid = eigs(M; nev=p.nev, which=:SR, tol=tol, v0=v0) 
        t = @belapsed eigs($M; nev=$p.nev, which=:SR, tol=$tol, v0=$v0) 

        residual = norm(M * vecs - vecs * Diagonal(vals))
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :arpack, vals, residual, niter, nmult, t/t₀])

        if p.symmetric

            # LOBPCG - Diagonal preconditioner

            # P = DiagonalPreconditioner(M)
            # z = LOBPCG(M, X₀, I, P, tol)
            # t = @belapsed LOBPCG($M, $X₀, $I, $P, $tol)
            # residual = norm(M * z.X - z.X * Diagonal(z.λ))
            # push!(df, [N, symmetric, density, p, :dftk_lobpcg, z.λ[1:q], residual, z.n_matvec, t/t₀])
            
            # PRIMME - Diagonal preconditioner

            pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            pyP = sp.spdiags(1. ./diag(M), [0], N, N)

            vals, vecs, info = primme.eigsh(pyM, p.nev, which = :SA, v0 = X₀, OPinv = pyP, return_stats = true, tol = tol);

            t = @belapsed primme.eigsh($pyM, $p.nev, which = :SA, v0 = $X₀, OPinv = $pyP, return_stats = false, tol = $tol);
            residual = norm(M * vecs - vecs * Diagonal(vals))

            push!(df, [p.N, p.η, p.symmetric, p.nev, density, :primme, vals, residual, info["numOuterIterations"], info["numMatvecs"], t/t₀])

            # SCIPY - LOBPCG

            pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            pyP = sp.spdiags(1. ./diag(M), [0], N, N)

            vals, vecs, trace = spla.lobpcg(pyM, X₀, M = pyP, largest = false, tol = tol, retResidualNormsHistory = true)

            t = @belapsed spla.lobpcg($pyM, $X₀, M = $pyP, largest = false, tol = $tol);

            push!(df, [p.N, p.η, p.symmetric, p.nev, density, :lopbcg, vals, residual, length(trace), missing, t/t₀])

            # # SLEPC - GD

            # pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            # v₀ = X₀[:, 1]
            # options = Dict("STType" => "precond", "KSPType" => "preonly", "PCType" => "jacobi")

            # vals, vecs = qu.linalg.slepc_linalg.eigs_slepc(pyM, p, which = :SR, v0 = v₀, tol = tol, EPSType = :gd, st_opts = options)
            # residual = norm(M * vecs - vecs * Diagonal(vals))

            # t = @belapsed qu.linalg.slepc_linalg.eigs_slepc($pyM, $p, which = :SR, v0 = $v₀, tol = $tol, EPSType = :gd, st_opts = $options)

            # push!(df, [N, symmetric, density, p, :slepc_gd, vals[1:q], residual, nothing, t/t₀])

            # # SLEPC - JD

            # pyM = isa(M, AbstractSparseMatrix) ? sp.csr_matrix(M) : M
            # v₀ = X₀[:, 1]
            # options = Dict("STType" => "precond", "KSPType" => "gmres", "PCType" => "jacobi")

            # vals, vecs = qu.linalg.slepc_linalg.eigs_slepc(pyM, p, which = :SR, v0 = v₀, tol = tol, EPSType = :jd, st_opts = options)
            # residual = norm(M * vecs - vecs * Diagonal(vals))

            # t = @belapsed qu.linalg.slepc_linalg.eigs_slepc($pyM, $p, which = :SR, v0 = $v₀, tol = $tol, EPSType = :jd, st_opts = $options)

            # push!(df, [N, symmetric, density, p, :slepc_jd, vals[1:q], residual, nothing, t/t₀])

        end

    else

        # Direct diagonalization
        
        q = 10

        z = eigen(Matrix(M))
        t = @belapsed eigen(Matrix($M))
        residual = norm(M * z.vectors - z.vectors * Diagonal(z.values))
        push!(df, [p.N, p.η, p.symmetric, p.nev, density, :direct, z.values[1:q], residual, missing, missing, t/t₀])
    end

    return df

end

