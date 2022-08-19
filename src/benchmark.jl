using Random
using LinearAlgebra, SparseArrays
using Arpack, IterativePerturbationTheory

using BenchmarkTools

using PyCall


@pyimport scipy.sparse as sp
@pyimport scipy.sparse.linalg as spla
@pyimport torch
@pyimport primme as primme
@pyimport pyscf

primme_methods = (:PRIMME_DEFAULT_MIN_TIME, :PRIMME_DEFAULT_MIN_MATVECS, :PRIMME_DYNAMIC, :PRIMME_Arnoldi, :PRIMME_GD, :PRIMME_GD_plusK, :PRIMME_GD_Olsen_plusK, :PRIMME_JD_Olsen_plusK, :PRIMME_RQI, :PRIMME_JDQR, :PRIMME_JDQMR, :PRIMME_JDQMR_ETol, :PRIMME_STEEPEST_DESCENT, :PRIMME_LOBPCG_OrthoBasis, :PRIMME_LOBPCG_OrthoBasis_Window)

all_methods = ((:IPT, :IPT_ACX, :ARPACK)..., primme_methods...)

some_methods = (:IPT, :IPT_ACX, :PRIMME_RQI, :PRIMME_GD, :PRIMME_JDQMR, :PRIMME_LOBPCG_OrthoBasis, :PRIMME_DYNAMIC)




ϵ(errors, tol) = errors == any(isnan, errors) || any(isinf, errors) ? missing : mapslices(res ->
        all(res .≤ tol) ? tol : minimum(res[res.>tol]), errors; dims=2)

function benchmark(M, k; method=:IPT, tol = 100 * eps(eltype(M)) * norm(M, Inf), condense_trace = true)

    GC.gc()

    N = size(M, 1)
    T = eltype(M)
    hermitian = M ≈ M'

    X₀ = Matrix{eltype(M)}(I, N, k)
    jiggle = similar(X₀); randn!(jiggle)
    X₀ .+= jiggle/1000

    t₀ = @belapsed $M * $X₀

    if method == :IPT

        z = ipt(M, k, X₀; tol=tol, acceleration=:none, trace=true)

        if z !== :Failed

            t = @belapsed ipt($M, $k, $X₀; tol=$tol, acceleration=:none, trace=false)

            return (
                values=z.values,
                trace= condense_trace ? ϵ(z.trace, tol) : z.trace,
                matvecs=z.matvecs, ## because IPT counts M * eigenmatrix
                relative_time=t/t₀,
                absolute_time=t
            )
        else
            return (
                values=:Failed,
                trace=:Failed,
                matvecs=:Failed,
                relative_time=:Failed
            )
        end

    elseif method == :IPT_ACX

        z = ipt(M, k, X₀; tol=tol, acceleration=:acx, trace=true)

        if z !== :Failed
            t = @belapsed ipt($M, $k, $X₀; tol=$tol, acceleration=:acx, trace=false)
            return (
                values=z.values,
                trace=condense_trace ? ϵ(z.trace, tol) : z.trace,
                matvecs=z.matvecs, 
                relative_time=t/t₀,
                absolute_time=t
            )
        else
            return (
                values=missing,
                trace=missing,
                matvecs=missing, ## because IPT counts M * eigenmatrix
                relative_time=missing,
                absolute_time = missing
            )
        end

    elseif method == :ARPACK

        vals, vecs, nconv, niter, nmult, resid = eigs(M; nev=k, which=:SR, tol=tol, v0=X₀[:, 1])

        t = @belapsed eigs($M; nev=$k, which=:SR, tol=$tol, v0=$X₀[:, 1])
        return (
            values=vals,
            trace=missing,
            matvecs=nmult,
            relative_time=t/t₀,
            absolute_time=t
        )

    elseif method == :LOBPCG

        @assert hermitian


        pyM = isa(M, AbstractSparseMatrix) ? sp.csc_matrix(M) : M
        pyP = sp.spdiags(one(T) ./ diag(M), [0], size(M)...)


        vals, vecs, trace = spla.lobpcg(pyM, X₀, M=pyP, largest=false, tol=tol, retResidualNormsHistory=true, maxiter=1e6)


        t = @belapsed spla.lobpcg($pyM, $X₀, M=$pyP, largest=false, tol=$tol, retResidualNormsHistory=false, maxiter=1e6)

        return (
            values=vals,
            trace= condense_trace ? ϵ(reduce(hcat, trace)', tol) : reduce(hcat, trace)',
            matvecs=missing,
            relative_time=t/t₀,
            absolute_time=t
        )

    elseif method ∈ primme_methods

        @assert hermitian


        pyM = isa(M, AbstractSparseMatrix) ? sp.csc_matrix(M) : M
        pyP = sp.spdiags(one(T) ./ diag(M), [0], size(M)...)

        # convtest = (eval, evec, resNorm) -> resNorm < tol

        vals, vecs, info = primme.eigsh(pyM, k, method=method, which=:SA, v0=X₀, OPinv=pyP, return_stats=true, return_history=true, maxiter=1e6, maxMatvecs = 1e6)


        t = @belapsed primme.eigsh($pyM, $k, method=$method, which=:SA, v0=$X₀, OPinv=$pyP, return_stats=false, return_history=false, maxiter=1e6, maxMatvecs = 1e6)

        return (
            values=vals,
            info=info,
            trace=[info["hist"]["resNorm"]..., maximum(info["rnorms"])],
            matvecs=[info["hist"]["numMatvecs"]..., info["numMatvecs"]],
            relative_time=t/t₀,
            absolute_time=t
        )

    elseif method == :PYSCF_DAVIDSON

        pyM = spla.aslinearoperator(isa(M, AbstractSparseMatrix) ? sp.csc_matrix(M) : M)
        pyP = Vector(diag(M))

        vals, vecs = pyscf.lib.davidson(pyM, X₀', pyP; nroots = k, tol = tol, max_cycle = Int(1e6))

        t = @belapsed pyscf.lib.davidson($pyM, $X₀', $pyP; nroots = $k, tol = $tol, max_cycle = Int(1e6))

        return (
            values=vals,
            info=missing,
            trace=missing,
            matvecs=missing,
            relative_time=t/t₀,
            absolute_time=t
        )


    elseif method == :DIRECT

        Z = eigen(Matrix(M))
        t = @belapsed eigen(Matrix($M))

        return (
            values=Z.values[1:10],
            trace=missing,
            matvecs=missing,
            relative_time=t/t₀,
            absolute_time=t
        )


    # elseif method == :MAGMA

    #     pyM = torch.tensor(M, device=torch.device("cuda"))
    #     t = @belapsed torch.linalg.eig($pyM)


    #     return (
    #         values=missing,
    #         trace=missing,
    #         matvecs=missing,
    #         relative_time=t/t₀,
    #         absolute_time=t
    #     )

    # elseif method == :CUSOLVER_SYEV

    #     @assert hermitian

    #     MM = copy(M)
    #     t = @belapsed CUDA.CUSOLVER.syevd!('V','U', $MM)

    #     return (
    #         values=missing,
    #         trace=missing,
    #         matvecs=missing,
    #         relative_time=t/t₀,
    #         absolute_time=t
    #     )

    # elseif method == :CUSOLVER_SYEVJ

    #     @assert hermitian
        
    #     MM = copy(M)
    #     t = @belapsed CUDA.CUSOLVER.syevjBatched!('V','U', $MM)

    #     return (
    #         values=missing,
    #         trace=missing,
    #         matvecs=missing,
    #         relative_time=t/t₀,
    #         absolute_time=t
    #     )

    end
end
