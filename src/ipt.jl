using LinearAlgebra
using NLsolve, SpeedMapping
using ElasticArrays

include("cuda-utils.jl")
include("acx.jl")

function eigs_ipt(H::AbstractMatrix; i = 1, acceleration = false, memory = 10, tol = 1e-12)

    # Epstein-Nesbet partitioning
    H0 = Array(diag(H))

    # Reduced resolvent
    gaps = H0[i] .- H0; R0 = 1. ./gaps; R0[i] = 0.

    # Unperturbed eigenvector (basis state)
    e = zeros(eltype(H), size(H, 1)); e[i] = 1.
    
    function q!(Q, v; i = 1)
        mul!(Q, H, v)
        Q .-= H0.*v
        Q .-= Q[i]*v
        Q .*= R0
        Q[i] = 1.
    end

    if acceleration == :anderson

        sol = NLsolve.fixedpoint(q!, e; method = :anderson, ftol = tol, m = memory, extended_trace = false, store_trace = true)

        return (
            vector = sol.zero, 
            value = (H*sol.zero)[i], 
            errors = [sol.trace[i].fnorm for i in 1:sol.iterations]
            )

    elseif acceleration == :ACX

        sol = SpeedMapping.speedmapping(e; m! = q!, tol = tol)
        return (vector = sol.minimizer, value = (H*sol.minimizer)[i], iterations = sol.maps)
    end
end

function ipt(M; 
    pairs = size(M, 1), 
    tol = 1e-10, 
    acceleration = :acx,
    save_residuals = true,
    initial_condition = nothing,
    acceleration_kwargs...
    )
    N = size(M, 1)
    #M = M[sortperm(diag(M)), sortperm(diag(M))]

    if initial_condition == nothing
        if issymmetric(M)
            initial_condition = Matrix{Float64}(I, N, pairs)
        else
            initial_condition = Matrix{ComplexF64}(I, N, pairs) .+ 1e-3im 
        end
    end


    Δ = M - Diagonal(M)
    D = diag(M)

    function F!(Y, X)
        mul!(Y, Δ, X)
        for k in 1:size(X, 2)
            Y[:, k] .-= Y[k, k]*X[:, k]
            Y[:, k] ./= D[k] .- D 
        end
        Y[diagind(Y)] .= 1.
    end


    if acceleration == :acx

        sol = acx(F!, initial_condition; acceleration_kwargs...)
        return (
            vectors = sol.solution, 
            values = D[1:pairs] + diag(Δ*sol.solution), 
            errors = sol.errors,
            matvecs = sol.f_calls
            )

    elseif acceleration == :anderson

            sol = NLsolve.fixedpoint(F!, initial_condition; method = :anderson, ftol = tol, m = memory, store_trace = save_residuals, acceleration_kwargs...)
    
            return (
                vectors = sol.zero, 
                values = D[1:pairs] + diag(Δ*sol.zero), 
                errors = save_residuals ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
                )
        
    end
end