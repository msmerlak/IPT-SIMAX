using LinearAlgebra
using NLsolve, SpeedMapping
using ElasticArrays

include("cuda-utils.jl")

function eigs_ipt(H::AbstractMatrix; i = 1, acceleration = :anderson, memory = 10, tol = 1e-12)

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
    acceleration = true,
    save_residuals = false
    )
    T = eltype(M)
    N = size(M, 1)
    #M = M[sortperm(diag(M)), sortperm(diag(M))]

    initial = typeof(M)(I, N, pairs)

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

    if T <: Real && acceleration

        sol = speedmapping(initial; m! = F!, store_info = save_residuals, tol = tol)
        @assert sol.converged "Fixed-point iteration did not converge!"
        vectors, values = sol.minimizer, D[1:pairs] + diag(Δ*sol.minimizer)

        if save_residuals
            states = sol.info.x
            residuals = [norm(states[i] - states[i-1], Inf) for i in 2:length(states)]
        else
            residuals = nothing
        end

        return (vectors = vectors, values = values, errors = residuals)

    else

        if !issymmetric(M) 
            initial .+= 1e-3im 
        end

        current = similar(initial)
        last = copy(current)

        sol = ElasticMatrix(zeros(ComplexF64, N, 0))


        while true

            F!(current, last)

            residuals = vec(mapslices(norm, current - last; dims = 1))
            converged = residuals .< tol

            if any(converged)
                append!(sol, current[:, converged])
                all(converged) && break
                current = current[:, .!converged]
            end
            last = copy(current)
        end
        return (vectors = sol, values = (M*sol./sol)[1, :], errors = nothing)
    end

        
end