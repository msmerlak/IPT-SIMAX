import LinearAlgebra:Diagonal, diag, diagind, issymmetric, I
import NLsolve.fixedpoint


function ipt(M, k = size(M, 1); 
    tol = 1e-10, 
    acceleration = :acx,
    save_residuals = true,
    initial_condition = nothing,
    acceleration_kwargs...
    )
    N = size(M, 1)
    T = eltype(M)
    #M = M[sortperm(diag(M)), sortperm(diag(M))]

    if initial_condition == nothing
        initial_condition = typeof(M)(I, N, k)
        if issymmetric(M)
            initial_condition = Complex.(initial_condition) .+ 1e-3im 
        end
    end

    Δ = M - Diagonal(M)
    D = diag(M)

    G = one(T) ./(D[1:k]' .- D)

    function F!(Y, X)
        Y .= Δ * X
        Y .-= X * Diagonal(Y)
        Y .*= G
        Y[diagind(Y)] .= one(T)
    end

    if acceleration == :acx

        sol = acx(F!, initial_condition; acceleration_kwargs...)

        return (
            vectors = sol.solution, 
            values = D[1:k] + diag(Δ*sol.solution), 
            errors = sol.errors,
            matvecs = sol.f_calls
            )

    elseif acceleration == :anderson

            sol = NLsolve.fixedpoint(F!, initial_condition; method = :anderson, ftol = tol, store_trace = save_residuals, acceleration_kwargs...)
    
            return (
                vectors = sol.zero, 
                values = D[1:k] + diag(Δ*sol.zero), 
                errors = save_residuals ? [sol.trace[i].fnorm for i in 1:sol.iterations] : nothing
                )
        
    end
end