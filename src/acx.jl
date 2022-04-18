"""
A simple implementation of Alternating Cyclic Extrapolation (ACX)
https://arxiv.org/pdf/2104.04974.pdf

See https://github.com/NicolasL-S/SpeedMapping.jl for the author's version, currently restricted to real maps.
"""

import LinearAlgebra:dot, norm

function acx(F!::Function, X₀; orders = [3, 2], tol = 1e-10, maxiters = 1000)

    P = length(orders)
    
    Δ⁰, Δ¹, Δ², Δ³ = [similar(X₀) for _ in 1:4]
    F¹, F², F³ = [similar(X₀) for _ in 1:3]

    X = copy(X₀)

    # p = NaN
    # σ = NaN

    f_calls = 0
    errors = Float64[norm(X - Δ⁰, Inf)]

    for k in 0:maxiters

        p = orders[(k % P) + 1]
        f_calls += p

        F!(F¹, X)
        F!(F², F¹)

        Δ⁰ = X 
        Δ¹ = F¹ - X
        Δ² = F² - 2F¹ + X

        if p == 2
            
            σ = abs(dot(Δ², Δ¹))/abs(dot(Δ², Δ²))
            X = Δ⁰ + 2σ * Δ¹ + σ^2 * Δ²

        elseif p == 3 
            
            F!(F³, F²)
            Δ³ = F³ - 3F² + 3F¹ - X

            σ = abs(dot(Δ³, Δ²))/abs(dot(Δ³, Δ³)) 
            X = Δ⁰ + 3σ*Δ¹ + 3σ^2*Δ² + σ^3*Δ³
        end

        push!(errors, norm(X - Δ⁰, Inf))
        errors[end] < tol && return (solution = X, errors = errors, f_calls = f_calls)
        k += 1
    end

    println("Didn't converge in $maxiters iterations.")
    
end








