using LinearAlgebra, LinearMaps


using ElasticArrays


function davidson(
    H::Union{AbstractMatrix, LinearMap},
    nev = 1,
    which = :smallest;
    search_space = nothing, 
    max_dimension = 50,
    diagonal = nothing,
    tol = 1e-12)

    N = size(H,1)
    k = max(2, nev)

    if search_space == nothing
        V = ElasticMatrix(Matrix{eltype(H)}(I, N, k) .+ 1e-3rand(eltype(H), N, k))
    else
        V = ElasticMatrix(search_space)
    end
    HV = ElasticMatrix(H*V)

    residuals = zeros(N, nev)
    errors = Float64[]

    d = diagonal == nothing ? view(H, diagind(H)) : diagonal

    matvecs = 0
    while true

        while size(V, 2) < max_dimension

            ## Rayleigh-Ritz
            ritz = eigen(Symmetric(V'*HV))
            X = V * ritz.vectors[:, 1:nev]
            θ = ritz.values[1:nev]

            ## compute residual vectors R = Hx - θx
            mul!(residuals, H, X);

            matvecs += 1

            mul!(residuals, X, Diagonal(θ), -1, 1)
            
            ϵ = norm(residuals)
            @show ϵ
            push!(errors, ϵ)

            ϵ < tol && return (vectors = X, values = θ, trace = errors, matvecs = matvecs)

            ## expand subspace

            @. residuals /= d - θ'
            append!(V, residuals)


            ## orthonormalize correction vector against current basis
            V .= ElasticMatrix(qr(Matrix(V)).Q * Matrix(I, size(V)...))
            append!(HV, H*V[:, end-nev+1:end])

            if all(V[:, end] .≈ 0.) 
                println("linear dependence: restarting")
                break 
            end

        end

        ## restart
        println("max size: restarting")

        if which == :largest
            V = ElasticMatrix(V*eigen(Hermitian(V'*HV)).vectors[:, 1:k])
        else
            V = ElasticMatrix(V*eigen(Hermitian(V'*HV)).vectors[:, end-k+1:end])
        end
        HV = ElasticMatrix(H * V)
    end

end

davidson(S, 1).values
eigen(S).values[1:2]