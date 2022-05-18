using LinearAlgebra, LinearMaps
using IterativeSolvers: bicgstabl!

import Base:*
*(L::LinearMap, X::AbstractMatrix) = mapslices(L, X; dims = 1)

using ElasticArrays

include("linalg.jl")
include("preconditioning.jl")

function davidson_method(
    H::Union{AbstractMatrix, LinearMap},
    which = :smallest;
    method = :davidson,
    search_space = nothing, 
    max_dimension = 50,
    diagonal = nothing,
    tol = 1e-12)

    N = size(H,1)
    if search_space == nothing
        V = ElasticMatrix(Matrix{eltype(H)}(I, N, 10) .+ 1e-3rand(eltype(H), N, 10))
    else
        V = ElasticMatrix(search_space)
    end
    HV = ElasticMatrix(H*V)

    k = size(V, 2)
    R = Ritz(NaN, Vector{Float64}(undef, N))

    r = fill(NaN, N)
    errors = Float64[]

    # if method == :davidson || method == :RQI
    #     P = DavidsonPreconditioner(diag(H), NaN)
    # elseif method == :JD
    #     P = JDPreconditioner(diag(H), NaN, fill(NaN, N))
    # end

    D = diagonal == nothing ? diag(H) : diagonal

    matvecs = 0
    while true

        while size(V, 2) < max_dimension

            ## diagonalize in subspace V
            rayleigh_ritz!(R, HV, V, which)

            ## compute residual vector r = Hx - θx
            mul!(r, H, R.vector);
            matvecs += 1
            
            axpy!(-R.value, R.vector, r); 

            ϵ = norm(r, Inf)
            push!(errors, ϵ)

            ϵ < tol && return (vector = R.vector, value = R.value, trace = errors, matvecs = matvecs)

            ## expand subspace
            if method == :davidson
                #P.value = R.value
                @show(R.value)
                r ./= D .- R.value

            elseif method == :olsen

                a = R.vector ./ (D .- R.value)
                b = r ./ (D .- R.value)
                r = (dot(R.vector, b)/dot(R.vector, a))*a - b

            elseif method == :RQI
                P.value = R.value
                bicgstabl!(r, H - R.value*I, R.vector; Pl = P)

            elseif method == :JD
                cg!(r, 
                LinearMap(y -> JD_map!(y, H, R.value, R.vector), N)
                , -r
                # ;
                # Pl = P,
                # rtol = 2.0^(-size(V, 2)), log = false
                )
            end

            append!(V, r)


            ## orthonormalize correction vector against current basis
            orthonorm!(V)
            append!(HV, H*V[:, end])
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
            V = ElasticMatrix(V*eigen(Hermitian(V'*HV)).vectors[:, end-(k-1):end])
        end
        HV = ElasticMatrix(H * V)
    end

end

