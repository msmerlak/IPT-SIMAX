include("linalg.jl")

using IterativeSolvers
using ElasticArrays
using Preconditioners
using LinearMaps

mutable struct DavidsonPreconditioner <: Preconditioners.AbstractPreconditioner
    diagonal::Vector{Float64}
    value::Float64
end

function ldiv!(P::DavidsonPreconditioner, x::AbstractArray)
    x ./= P.diagonal .- P.value
end
function ldiv!(y::AbstractArray, P::DavidsonPreconditioner, x::AbstractArray)
    y .= x./(P.diagonal .- P.value)
end

mutable struct JDPreconditioner <: Preconditioners.AbstractPreconditioner
    diagonal::Vector{Float64}
    value::Float64
    vector::Vector{Float64}
end

function ldiv!(P::JDPreconditioner, x::AbstractVector)
    orthogonalize!(x, P.vector)
    x ./= P.diagonal - P.value
    orthogonalize!(x, P.vector)
end


function davidson_method(
    H::Hermitian, 
    which = :smallest;
    method = :davidson,
    search_space = nothing, 
    max_dimension = 50,
    tol = 1e-12)

    N = size(H,1)
    if search_space == nothing
        V = ElasticMatrix(Matrix{Float64}(I, N, 2))
        HV = ElasticMatrix(H*V)
    end

    k = size(V, 2)
    R = Ritz(NaN, Vector{Float64}(undef, N))

    r = fill(NaN, N)
    errors = Float64[]

    if method == :davidson || method == :RQI
        P = DavidsonPreconditioner(diag(H), NaN)
    elseif method == :JD
        P = JDPreconditioner(diag(H), NaN, fill(NaN, N))
    end

    D = diag(H)

    while true

        while size(V, 2) < max_dimension

            ## diagonalize in subspace V
            rayleigh_ritz!(R, HV, V, which)
            ## compute residual vector r = Hx - θx
            mul!(r, H, R.vector);
            axpy!(-R.value, R.vector, r); 

            ϵ = norm(r, Inf)
            push!(errors, ϵ)

            ϵ < tol && return (vector = R.vector, value = R.value, errors = errors)

            ## expand subspace
            if method == :davidson
                P.value = R.value
                ldiv!(P, r)
                #t = r./(D .- R.value)
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
            orthonormalize!(V)
            append!(HV, H*V[:, end])

        end

        ## restart
        println("restarting")

        if which == :largest
            V = V*eigen(Hermitian(V'*HV)).vectors[:, 1:k]
        else
            V = V*eigen(Hermitian(V'*HV)).vectors[:, end-(k-1):end]
        end
        HV = H * V
    end

end


function JD_map!(y, A, θ, x)
    ## y = (I - xx')(A - θI)(I - xx')
    orthogonalize!(y, x)
    mul!(y, A, x)
    axpy!(-θ, x, y)
    orthogonalize!(y, x)
end


function JD_correction(H, θ, x)
    r = H*x - θ*x
    @show r
    t = ([H - θ*I x; x' 0.]\[-(I - x*x')*r; 0.])
    @show (I - x*x')*(H - θ*I)*(I - x*x')*t
end