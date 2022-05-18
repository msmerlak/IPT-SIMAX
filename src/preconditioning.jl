using Preconditioners
import LinearAlgebra:ldiv!

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

function ldiv!(y::AbstractVector, P::DiagonalPreconditioner, x::AbstractVector)
    y .= x./P.D
end