using SparseArrays
using LinearAlgebra
using ElasticArrays

mutable struct Ritz{T}
    value::T
    vector::Vector{T}
end

# function rayleigh_ritz!(RR, A, V = nothing)
#     if V == nothing
#         V = Matrix{Float64}(I, size(A,1), 5)
#     end
#     X = eigen(Hermitian(V'*A*V))
#     return (vectors = V*X.vectors, values = X.values)
# end


function rayleigh_ritz!(R::Ritz, AV::AbstractMatrix, V::AbstractMatrix, which::Symbol)

    X = eigen(Hermitian(V'*AV))
    if which == :largest 
        mul!(R.vector, V, X.vectors[:, end])
        R.value = X.values[end]
    elseif which == :smallest
        mul!(R.vector, V, X.vectors[:, 1])
        R.value = X.values[1]
    end
end

function orthogonalize!(x::AbstractVector, y::AbstractVector)
    axpy!(-dot(y,x), x, x)
end

function orthonormalize!(V::Matrix)
    V .= qr(V).Q * Matrix(I, size(V)...)
end

function orthonormalize!(V::ElasticMatrix)
    V .= ElasticMatrix(qr(Matrix(V)).Q * Matrix(I, size(V)...))
end