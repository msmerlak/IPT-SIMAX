using LinearAlgebra: mul!, qr, I, Hermitian, eigen, axpy!, dot
using ElasticArrays

mutable struct Ritz{T}
    value::T
    vector::Vector{T}
end

function rayleigh_ritz!(R::Ritz, AV::AbstractMatrix, V::AbstractMatrix, which::Symbol)
    X = eigen(Symmetric(V'*AV))
    if which == :largest 
        mul!(R.vector, V, X.vectors[:, end])
        R.value = X.values[end]
    elseif which == :smallest
        mul!(R.vector, V, X.vectors[:, 1])
        R.value = X.values[1]
    end
end

function orthogon!(x::AbstractVector, y::AbstractVector)
    axpy!(-dot(y,x), x, x)
end

function orthonorm!(V::Matrix)
    V .= qr(V).Q * Matrix(I, size(V)...)
end

function orthonorm!(V::ElasticMatrix)
    
    V .= ElasticMatrix(qr(Matrix(V)).Q * Matrix(I, size(V)...))
end

function orthonorm2!(w::AbstractVector, V::ElasticMatrix)
    # Orthogonalize using BLAS-1 ops and column views.
    for i = 1 : size(V, 2)
        column = view(V, :, i)
        BLAS.axpy!(-dot(column, w), column, w)
    end

    nrm = norm(w)
    lmul!(one(eltype(w)) / nrm, w)

end
