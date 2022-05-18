function davidson(A::AbstractMatrix, NTargetEigvals::Int=1, subspaceincrement::Int=8, maxsubspacesize::Int=500, ϑ::Float64=1e-8; preconditioner = I)
  k = subspaceincrement
  M = min(maxsubspacesize, size(A,2))
 
  V = Matrix{Float64}(I,size(A,1),k)
  eigvals = ones(NTargetEigvals) 
  for m in k:k:M
    V   = orthonormaliz(V)
 
    AV  = A * V
    T   = Symmetric(V' * AV)
    eig = eigen(T)                                                                                          
    
    w = [AV*y - λ*(V*y) for (λ,y) in zip(eig.values[1:k], eachcolumn(eig.vectors)[1:k])]
    V = [V w...]
 
    eigvalsN = eig.values[1:NTargetEigvals]
    eigvals  = norm(eigvalsN-eigvals)>=ϑ ? eigvalsN : return eig.values
  end
  error("Davidson did not converge in $maxsubspacesize-dimensional subspace")
end

function eachcolumn(V::AbstractMatrix)
    [V[:,i] for i in 1:size(V,2)]
  end

function orthonormaliz(V::AbstractMatrix)
    Q, _ = qr(V)
    return Q*Matrix{Float64}(I,size(V)...)
end