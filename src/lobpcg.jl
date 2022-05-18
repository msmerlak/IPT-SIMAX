function lopbcg(M, k = 1; 
    initial_condition == nothing,
    P == nothing
    )

    @assert issymmetric(M) "Matrix is not symmetric"

    N = size(M, 1)
    T = eltype(M)

    if initial_condition == nothing
        X = typeof(M)(I, N, k)
    end
    
    Y = similar(X)
    R = Ritz(NaN, Vector{Float64}(undef, N))

    while true
        rayleigh_ritz!
    end
end