function davidson2(A, SS::AbstractArray; maxiter=100,
                  tol=20size(A,2)*eps(eltype(A)),
                  maxsubspace=8size(SS, 2))
    m = size(SS, 2)
    prec 
    for i in 1:maxiter
        Ass = A * SS

        # Use eigen specialised for Hermitian matrices
        rvals, rvecs = eigen(Hermitian(SS' * Ass))
        rvals = rvals[1:m]
        rvecs = rvecs[:, 1:m]
        Ax = Ass * rvecs

        R = Ax - SS * rvecs * Diagonal(rvals)
        if norm(R) < tol
            return rvals, SS * rvecs
        end

        println(i, "  ", size(SS, 2), "  ", norm(R))

        # Use QR to orthogonalise the subspace.
        if size(SS, 2) + m > maxsubspace
            SS = typeof(R)(qr(hcat(SS * rvecs, prec * R)).Q)
        else
            SS = typeof(R)(qr(hcat(SS, prec * R)).Q)
        end
    end
    error("not converged.")
end