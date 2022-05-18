using DrWatson, Revise
include(srcdir("IterativePerturbationTheory.jl"))
include(srcdir("DavidsonMethods.jl"))

using Preconditioners


function eig(M, p, smallest = true; method)
    
    N = size(M, 1) 
    P = DiagonalPreconditioner(M)

    if method == :ipt
        
        return ipt(M, p)

    elseif method == :lobpcg

        z = LOBPCG(M, typeof(M)(I, N, p), I, P, TOL)
        return (values = z.λ, vectors = z.X, trace  = z.residual_history, matvecs = z.n_matvec)

    elseif method == :ks

        vals, vecs, info = eigsolve(M, p, smallest ? :SR : :LR)
        return (values = vals[1:p], vectors = vecs[1:p], matvecs = info.numops)

    end


end
