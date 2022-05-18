using DrWatson
@quickactivate

using LinearAlgebra, SparseArrays, DataFrames, Arrow

include(srcdir("benchmark.jl"))

processors = Sys.cpu_info()
system = "$(BLAS.get_num_threads()) x $(processors[1].model) with $(BLAS.get_config().loaded_libs[1])"

println(system)

den(N, η) = diagm(1:N) + η * rand(N, N)
spar(N, η, density) = spdiagm(1:N) + η * sprand(N, N, density)
morgan(N) = spdiagm(0 => 1:N, 1 => fill(0.5, N - 1), -1 => fill(0.5, N - 1))

results = DataFrame(
    N=[],
    η = [],
    symmetric=[],
    nev=[],
    density=[],
    method=[],
    eigenvalue=[],
    residuals=[],
    iterations = [],
    matvecs=[],
    time=[],
)

N = 5000

A = morgan(N)
for nev in Int[1, 10, N]
    params = (N = N, symmetric = true, η = missing, nev = nev)
    benchmark!(results, A, params)
    println(results)
end
Arrow.write(datadir("CPU", "morgan"), results)


results = DataFrame(
    N=[],
    η = [],
    symmetric=[],
    nev=[],
    density=[],
    method=[],
    eigenvalue=[],
    residuals=[],
    iterations = [],
    matvecs=[],
    time=[],
)

for ν in 8:13, symmetric in [true, false], dense in [true, false], η in [1e-3, 1e-2, 1e-1, 1.]

    N = 2^ν

    A = dense ? den(N, η) : spar(N, η, 10 / N)
    if symmetric
        A = (A + A') / 2
    end

    for nev in Int[1, 10, N]
        params = (N = N, symmetric = symmetric, η = η, nev = nev)
        benchmark!(results, A, params)
        println(results)
    end

end
Arrow.write(datadir("CPU", "random"), results)