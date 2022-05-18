using DrWatson
@quickactivate

using DataFrames, JDF

include(srcdir("benchmark_CUDA.jl"))

println("Run on $(collect(CUDA.devices())[1]) using CUSOLVER $(CUSOLVER.version()).")

den(N, η) = diagm(1:N) + η * rand(N, N)

results = DataFrame(
    N=[],
    method=[],
    eigenvalue=[],
    residual=[],
    time=[],
)

symmetric = true
dense = true 

for ν in 6:12

    N = 2^ν
    a = den(N, .1)
    a = (a + a') / 2
    A = cu(a);

    benchmark_CUDA!(results, A)
end

JDF.save(datadir("timings_GPU.csv"), results)

