ncores = Threads.nthreads()/2

using DrWatson
@quickactivate

include(srcdir("benchmark.jl"))

N = 2^10
λ = 0.01
M = diagm(1:N) + λ * rand(N, N)


results = DataFrame(
    cores=[],
    time_eigen=[],
    time_ipt=[],
    time_ipt_acx=[],
)

for q in 0:Int(log2(ncores))

    BLAS.set_num_threads(2^q)

    result = [
        2^q,
        benchmark(M, N; method=:DIRECT).relative_time,
        benchmark(M, N; method=:IPT).relative_time,
        benchmark(M, N; method=:IPT_ACX).relative_time
    ]

    @show result
    push!(results, result)
end

Arrow.write(datadir("parallel"), results)