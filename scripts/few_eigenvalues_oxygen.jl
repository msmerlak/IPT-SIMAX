using DrWatson
@quickactivate

using Plots;
gr(dpi=500, xtickfont="Computer Modern", ytickfont="Computer Modern", guidefontfamily="Computer Modern");

include(srcdir("benchmark.jl"))
include(srcdir("matrices.jl"))


M = oxygen;
TOL = 1e-12


results = Dict()
for method in some_methods
    @show method
    try
        results[method] = benchmark(M, 1; method=method, tol=TOL)
    catch e
        print(e)
    end
end

### Residual history vs matvecs

plot(
    yaxis=:log,
    color_palette=palette(:tab10),
    legend=:topright,
    xlabel="Number of matrix-vector products",
    ylabel="Residual norm (one eigenpair)"
)

for method in some_methods
    plot!(
        results[method].matvecs,
        results[method].trace,
        label = method == :PRIMME_LOBPCG_OrthoBasis ? "PRIMME_LOBPCG" : String(method),
        markers=:auto
    )
end
current()
savefig(plotsdir("oxygen_residual_vs_matvecs"))


### Relative times
bar(
    color_palette=palette(:tab10),
legend = false,
ylabel= "Time(one eigenpair) / Time(one matvec)"
)

for method in some_methods
    bar!(
        [method == :PRIMME_LOBPCG_OrthoBasis ? "PRIMME_LOBPCG" : String(method)],
        [results[method].relative_time],
        xtickfontsize=5
    )
end
current()
savefig(plotsdir("oxygen_time_vs_method"))