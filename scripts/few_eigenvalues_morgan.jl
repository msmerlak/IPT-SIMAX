using DrWatson
@quickactivate

using Plots;
gr(dpi=500, xtickfont="Computer Modern", ytickfont="Computer Modern", guidefontfamily="Computer Modern");

include(srcdir("benchmark.jl"))

morgan(N) = spdiagm(0 => 1:N, 1 => fill(0.5, N - 1), -1 => fill(0.5, N - 1))

M = morgan(5000);
TOL = 1e-12
nevs = [1, 5, 10, 20, 50]

few_eigenvalues = Dict()

for method in some_methods
    @show method
    try
        few_eigenvalues[method] = [benchmark(M, nev; method=method, tol=TOL) for nev in nevs]
    catch e
        println(e)
    end
end

# Time vs number of eigenvalues requested

plot(
    color_palette=palette(:tab10),
    legend=:topleft,
    xlabel="Number of eigenvalues computed",
    ylabel="Time (s)"
)

for method in some_methods
    try
        plot!(
            nevs,
            [b.absolute_time for b in few_eigenvalues[method]],
            label=method == :PRIMME_LOBPCG_OrthoBasis ? "PRIMME_LOBPCG" : String(method),
            markers=:auto,
            xticks=nevs
        )
    catch e
    end
end
current()
savefig(plotsdir("morgan_time_vs_nev"))

### Residual history vs matvecs

plot(
    yaxis=:log,
    color_palette=palette(:tab10),
    legend=:topright,
    xlabel="Number of matrix-vector products",
    ylabel="Residual norm (one eigenpair)"
)

for method in some_methods
    try
        plot!(
            few_eigenvalues[method][1].matvecs,
            few_eigenvalues[method][1].trace,
            label=method == :PRIMME_LOBPCG_OrthoBasis ? "PRIMME_LOBPCG" : String(method),
            markers=:auto
        )
    catch e

    end
end
current()
savefig(plotsdir("morgan_residual_vs_matvecs"))


### Relative times
bar(
    color_palette=palette(:tab10),
    legend=false,
    ylabel="Time(one eigenpair) / Time(one matvec)"
)
for method in some_methods
    try
    bar!(
        [method == :PRIMME_LOBPCG_OrthoBasis ? "PRIMME_LOBPCG" : String(method)],
        [few_eigenvalues[method][1].relative_time],
        xtickfontsize=5
    )
    catch e
    end
end
current()
savefig(plotsdir("morgan_time_vs_method"))


####

morgan(N, ε) = spdiagm(0 => 1:N, 1 => fill(0.5, N - 1), -1 => fill(ε, N - 1))

N= 5000
TOL = 1e-12

plot(
    yaxis=:log,
    color_palette=palette(:tab10),
    legend=:topright,
    xlabel="Number of matrix-vector products",
    ylabel="Residual norm (one eigenpair)"
)

i = 0
for ε ∈ (.1, .5, 1., 5., 10.)
    @show ε
    i += 1

    plot!([], 
    color = palette(:default)[i],
    label =  "ε = $ε")
    M = morgan(5000, ε)
    try 
        z = ipt(M, 1; trace = true, acceleration = :none)
        plot!(z.matvecs,
        z.trace, 
        markers = :cross,
        color = palette(:default)[i],
        label = false
        )
    catch e
        print(e)
    end

    try
        z = ipt(M, 1; trace = true, acceleration = :acx)
        plot!(
        z.matvecs,
        z.trace, 
        markers = :circle,
        color = palette(:default)[i],
        label = false
        )
    catch e
        print(e)
    end

end
current()
savefig(plotsdir("residual_vs_eps"))