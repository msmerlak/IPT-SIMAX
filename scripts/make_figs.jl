using DrWatson
@quickactivate

using Plots, StatsPlots
gr(dpi=500, xtickfont="Computer Modern", ytickfont="Computer Modern", guidefontfamily="Computer Modern");
using LaTeXStrings

using Arrow, DataFrames, DataFramesMeta, StatsPlots, JLD

### parallel scaling

parallel = DataFrame( Arrow.Table(datadir("parallel")) )

plot(
    color_palette=palette(:tab10),
    markers = :auto,
    xlabel = "Number of CPU cores",
    ylabelfontsize = 8,
    ylabel = "Time(eigendecomposition) / Time(matrix-matrix product)",
    legend = :topleft
)

plot!(parallel.cores, parallel.time_ipt, markers = :auto, label = "IPT")
plot!(parallel.cores, parallel.time_ipt_acx, markers = :auto, label = "IPT_ACX")
plot!(parallel.cores, parallel.time_eigen, markers = :auto, label = "LAPACK (DGEEV)")

savefig(plotsdir("parallel"))


### GPU: time vs size

GPU_T_vs_N = DataFrame( Arrow.Table(datadir("GPU_T_vs_N")) )

### CPU: time vs size

CPU_T_vs_N = DataFrame( Arrow.Table(datadir("CPU_T_vs_N")) )

plot(
    color_palette=palette(:tab10),
    markers = :auto,
    xlabel = L"N",
    ylabelfontsize = 8,
    ylabel = "Time for eigendecomposition (s)",
    legend = :topleft,
    xaxis = :log,
    yaxis = :log,
    xticks = (Int.(2.0.^(5:14)), Int.(2.0.^(5:14))),
    yticks = (10.0.^(-4:1), 10.0.^(-4:1)),
)

@df @subset(CPU_T_vs_N, :symm .== true) plot!(
    :N, :time_ipt, 
    markershape = :cross,
    label = "IPT (CPU)"
    )
@df @subset(CPU_T_vs_N, :symm .== true) plot!(
        :N, :time_eigen, 
        markershape = :square,
        color = palette(:tab10)[5],
        label = "SYEVD (CPU: LAPACK)"
        )
@df @subset(CPU_T_vs_N, :symm .== false) plot!(
        :N, :time_eigen, 
        markershape = :circle,
        label = "GEEV (CPU: LAPACK)"
        )
@df GPU_T_vs_N plot!(
        :N, :time_syevd, 
        markershape = :square,
        linestyle = :dash,
        color = palette(:tab10)[5],
        label = "SYEVD (GPU: CUSOLVER)"
        )
        @df GPU_T_vs_N plot!(
            :N, :time_ipt, 
            markershape = :cross,
            linestyle = :dash,
            color = palette(:tab10)[1],
            label = "IPT (GPU)"
            )
savefig(plotsdir("timings"))
### few eigenvalues