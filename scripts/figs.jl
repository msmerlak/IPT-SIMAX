using DrWatson
@quickactivate

using Arrow

morg = Arrow.Table(datadir("CPU", "morgan")) |> DataFrame

