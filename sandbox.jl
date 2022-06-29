using Laplacians, SparseArrays
using IterativePerturbationTheory
using Distributions

g = hypercube(8)
N = size(g, 1)
Δ = lap(g)

h = 10000
d = sort(rand(Uniform(-h, h), N))
H = Δ + spdiagm(d);
ipt(H, 1)


vals = eigen(Matrix(H)).values

@time ipt(H, N; 
trace = true, 
acceleration = :acx, 
timed = true, 
diagonal = 1:N
).values ≈ vals

using LinearAlgebra
function Q(d) 
    q = (H - Diagonal(d))./(transpose(d) .- d)
    q[diagind(q)] .= 0.
    return norm(q)
end

using Optim
z = optimize(Q, rand(N), SimulatedAnnealing(),Optim.Options(iterations = 100,
                             store_trace = false,
                             show_trace = true))

