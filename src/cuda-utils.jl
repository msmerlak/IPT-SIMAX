using CUDA
import LinearAlgebra:mul!

function mul!(Y::CuArray, A::CuArray, B::CuArray)
    Y .= A * B
end