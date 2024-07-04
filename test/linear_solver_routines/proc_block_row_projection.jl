# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard

module ProceduralTestLSBCLQ

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSBC Projection Block LQ -- Procedural" begin
    # Supertype and aliases
    @test supertype(LinSysBlkRowLQ) == LinSysBlkRowProjection 
    @test BlockKaczmarz == LinSysBlkRowLQ

    # Verify that the residual projection is zero
    Random.seed!(1010)

    A = rand(10,5)
    x = rand(5)
    b = A * x
    tr = zeros(5)
    rsub = LinSysBlkRowLQ()
    for i = 1:5
        @test let
            # Initialization of iteration
            z = rand(5)
            zc = deepcopy(z)
            # Search Direction
            e = zeros(Int, 2)
            e[1] = (i - 1) * 2 + 1 
            e[2] = i * 2
            # Update
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A[e,:], (A[e, :] * z - b[e])), i)
            #Comparison solver
            zc .-= A[e, :]' * pinv(A[e, :] * A[e, :]') * (A[e, :] * zc - b[e])
            norm(z - zc) < eps() * 1e2
        end
    end
end
end # End module
