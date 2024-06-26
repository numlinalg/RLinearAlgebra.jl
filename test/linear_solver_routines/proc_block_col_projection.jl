# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard

module ProceduralTestLSBCGent

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSBC Gent Projection -- Procedural" begin
    # Supertype and aliases
    @test supertype(LinSysBlkColGent) == LinSysBlkColProjection 
    @test BlockCoordinateDescent == LinSysBlkColGent 

    # Verify that the residual projection is zero
    Random.seed!(1010)

    A = rand(5,10)
    x = rand(10)
    b = A * x

    rsub = LinSysBlkColGent()
    for i = 1:5
        @test let
            # Initialization of iteration
            z = rand(10)
            zc = deepcopy(z)
            # Search Direction
            e = zeros(Int, 10, 2)
            e[(i - 1) * 2 + 1, 1] = 1 
            e[i * 2, 2] = 1
            # Update
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A * e,  e' * A' * (A * z - b), A * z - b), i)
            #Comparison solver
            zc -= e * (A * e \ (A * zc - b))
            norm(z - zc) < eps() * 1e2
        end
    end
end
end # End module
