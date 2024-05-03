# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBCProj

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSBC Projection -- Procedural" begin
    # Supertype and aliases
    @test supertype(LinSysBlkColProj) == LinSysBlkColProjection 
    @test BlockCoordinateDescent == LinSysBlkColProj 

    # Verify that the residual projection is zero
    Random.seed!(1010)

    A = rand(5,10)
    x = rand(10)
    b = A * x

    rsub = LinSysBlkColProj()
    for i = 1:5
        @test let
            # Initialization of iteration
            z = rand(10)
            zc = deepcopy(z)
            # Search Direction
            e = zeros(Int, 2)
            e[1] = (i - 1) * 2 + 1 
            e[2] = i * 2
            # Update
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A[:, e], A' * (A * z - b), A * z - b), i)
            #Comparison solver
            zc[e] -= A[:, e] \ (A * zc - b)
            norm(z - zc) < eps()*1e2
        end
    end
end
end # End module
