# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCProjStd

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVC Projection Standard -- Procedural" begin
    # Supertype and aliases
    @test supertype(LinSysVecColProjStd) == LinSysVecColProjection
    @test CoordinateDescent == LinSysVecColProjStd
    @test GaussSeidel == LinSysVecColProjStd

    # Verify that the residual projection is zero
    Random.seed!(1010)

    A = rand(5,10)
    x = rand(10)
    b = A * x

    rsub = GaussSeidel()

    for i = 1:10
        @test let
            # Initialization of iteration
            z = rand(10)

            # Search Direction
            e = zeros(10)
            e[i] = 1.0

            # Update
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A, dot(A * e, A * z - b)), i)

            abs(dot(A[:,i], A * z - b)) < eps()*1e2
        end
    end
end
end # End module
