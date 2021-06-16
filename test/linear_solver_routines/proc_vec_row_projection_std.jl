# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRProjStd

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVR Projection Standard -- Procedural" begin

    # Supertype and aliases
    @test supertype(LinSysVecRowProjStd) == LinSysVecRowProjection
    @test Kaczmarz == LinSysVecRowProjStd
    @test ART == LinSysVecRowProjStd

    # Verify that residual projection is zero
    Random.seed!(1010)

    A = rand(10,5)
    x = rand(5)
    b = A * x

    rsub = Kaczmarz()

    for i = 1:10
        @test let

            z = rand(5)
            RLinearAlgebra.rsubsolve!(rsub, z, (A[i,:], b[i]), i)

            dot(z, A[i,:]) ≈ b[i]
        end
    end

end

end # End module
