# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRProjPO

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVR Projection Partial Orthogonal -- Procedural" begin

    # Supertype
    @test supertype(LinSysVecRowProjPO) == LinSysVecRowProjection

    # Testing context
    Random.seed!(1010)

    A = rand(10,5)
    x = rand(5)
    b = A * x

    z = rand(5)

    rsub = LinSysVecRowProjPO()

    # Verify that Z (storage) is nothing
    @test isnothing(rsub.Z)

    # Verify that Z (storage) is appropriately initialized
    @test let
        RLinearAlgebra.rsubsolve!(rsub, z, (A[1,:], b[1]), 1)

        (length(rsub.Z) == rsub.m) & reduce(&, [length(z)==5 for z in rsub.Z])
    end

    # Verify that solution is orthogonalized against previous m search directions
    @test let
        for i = 1:6
            RLinearAlgebra.rsubsolve!(rsub, z, (A[i,:], b[i]), i)
        end

        norm( (A * z - b)[2:6] ) < 1e-15
    end

    @test let
        for i = 1:8
            RLinearAlgebra.rsubsolve!(rsub, z, (A[i,:], b[i]), i)
        end

        norm((A * z - b)[4:8]) < 1e-15
    end

end

end # End Module
