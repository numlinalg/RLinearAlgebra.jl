# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRProjFO

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVR Projection Full Orthogonal -- Procedural" begin

    # Supertype
    @test supertype(LinSysVecRowProjFO) == LinSysVecRowProjection

    # Testing context
    Random.seed!(1010)

    A = rand(10,5)
    x = rand(5)
    b = A * x

    z = rand(5)

    rsub = LinSysVecRowProjFO()

    # Verify that S (storage) is nothing
    @test isnothing(rsub.S)

    # Verify that S (storage) is appropriately initialized
    @test let
        RLinearAlgebra.rsubsolve!(rsub, z, (A[1,:], b[1]), 1)

        typeof(rsub.S) <: Matrix
    end

    # Verify that solution is orthogonalized against all previous hyperplanes
    @test let
        for i = 1:10
            RLinearAlgebra.rsubsolve!(rsub, z, (A[i,:], b[i]), i)
        end

        norm(A * z - b) < 1e-15
    end
end

end # End Module
