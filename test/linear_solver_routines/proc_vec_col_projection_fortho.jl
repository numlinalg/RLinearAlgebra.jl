# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCProjFO

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVC Projection Full Orthogonal -- Procedural" begin

    # Supertype
    @test supertype(LinSysVecColProjFO) == LinSysVecColProjection

    # Testing context
    Random.seed!(1010)

    A = rand(15,10)
    x = rand(10)
    b = A * x

    z = rand(10)

    rsub = LinSysVecColProjFO()

    # Verify that S (storage) is nothing
    @test isnothing(rsub.S)

    # Verify that S (storage) is appropriately initialized
    @test let
        e = zeros(10)
        e[1] = 1.0
        res = dot(A[:,1], A * z - b)
        RLinearAlgebra.rsubsolve!(rsub, z, (e, A, res), 1)
        typeof(rsub.S) <: Matrix
    end

    # Verify that solution is orthogonalized against all previous hyperplanes
    @test let
        for i = 1:13
            e = zeros(10)
            e[mod(i, 1:10)] = 1.0
            res = dot(A[:,mod(i, 1:10)], A * z - b)
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A, res), i)
        end
        
        norm(A' * (A * z - b) ) < 1e-11
    end
end

end # End Module
