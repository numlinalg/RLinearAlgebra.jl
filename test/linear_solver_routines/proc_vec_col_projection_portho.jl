# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCProjPO

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVC Projection Partial Orthogonal -- Procedural" begin

    # Supertype
    @test supertype(LinSysVecColProjPO) == LinSysVecColProjection

    # Testing context
    Random.seed!(1010)

    A = rand(5, 10)
    x = rand(10)
    b = A * x

    z = rand(10)

    rsub = LinSysVecColProjPO()

    # Verify that Z (storage) is nothing
    @test isnothing(rsub.Z)

    # Verify that Z (storage) is appropriately initialized
    @test let
        e = zeros(10)
        e[1] = 1.0
        res = dot(A[:,1], A * z - b)
        RLinearAlgebra.rsubsolve!(rsub, z, (e, A, res), 1)
        (length(rsub.Z) == rsub.m) & reduce(&, [length(z)==10 for z in rsub.Z])
    end

    # Verify that solution has zero residual against previous directions
    @test let
        for i = 1:6
            e = zeros(10)
            e[i] = 1.0
            res = dot(A[:,i], A * z - b)
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A, res), i)
        end

        norm( A[:,2:6]'*(A * z - b)) < eps()*1e3 #Ajdust for condition number
    end

    @test let
        for i = 1:8
            e = zeros(10)
            e[i] = 1.0
            res = dot(A[:,i], A * z - b)
            RLinearAlgebra.rsubsolve!(rsub, z, (e, A, res), i)
        end

        norm( A[:,4:8]'*(A * z - b)) < eps()*1e3 #Adjust for condition number
    end
end

end # End module
