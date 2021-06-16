# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRPropToNormSampler

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Strohmer-Vershynin Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowPropToNormSampler) == LinSysVecRowSampler

    # Verify alias
    @test LinSysVecRowPropToNormSampler == LinSysVecRowSVSampler

    # Test construction
    @test let
        A = rand(10,3)
        b = rand(10)
        x = rand(3)

        samp = LinSysVecRowSVSampler()

        α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

        true
    end

end

end
