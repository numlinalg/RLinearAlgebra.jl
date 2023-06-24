# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRUnifSampler

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Uniform Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowUnifSampler) == LinSysVecRowSampler

    # Test construction
    @test let
        A = rand(10,3)
        b = rand(10)
        x = rand(3)

        samp = LinSysVecRowUnifSampler()

        α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

        true
    end

end

end
