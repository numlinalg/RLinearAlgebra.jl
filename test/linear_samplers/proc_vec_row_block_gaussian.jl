# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRBlockGaussSampler

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSVR Block Gaussian Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowBlockGaussian) == LinSysVecRowSampler

    # Test construction
    A = rand(5,10)
    b = rand(5)
    x = rand(10)

    samp = LinSysVecRowBlockGaussian()

    S, SA, res  = RLinearAlgebra.sample(samp, A, b, x, 1)

    @test norm(SA - S * A) < eps() * 1e2
    @test norm(res - (S * A * x - S * b)) < eps() * 1e2

end

end
