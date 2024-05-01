# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCBlockGaussSampler

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSVC Block Gaussian Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecColBlockGaussian) == LinSysVecColSampler

    # Test construction
    A = rand(5,10)
    b = rand(5)
    x = rand(10)

    samp = LinSysVecColBlockGaussian()

    S, AS, grad, res  = RLinearAlgebra.sample(samp, A, b, x, 1)

    @test norm(AS - A * S) < eps() * 1e2
    @test norm(res - (A * x - b)) < eps() * 1e2
    @test norm(grad - S' * A' * (A * x - b)) < eps() * 1e2


end

end
