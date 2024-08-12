# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBCGaussSampler

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBC Gaussian Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkColGaussSampler) == LinSysBlkColSampler 

    # Test construction
    A = rand(5,10)
    b = rand(5)
    x = rand(10)

    samp = LinSysBlkColGaussSampler()
    for i in 1:5
        S, AS, grad, res  = RLinearAlgebra.sample(samp, A, b, x, i)

        @test norm(AS - A * S) < eps() * 1e2
        @test norm(res - (A * x - b)) < eps() * 1e2
        @test norm(grad - S' * A' * (A * x - b)) < eps() * 1e2
    end
    # Test warnings and assertions
    samp = LinSysBlkColGaussSampler(-1)
    @test_throws AssertionError("`block_size` must be positive.") RLinearAlgebra.sample(samp, A, b, x, 1)
    samp = LinSysBlkColGaussSampler(11)
    @test_logs (:warn, "`block_size` shoould be less than col dimension.") RLinearAlgebra.sample(samp, A, b, x, 1)

end

end
