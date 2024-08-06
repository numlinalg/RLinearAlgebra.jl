# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard  

module ProceduralTestLSBRGaussSampler

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBR Gaussian Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkRowGaussSampler) == LinSysBlkRowSampler

    # Test construction
    A = rand(5,10)
    b = rand(5)
    x = rand(10)

    samp = LinSysBlkRowGaussSampler()
    for i in 1:5
        S, SA, res  = RLinearAlgebra.sample(samp, A, b, x, i)

        @test norm(SA - S * A) < eps() * 1e2
        @test norm(res - (S * A * x - S * b)) < eps() * 1e2
    end
    # Test blocksize must be greater than zero
    samp = LinSysBlkRowGaussSampler(-1)
    @test_throws "`block_size` must be positve." RLinearAlgebra.sample(samp, A, b, x, 1)
    samp = LinSysBlkRowGaussSampler(11)
    @test_warn "`block_size` should be less than row dimension" RLinearAlgebra.sample(samp, A, b, x, 1)
    
end

end
