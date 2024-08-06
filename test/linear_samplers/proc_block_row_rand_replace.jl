# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard  

module ProceduralTestLSBRReplace

using Test, RLinearAlgebra, Random

import LinearAlgebra: norm

Random.seed!(1010)

@testset "LSBR Random Sampling with Replacement -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkRowReplace) == LinSysBlkRowSampler

    # Test whether row ordering remains fixed
    A = rand(10,6)
    b = rand(10)
    x = rand(6)

    # Test assertions
    # Positivity test
    cyc = LinSysBlkRowReplace(block_size=-12)
    @test_throws "`block_size` must be positive" RLinearAlgebra.sample(cyc, A, b, x, 1)
    # Less than matrix size test
    cyc = LinSysBlkRowReplace(block_size=12)
    @test_throws "`block_size` must be less than row dimension" RLinearAlgebra.sample(cyc, A, b, x, 1)


    cyc = LinSysBlkRowReplace()
    
    v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    for j = 2:5
        v, M, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        @test norm(res - (v * A * x - v * b)) < eps() * 1e2
    end

end

end
