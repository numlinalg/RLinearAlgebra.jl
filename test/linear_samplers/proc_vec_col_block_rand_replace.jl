# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBCReplace

using Test, RLinearAlgebra, Random

import LinearAlgebra: norm

Random.seed!(1010)

@testset "LSBC Random Sampling with Replacement -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkColReplace) == LinSysBlkColSampler

    # Test whether row ordering remains fixed
    A = rand(10,6)
    b = rand(10)
    x = rand(6)

    cyc = LinSysBlkColReplace()

    v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    for j = 2:3
        v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        @test norm(grad - A[:, v]'res) < eps() * 1e2
    end


end

end
