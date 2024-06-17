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

    cyc = LinSysBlkRowReplace()

    v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    for j = 2:5
        v, M, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        @test norm(res - (A[v, :] * x - b[v])) < eps() * 1e2
    end


end

end
