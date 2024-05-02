# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCBlockRandCyclic

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVC Block Random Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecColBlockRandCyclic) == LinSysVecColSampler

    # Test whether row ordering remains fixed
    A = rand(10,6)
    b = rand(10)
    x = rand(6)

    cyc = LinSysVecColBlockRandCyclic()

    v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    order = copy(cyc.order)
    block = copy(cyc.blocks)
    for j = 2:length(cyc.order)
        v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        block_num = order[j]
        @test v[1] == block[(block_num - 1) * 2 + 1]
        @test v[2] == block[(block_num - 1) * 2 + 2]
    end


end

end
