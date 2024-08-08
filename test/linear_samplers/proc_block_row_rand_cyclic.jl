# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard  

module ProceduralTestLSBRRandCyclic

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBR Random Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkRowRandCyclic) == LinSysBlkRowSampler

    # Test whether row ordering remains fixed
    A = rand(10,6)
    b = rand(10)
    x = rand(6)

    cyc = LinSysBlkRowRandCyclic()

    v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    order = copy(cyc.order)
    block = copy(cyc.blocks)
    for j = 2:length(cyc.order)
        v, M, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        block_num = order[j]
        @test v == Matrix{Float64}(I, 10, 10)[block[block_num], :]
    end

    # Test the reshuffling
    cyc = LinSysBlkRowRandCyclic(n_blocks = 8)
    v, M, res = RLinearAlgebra.sample(cyc, A, b, x, 1)
    order = deepcopy(cyc.order)
    RLinearAlgebra.sample(cyc, A, b, x, length(cyc.order) + 1)
    @test cyc.order != order

end

end
