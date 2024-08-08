# This file is part of RLinearAlgebra.jl
# This file was written by Nathaniel Pritchard  

module ProceduralTestLSBCRandCyclic

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBC Random Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkColRandCyclic) == LinSysBlkColSampler 

    # Test whether row ordering remains fixed
    A = rand(10,6)
    b = rand(10)
    x = rand(6)

    cyc = LinSysBlkColRandCyclic()

    v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, 1)

    order = deepcopy(cyc.order)
    block = deepcopy(cyc.blocks)
    for j = 2:length(cyc.order)
        v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        block_num = order[j]
        @test v == Matrix{Float64}(I, 6, 6)[:, block[block_num]]
    end

    # Test the reshuffling
    cyc = LinSysBlkColRandCyclic(n_blocks = 6)
    v, M, grad, res = RLinearAlgebra.sample(cyc, A, b, x, 1)
    order = deepcopy(cyc.order)
    RLinearAlgebra.sample(cyc, A, b, x, length(cyc.order) + 1)
    @test cyc.order != order

end

end
