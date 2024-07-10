# This file is part of RLinearAlgebra.jl

module ProceduralTestCyclicIntializer

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "Cyclic Intializer -- Procedural" begin

    # Test case when block size equally divides indices
    blk = LinSysBlkColRandCyclic()
    RLinearAlgebra.init_blocks_cyclic!(blk, 4)
    @test [[1; 2], [3; 4]] == blk.blocks

    # Test case when blocks size specified but no equal division
    blk = LinSysBlkColRandCyclic()
    RLinearAlgebra.init_blocks_cyclic!(blk, 5)
    @test [[1; 2], [3; 4; 5]] == blk.blocks

    # Test that nblocks changes when blocks are prespecified
    blk = LinSysBlkColRandCyclic(blocks = [[1; 2], [3; 4], [5]])
    RLinearAlgebra.init_blocks_cyclic!(blk, 5)
    @test blk.n_blocks == 3
end

end
