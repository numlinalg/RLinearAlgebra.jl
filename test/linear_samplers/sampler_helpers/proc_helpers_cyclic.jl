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

    # Test assertion  about positive blocks
    blk.blocks = nothing
    blk.n_blocks = -1
    @test_throws AssertionError("Number of blocks must be positive") RLinearAlgebra.init_blocks_cyclic!(blk, 5)
    # Test warning  about too big block size
    blk.n_blocks = 6 
    @test_logs (:warn, "Setting `n_blocks` to be equal to the dimension of the system. No obvious way to set blocks when `n_blocks` is greater than dimension. If you would like to do so, create a `Vector{Vector{Int64}}` with each sub vector containing indices of a block.") RLinearAlgebra.init_blocks_cyclic!(blk, 5)
    # Test when missing indices in block 
    blk = LinSysBlkColRandCyclic(blocks = [[1], [3; 4]])
    @test_logs (:warn, "Indices $(Set(2)) are unused in your blocks") RLinearAlgebra.init_blocks_cyclic!(blk, 4)

end

end
