# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBCRandCyclic

using Test, RLinearAlgebra, Random

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
        @test v.S[1] == block[(block_num - 1) * 2 + 1]
        @test v.S[2] == block[(block_num - 1) * 2 + 2]
    end

    c = rand(2)
    # Test multiplication
    @test (v * c)[v.S] == c
    @test A * v == A[:, v.S]

    # test three input mul!
    sA = ones(10,2)
    sb = ones(10)
    mul!(sb, v, c)
    mul!(sA, v, A)
    @test sb[v.S] == c
    @test sA == A[:, v.S]

    #test five input mul!
    mul!(sb, v, c, 2, 1)
    mul!(sA, v, A, 2, 1)
    @test sb[v.S] == 3 * c
    @test sA == 3 * A[:, v.S]


end

end
