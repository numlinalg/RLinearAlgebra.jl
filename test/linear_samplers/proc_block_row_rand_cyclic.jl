# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBRRandCyclic

using Test, RLinearAlgebra, Random

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

    order = deepcopy(cyc.order)
    block = deepcopy(cyc.blocks)
    for j = 2:length(cyc.order)
        v, M, res = RLinearAlgebra.sample(cyc, A, b, x, j)
        block_num = order[j]
        @test v.S[1] == block[(block_num - 1) * 2 + 1]
        @test v.S[2] == block[(block_num - 1) * 2 + 2]
    end
    # Test multiplication
    @test v * b == b[v.S]
    @test v * A == A[v.S, :]

    # test three input mul!
    sA = zeros(2,6)
    sb = zeros(2)
    mul!(sb, v, b)
    mul!(sA, v, A)
    @test sb == b[v.S]
    @test sA == A[v.S, :]

    #test five input mul!
    mul!(sb, v, b, 2, 1)
    mul!(sA, v, A, 2, 1)
    @test sb == 3 * b[v.S]
    @test sA == 3 * A[v.S, :]
    
end

end
