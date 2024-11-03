# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBRSRHT

using Test, RLinearAlgebra, Random, Hadamard

import LinearAlgebra: norm

Random.seed!(1010)

@testset "LSBR SRHT -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkRowSRHT) == LinSysBlkRowSampler

    # Test whether row ordering remains fixed and padding required
    A = rand(10,6)
    b = rand(10)
    x = rand(6)
    Ap = zeros(16, 6)
    bp = zeros(16)
    Ap[1:10, :] .= A
    bp[1:10] .= b

    samp = LinSysBlkRowSRHT()

    v, M, res = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    H = hadamard(16)
    scaling = sqrt(1/2)
    for j = 2:5
        v, M, res = RLinearAlgebra.sample(samp, A, b, x, j)
        Ab = v * Ap 
        bb = v * bp 
        @test norm(res - (Ab * x - bb)) < eps() * 1e2
    end


    # Test whether row ordering remains fixed and no padding required
    A = rand(16,6)
    b = rand(16)
    x = rand(6)
    Ap = zeros(16, 6)
    bp = zeros(16)
    Ap[1:16, :] .= A
    bp[1:16] .= b

    samp = LinSysBlkRowSRHT()

    v, M, res = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    H = hadamard(16)
    scaling = sqrt(1/2)
    for j = 2:5
        v, M, res = RLinearAlgebra.sample(samp, A, b, x, j)
        Ab = v * Ap
        bb = v * bp 
        @test norm(res - (Ab * x - bb)) < eps() * 1e2
    end

end

end