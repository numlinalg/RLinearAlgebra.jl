# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBRFJLT

using Test, RLinearAlgebra, Random, Hadamard

import LinearAlgebra: norm

Random.seed!(1010)

@testset "LSBR FJLT -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlkRowFJLT) == LinSysBlkRowSampler

    # Test whether row ordering remains fixed and padding required
    A = rand(10,6)
    b = rand(10)
    x = rand(6)
    Ap = zeros(16, 6)
    bp = zeros(16)
    Ap[1:10, :] .= A
    bp[1:10] .= b

    samp = LinSysBlkRowFJLT()

    S, M, res = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    for j = 2:5
        S, M, res = RLinearAlgebra.sample(samp, A, b, x, j)
        Ab = S * Ap 
        bb = S * bp 
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

    samp = LinSysBlkRowFJLT()

    S, M, res = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    for j = 2:5
        S, M, res = RLinearAlgebra.sample(samp, A, b, x, j)
        Ab = S * Ap 
        bb = S * bp 
        @test norm(res - (Ab * x - bb)) < eps() * 1e2
    end

end

end