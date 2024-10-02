# This file is part of RLinearAlgebra.jl

module ProceduralTestLSBCSRHT

using Test, RLinearAlgebra, Random, Hadamard

import LinearAlgebra: norm, Diagonal

Random.seed!(1010)

@testset "LSBC SRHT -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysBlockColSRHT) == LinSysBlkColSampler

    # Test whether col ordering remains fixed and padding required
    A = rand(16,6)
    b = rand(16)
    x = rand(6)
    Ap = zeros(16, 8)
    Ap[:, 1:6] .= A

    samp = LinSysBlockColSRHT()

    v, M, res, grad = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    H = hadamard(8)
    scaling = sqrt(2/8)
    for j = 2:5
        v, M, res, grad = RLinearAlgebra.sample(samp, A, b, x, j)
        sgn = samp.signs
        signs = [sgn[i] ? 1 : -1 for i in 1:8]  
        Ab = ((Ap * Diagonal(signs)) * H .* scaling)[:, samp.block]
        @test norm(grad - Ab' * (A * x - b)) < eps() * 1e2
    end


    # Test whether col ordering remains fixed and no padding required
    A = rand(16,8)
    b = rand(16)
    x = rand(8)
    Ap = zeros(16, 8)
    Ap[1:16, :] .= A

    samp = LinSysBlockColSRHT()

    v, M, res, grad = RLinearAlgebra.sample(samp, A, b, x, 1)
    
    H = hadamard(8)
    scaling = sqrt(2/8)
    for j = 2:5
        v, M, res, grad = RLinearAlgebra.sample(samp, A, b, x, j)
        sgn = samp.signs
        signs = [sgn[i] ? 1 : -1 for i in 1:8]  
        Ab = ((Ap * Diagonal(signs)) * H .* scaling)[:, samp.block]
        @test norm(grad - Ab' * (A * x - b)) < eps() * 1e2
    end

end

end
