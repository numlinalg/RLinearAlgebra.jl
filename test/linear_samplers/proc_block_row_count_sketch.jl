# This file is part of RLinearAlgebra.jl
# Date: 07/08/2024
# Author: Christian Varner
# Purpose: Test the sampling procedure CountSketch from
# src/linear_samplers/block_row_count_sketch.jl

module ProceduralTestLSBRCountSketch

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBR Count Sketch -- Procedural" begin

    ## Verify appropriate super type
    @test supertype(LinSysBlkRowCountSketch) <: LinSysVecRowSelect
    
    # test first constructor
    sampler = LinSysBlkRowCountSketch(10, nothing, nothing)
    @test sampler.blockSize == 10
    @test isnothing(sampler.S)
    @test isnothing(sampler.signs)

    sampler = LinSysBlkRowCountSketch(100, zeros(10, 10), zeros(10))
    @test sampler.blockSize == 100
    @test sampler.S == zeros(10, 10)
    @test sampler.signs == zeros(10)

    # test second constructor
    sampler = LinSysBlkRowCountSketch(225)
    @test sampler.blockSize == 225
    @test isnothing(sampler.S) 
    @test isnothing(sampler.signs)

    ## test procedure and output of method
    A = randn(10, 5)
    b = randn(10)

    # initialize sampler with blockSize 6
    blockSize = 6
    sampler = LinSysBlkRowCountSketch(blockSize, ones(100, 100), ones(100))

    # initial iteration
    x0 = zeros(5)
    S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (blockSize, size(A)[1])
    @test sampler.signs == [-1, 1]

    # test if sketch sizes are correct
    @test size(S) == (blockSize, size(A)[1])
    @test size(SA) == (blockSize, size(A)[2])

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S) == Matrix{Int64}
    for j in 1:10
        # sum of elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:6
            s += abs(S[i, j])
        end

        @test s == 1 
    end

    # test if method throws an error for blockSize <= 0 if iter == 1
    try
        sampler = LinSysBlkRowCountSketch(0)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkRowCountSketch(-1)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

end # End module

end