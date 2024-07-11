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
    sampler = LinSysBlkRowCountSketch(10, nothing, nothing, nothing)
    @test sampler.blockSize == 10
    @test isnothing(sampler.labels)
    @test isnothing(sampler.signs)
    @test isnothing(sampler.S)

    sampler = LinSysBlkRowCountSketch(10, zeros(5), zeros(6), zeros(10, 10))
    @test sampler.blockSize == 10
    @test sampler.labels == zeros(5)
    @test sampler.signs == zeros(6)
    @test sampler.S == zeros(10, 10)

    # test second constructor
    sampler = LinSysBlkRowCountSketch(225)
    @test sampler.blockSize == 225
    @test isnothing(sampler.labels)
    @test isnothing(sampler.signs)
    @test isnothing(sampler.S) 

    ## test procedure and output of method
    A = randn(10, 5)
    b = randn(10)

    # initialize sampler with blockSize 6
    sampler = LinSysBlkRowCountSketch(6)

    # initial iteration
    x0 = zeros(5)
    S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # test if buffer arrays are initialized to the correct sizes
    @test size(sampler.labels)[1] == 10
    @test size(sampler.signs)[1] == 10
    @test size(sampler.S) == (6, 10)

    # test if sketch sizes are correct
    @test size(S) == (6, 10)
    @test size(SA) == (6, 5)

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    for j in 1:10
        # sum elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:size(S)[1]
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