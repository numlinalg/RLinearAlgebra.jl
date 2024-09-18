# This file is part of RLinearAlgebra.jl
# Date: 07/19/2024
# Author: Christian Varner
# Purpose: Test the sampling procedure CountSketch from
# src/linear_samplers/block_col_count_sketch.jl

module ProceduralTestLSBCCountSketch

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBC Count Sketch -- Procedural" begin

    ## Verify appropriate super type
    @test supertype(LinSysBlkColCountSketch) <: LinSysVecColSelect
    
    # test first constructor
    sampler = LinSysBlkColCountSketch(10, nothing, nothing)
    @test sampler.block_size == 10
    @test isnothing(sampler.S)
    @test isnothing(sampler.signs)

    sampler = LinSysBlkColCountSketch(100, zeros(10, 10), zeros(10))
    @test sampler.block_size == 100
    @test sampler.S == zeros(10, 10)
    @test sampler.signs == zeros(10)

    # test second constructor
    sampler = LinSysBlkColCountSketch(225)
    @test sampler.block_size == 225
    @test isnothing(sampler.S) 
    @test isnothing(sampler.signs)

    # test default constructor
    sampler = LinSysBlkColCountSketch()
    @test sampler.block_size == 2
    @test isnothing(sampler.S)
    @test isnothing(sampler.signs)

    ## test procedure and output of method
    A = randn(5, 10)
    b = randn(5)

    # initialize sampler with block_size 6; 
    # Inputs to S and signs should be rewritten on the first iteration.
    block_size = 6
    sampler = LinSysBlkColCountSketch(block_size, ones(100, 100), ones(100))

    # initial iteration
    x0 = zeros(10)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 4
    S, AS, res, grad = sketch_result[1], sketch_result[2], sketch_result[3], sketch_result[4]

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (size(A)[2], block_size) 
    @test sampler.signs == [-1, 1]

    # test if sketch sizes are correct
    @test size(S) == (size(A)[2], block_size)
    @test size(AS) == (size(A)[1], block_size)

    # test if reported sketch matrix and sketched residual are correct
    @test norm(AS - A * S) < eps() * 1e2
    @test norm(res - (A * x0 - b)) < eps() * 1e2
    @test norm(grad - AS' * (A * x0 - b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S) == Matrix{Int64} # (size(A)[2], block_size)
    for i in 1:size(A)[2]
        # sum of elements in each row should be equal to 1.
        s = 0
        for j in 1:block_size
            s += abs(S[i, j])
        end

        @test s == 1 
    end

    # test if method throws an error for block_size <= 0 if iter == 1
    try
        sampler = LinSysBlkColCountSketch(0)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkColCountSketch(-1)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

end # End module

end