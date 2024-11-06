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
    
    # test first constructor -- S is nothing
    sampler = LinSysBlkColCountSketch(10, nothing)
    @test sampler.block_size == 10
    @test isnothing(sampler.S)

    # test first constructor -- S is randomly initialized
    S = rand(collect(1:1000)) * ones(10, 10)
    sampler = LinSysBlkColCountSketch(100, S)
    @test sampler.block_size == 100
    @test sampler.S == S

    # test second constructor -- default value
    sampler = LinSysBlkColCountSketch()
    @test sampler.block_size == 2
    @test isnothing(sampler.S)

    # test second constructor -- special value
    sampler = LinSysBlkColCountSketch(225)
    @test sampler.block_size == 225
    @test isnothing(sampler.S) 

    ## test procedure and output of method
    A = randn(5, 10)
    b = randn(5)

    ############################################
    # Test sampler with random initialized S
    ############################################

    # initialize sampler with block_size 6; Random initialized S -- should be overwritten
    block_size = 6
    S = rand(collect(1:1000)) * ones(10, 10)
    sampler = LinSysBlkColCountSketch(block_size, S)

    # initial iteration
    x0 = zeros(10)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 4
    S, AS, res, grad = sketch_result[1], sketch_result[2], sketch_result[3], sketch_result[4]
    S1 = zeros(size(A)[2], block_size)
    S1 .= S

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (size(A)[2], block_size) 

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

    # second iteration
    x0 = zeros(10)
    S2, AS2, res2, grad2 = RLinearAlgebra.sample(sampler, A, b, x0, 2)

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (size(A)[2], block_size) 

    # test if sketch sizes are correct
    @test size(S2) == (size(A)[2], block_size)
    @test size(AS2) == (size(A)[1], block_size)

    # test if reported sketch matrix and sketched residual are correct
    @test norm(AS2 - A * S2) < eps() * 1e2
    @test norm(res2 - (A * x0 - b)) < eps() * 1e2
    @test norm(grad2 - AS2' * (A * x0 - b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S2) == Matrix{Int64} # (size(A)[2], block_size)
    for i in 1:size(A)[2]
        # sum of elements in each row should be equal to 1.
        s = 0
        for j in 1:block_size
            s += abs(S2[i, j])
        end

        @test s == 1 
    end

    # make sure we generated a new matrix
    @test S1 != S2
    ############################################
    # Test sampler with random initialized S
    ############################################


    ############################################
    # Test sampler with default S
    ############################################

    # initialize sampler with block_size 6; S is not initialized
    block_size = 6
    sampler = LinSysBlkColCountSketch(block_size)

    # initial iteration
    x0 = zeros(10)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 4
    S, AS, res, grad = sketch_result[1], sketch_result[2], sketch_result[3], sketch_result[4]
    S1 = zeros(size(A)[2], block_size)

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (size(A)[2], block_size) 

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

    # second iteration
    x0 = zeros(10)
    S2, AS2, res2, grad2 = RLinearAlgebra.sample(sampler, A, b, x0, 2)

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (size(A)[2], block_size) 

    # test if sketch sizes are correct
    @test size(S2) == (size(A)[2], block_size)
    @test size(AS2) == (size(A)[1], block_size)

    # test if reported sketch matrix and sketched residual are correct
    @test norm(AS2 - A * S2) < eps() * 1e2
    @test norm(res2 - (A * x0 - b)) < eps() * 1e2
    @test norm(grad2 - AS2' * (A * x0 - b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S2) == Matrix{Int64} # (size(A)[2], block_size)
    for i in 1:size(A)[2]
        # sum of elements in each row should be equal to 1.
        s = 0
        for j in 1:block_size
            s += abs(S2[i, j])
        end

        @test s == 1 
    end

    # make sure we generated new matrix
    @test S1 != S2
    ############################################
    # End: Test sampler with default S
    ############################################


    #############################################
    # Check for error messages -- Constructor
    #############################################
    # check if error is thrown when using the defined constructor
    # when block_size <= 0
    try
        sampler = LinSysBlkColCountSketch(0)
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkColCountSketch(-1)
    catch e
        @test isa(e, DomainError)
    end
    #############################################
    # End Check for error messages -- Constructor
    #############################################

    #############################################
    # Check for error messages -- first iteration
    #############################################
    # check if error is thrown when we use default constructor
    # when block_size <= 0
    try
        sampler = LinSysBlkColCountSketch(0, zeros(10,10))
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkColCountSketch(0)
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkColCountSketch(-1, zeros(10,10))
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkColCountSketch(-1)
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end
    #############################################
    # Check for error messages -- first iteration
    #############################################

    #############################################
    # Check for warning messages 
    #############################################

    # test that a warning is thrown when block size is larger than number
    # of columns in A when using defined constructor 
    sampler = LinSysBlkColCountSketch(100)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    @test_logs (:warn, "block_size is greater than the number of columns in A!")

    # test that warning is thrown when block size is larger than
    # number of columns in A when using default constructor
    sampler = LinSysBlkColCountSketch(100, zeros(10, 10))
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    @test_logs (:warn, "block_size is greater than the number of columns in A!")
    #############################################
    # Check for warning messages 
    #############################################
end # End module

end