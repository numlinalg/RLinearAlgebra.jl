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
    
    # test first constructor -- S is nothing
    sampler = LinSysBlkRowCountSketch(10, nothing)
    @test sampler.block_size == 10
    @test isnothing(sampler.S)

    # test first constructor -- S is randomly initialized
    S = rand(collect(1:1000)) * ones(10, 10)
    sampler = LinSysBlkRowCountSketch(100, S)
    @test sampler.block_size == 100
    @test sampler.S == S

    # test second constructor -- default value
    sampler = LinSysBlkRowCountSketch()
    @test sampler.block_size == 2
    @test isnothing(sampler.S)

    # test second constructor -- special value
    sampler = LinSysBlkRowCountSketch(225)
    @test sampler.block_size == 225
    @test isnothing(sampler.S) 

    ## test procedure and output of method
    A = randn(10, 5)
    b = randn(10)

    ############################################
    # Test sampler with random initialized S
    ############################################

    # initialize sampler with block_size 6; Random initialized S -- should be overwritten
    block_size = 6
    S = rand(collect(1:1000)) * ones(10, 10)
    sampler = LinSysBlkRowCountSketch(block_size, S)

    # initial iteration
    x0 = zeros(5)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 3
    S, SA, res = sketch_result[1], sketch_result[2], sketch_result[3]
    S1 = zeros(block_size, size(A)[1])
    S1 .= S

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (block_size, size(A)[1])

    # test if sketch sizes are correct
    @test size(S) == (block_size, size(A)[1])
    @test size(SA) == (block_size, size(A)[2])

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S) == Matrix{Int64} # (size(A)[2], block_size)
    for j in 1:10
        # sum of elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:6
            s += abs(S[i, j])
        end

        @test s == 1 
    end
    
    # second iteration
    x0 = zeros(5)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 3
    S2, SA2, res2 = sketch_result[1], sketch_result[2], sketch_result[3]

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (block_size, size(A)[1])

    # test if sketch sizes are correct
    @test size(S2) == (block_size, size(A)[1])
    @test size(SA2) == (block_size, size(A)[2])

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA2 - S2 * A) < eps() * 1e2
    @test norm(res2 - (S2 * A * x0 - S2 * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S2) == Matrix{Int64} # (size(A)[2], block_size)
    for j in 1:10
        # sum of elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:6
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
    sampler = LinSysBlkRowCountSketch(block_size)

    # initial iteration
    x0 = zeros(5)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 3
    S, SA, res = sketch_result[1], sketch_result[2], sketch_result[3]
    S1 = zeros(block_size, size(A)[1])
    S1 .= S

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (block_size, size(A)[1])

    # test if sketch sizes are correct
    @test size(S) == (block_size, size(A)[1])
    @test size(SA) == (block_size, size(A)[2])

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S) == Matrix{Int64} # (size(A)[2], block_size)
    for j in 1:10
        # sum of elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:6
            s += abs(S[i, j])
        end

        @test s == 1 
    end

    # second iteration
    x0 = zeros(5)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that return is correct length
    @test length(sketch_result) == 3
    S2, SA2, res2 = sketch_result[1], sketch_result[2], sketch_result[3]

    # test if buffer arrays are (re)-initialized to the correct sizes and with correct values
    @test size(sampler.S) == (block_size, size(A)[1])

    # test if sketch sizes are correct
    @test size(S2) == (block_size, size(A)[1])
    @test size(SA2) == (block_size, size(A)[2])

    # test if reported sketch matrix and sketched residual are correct
    @test norm(SA2 - S2 * A) < eps() * 1e2
    @test norm(res2 - (S2 * A * x0 - S2 * b)) < eps() * 1e2

    # Test if returned sketch matrix has correct structure
    @test typeof(S2) == Matrix{Int64} # (size(A)[2], block_size)
    for j in 1:10
        # sum of elements in each column should be one (corresponding to one label for each row of `A`)
        s = 0
        for i in 1:6
            s += abs(S2[i, j])
        end

        @test s == 1 
    end

    # make sure we generated a new matrix
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
        sampler = LinSysBlkRowCountSketch(0)
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkRowCountSketch(-1)
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
        sampler = LinSysBlkRowCountSketch(0, zeros(10,10))
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkRowCountSketch(0)
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkRowCountSketch(-1, zeros(10,10))
        sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    catch e
        @test isa(e, DomainError)
    end

    try
        sampler = LinSysBlkRowCountSketch(-1)
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
    sampler = LinSysBlkRowCountSketch(100)
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    @test_logs (:warn, "block_size is greater than the number of columns in A!")

    # test that warning is thrown when block size is larger than
    # number of columns in A when using default constructor
    sampler = LinSysBlkRowCountSketch(100, zeros(10, 10))
    sketch_result = RLinearAlgebra.sample(sampler, A, b, x0, 1) 
    @test_logs (:warn, "block_size is greater than the number of columns in A!")
    #############################################
    # Check for warning messages 
    #############################################
    
end # End module

end