# This file is part of RLinearAlgebra.jl
# Date: 07/16/2024
# Author: Christian Varner
# Purpose: Test the sampling procedure in the file
# src/linear_samplers/block_row_sample_wo_replacement.jl

module ProceduralTestLSBRWoReplacement

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "LSBR Without Replacement -- Procedural" begin

    #####################
    # test constructors #
    #####################
    # full constructor
    sampler = LinSysBlkRowSelectWoReplacement(100, nothing, nothing, nothing, nothing)
    @test sampler.block_size == 100
    @test isnothing(sampler.probability)
    @test isnothing(sampler.population)
    @test isnothing(sampler.rows_sampled)
    @test isnothing(sampler.S)

    # test full constructor with arbitrary arguments
    sampler = LinSysBlkRowSelectWoReplacement(100, [1., -1.], collect(1:2), ones(1000), ones(100,100))
    @test sampler.block_size == 100
    @test sampler.probability == [1., -1.]
    @test sampler.population == collect(1:2)
    @test sampler.rows_sampled == ones(1000)
    @test sampler.S == ones(100,100)

    # test default constructor
    sampler = LinSysBlkRowSelectWoReplacement()
    @test sampler.block_size == 2
    @test isnothing(sampler.probability) && isnothing(sampler.population) && isnothing(sampler.rows_sampled) && isnothing(sampler.S)

    # test keyword constructor
    sampler = LinSysBlkRowSelectWoReplacement(block_size=101, probability = [.5, .5])
    @test sampler.block_size == 101
    @test sampler.probability == [.5, .5]
    @test isnothing(sampler.population) && isnothing(sampler.rows_sampled) && isnothing(sampler.S)
    #####################
    #####################

    ####################
    ### test sampler ###
    ####################

    # problem context
    A = randn(100, 5)
    b = randn(100)
    x0 = zeros(5)

    # initialize sampler
    block_size = 10
    sampler = LinSysBlkRowSelectWoReplacement(block_size = block_size)
    
    # first iteration
    sample_sketch = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check initializations
    @test size(sampler.S) == (block_size, size(A)[1])
    @test size(sampler.rows_sampled) == (block_size, )
    @test sampler.probability == repeat([1/size(A)[1]], outer = size(A)[1]) && isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[1])

    # check output
    @test length(sample_sketch) == 3
    S, SA, res = sample_sketch[1], sample_sketch[2], sample_sketch[3]

    # check sketched matrix and residual is correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(SA - A[sampler.rows_sampled, :]) < eps() * 1e2
    @test norm(S * A - A[sampler.rows_sampled, :]) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # check that sketched matrix has structure
    flag_row = true
    for j in 1:size(S)[1]
        # each row should have only one 1 as this corresponds to selecting a row
        sum_to_one = sum(S[j, :]) == 1
        non_negative = sum(S[j, :] .>= 0) == size(A)[1]
        correct_position = (S[j, sampler.rows_sampled[j]] == 1)
        flag_row = flag_row && (sum_to_one && non_negative && correct_position)
    end
    @test flag_row

    flag_col = true
    for i in 1:size(S)[2]
        # each column should only have one 1 or all 0 as we sample w.o. replacement
        sum_to_one = sum(S[:, i]) == 1 || sum(S[:, i]) == 0
        non_negative = sum(S[:, i] .>= 0) == block_size
        flag_col = flag_col && (sum_to_one && non_negative)
    end
    @test flag_col

    # second iteration; test if resampled
    old_rows = copy(sampler.rows_sampled)
    sample_sketch_2 = RLinearAlgebra.sample(sampler, A, b, x0, 2)
    @test old_rows != sampler.rows_sampled

    # test changing weights
    probability = repeat([0.], outer = size(A)[1])
    probability[1:20] .= 1/20
    block_size = 20

    sampler = LinSysBlkRowSelectWoReplacement(block_size = block_size, probability = probability)
    
    # sample
    S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check probability initializations
    @test size(sampler.S) == (block_size, size(A)[1])
    @test size(sampler.rows_sampled) == (block_size, )
    @test isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[1])

    # check that no rows greater than 20 were sampled
    @test sum(sampler.rows_sampled .<= 20) == length(sampler.rows_sampled)

    # test error checking in keyword constructor

    # weights do not add to 1.
    try
        sampler = LinSysBlkRowSelectWoReplacement(block_size = 2, probability = [1., 1.])
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

    # weights are not non-negative
    try
        sampler = LinSysBlkRowSelectWoReplacement(block_size = 2, probability = [2., -1.])
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DomainError)
    end

    # weights do not form a valid distribution over rows
    try
        nrow = size(A)[1] - 50
        probability = Weights(repeat([1/nrow], outer = nrow))
        sampler = LinSysBlkRowSelectWoReplacement(block_size = 2, probability = probability)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test isa(e, DimensionMismatch)
    end

    # not enough non-zero probabilities
    try
        nrow = size(A)[1]
        probability = Weights(repeat([0.], outer = nrow))
        probability[1] = 1.
        sampler = LinSysBlkRowSelectWoReplacement(block_size = 2, probability = probability)
        S, SA, res = RLinearAlgebra.sample(sampler, A, b, x0, 1)
    catch e
        @test size(probability)[1] == size(A)[1] # ignores the first dim mismatch
        @test isa(e, DimensionMismatch)
    end

    # initialize sampler with arbitrary arguments 
    block_size = 10
    sampler = LinSysBlkRowSelectWoReplacement(block_size, nothing, collect(1:10000), collect(1:10000), ones(10000,10000))
    
    # first iteration
    sample_sketch = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check that buffer arrays were over-written with correct values
    @test size(sampler.S) == (block_size, size(A)[1])
    @test size(sampler.rows_sampled) == (block_size, )
    @test sampler.probability == repeat([1/size(A)[1]], outer = size(A)[1]) && isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[1])

    # check output
    @test length(sample_sketch) == 3
    S, SA, res = sample_sketch[1], sample_sketch[2], sample_sketch[3]

    # check sketched matrix and residual is correct
    @test norm(SA - S * A) < eps() * 1e2
    @test norm(SA - A[sampler.rows_sampled, :]) < eps() * 1e2
    @test norm(S * A - A[sampler.rows_sampled, :]) < eps() * 1e2
    @test norm(res - (S * A * x0 - S * b)) < eps() * 1e2

    # check that sketched matrix has structure
    flag_row = true
    for j in 1:size(S)[1]
        # each row should have only one 1 as this corresponds to selecting a row
        sum_to_one = sum(S[j, :]) == 1
        non_negative = sum(S[j, :] .>= 0) == size(A)[1]
        correct_position = (S[j, sampler.rows_sampled[j]] == 1)
        flag_row = flag_row && (sum_to_one && non_negative && correct_position)
    end
    @test flag_row

    flag_col = true
    for i in 1:size(S)[2]
        # each column should only have one 1 or all 0 as we sample w.o. replacement
        sum_to_one = sum(S[:, i]) == 1 || sum(S[:, i]) == 0
        non_negative = sum(S[:, i] .>= 0) == block_size
        flag_col = flag_col && (sum_to_one && non_negative)
    end
    @test flag_col

    ####################
    ####################
end

end # end module