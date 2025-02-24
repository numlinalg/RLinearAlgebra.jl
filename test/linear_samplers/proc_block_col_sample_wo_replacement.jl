# This file is part of RLinearAlgebra.jl
# Date: 07/17/2024
# Author: Christian
# Purpose: Test the sampling procedure in
# src/linear_samplers/block_col_sample_wo_replacement.jl

module ProceduralTestLSBCWoReplacment

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "LSBC Without Replacement -- Procedural" begin

    @test supertype(LinSysBlkColSelectWoReplacement) == LinSysBlkColSampler

    #####################
    # test constructors #
    #####################
    # full constructor
    sampler = LinSysBlkColSelectWoReplacement(100, nothing, nothing, nothing, nothing)
    @test sampler.block_size == 100
    @test isnothing(sampler.probability)
    @test isnothing(sampler.population)
    @test isnothing(sampler.col_sampled)
    @test isnothing(sampler.S)

    # test full constructor with arbitrary arguments
    sampler = LinSysBlkColSelectWoReplacement(100, [1., -1.], collect(1:2), ones(1000), ones(100,100))
    @test sampler.block_size == 100
    @test sampler.probability == [1., -1.]
    @test sampler.population == collect(1:2)
    @test sampler.col_sampled == ones(1000)
    @test sampler.S == ones(100,100)

    # test default constructor
    sampler = LinSysBlkColSelectWoReplacement()
    @test sampler.block_size == 2
    @test isnothing(sampler.probability) && isnothing(sampler.population) && isnothing(sampler.col_sampled) && isnothing(sampler.S)

    # test keyword constructor
    sampler = LinSysBlkColSelectWoReplacement(block_size=101, probability = [.5, .5])
    @test sampler.block_size == 101
    @test sampler.probability == [.5, .5]
    @test isnothing(sampler.population) && isnothing(sampler.col_sampled) && isnothing(sampler.S)

    # test assertion error
    @test_throws AssertionError LinSysBlkColSelectWoReplacement(block_size = -1)

    ####################
    ### test sampler ###
    ####################

    # problem context
    A = randn(5, 100)
    b = randn(5)
    x0 = zeros(100)

    # initialize sampler
    block_size = 10
    sampler = LinSysBlkColSelectWoReplacement(block_size = block_size)
    
    # first iteration
    sample_sketch = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check initializations
    @test size(sampler.S) == (size(A)[2], block_size)
    @test size(sampler.col_sampled) == (block_size, )
    @test sampler.probability == repeat([1/size(A)[2]], outer = size(A)[2]) && isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[2])

    # check output
    @test length(sample_sketch) == 4
    S, AS, res, grad = sample_sketch[1], sample_sketch[2], sample_sketch[3], sample_sketch[4]

    # check sketched matrix and residual is correct
    @test norm(AS - A * S) < eps() * 1e2
    @test norm(AS - A[:, sampler.col_sampled]) < eps() * 1e2
    @test norm(A * S - A[:, sampler.col_sampled]) < eps() * 1e2
    @test norm(res - (A * x0 - b)) < eps() * 1e2
    @test norm(grad - (S' * A' * (A * x0 - b))) < eps() * 1e2

    # check that sketched matrix has correct structure
    flag_row = true
    for j in 1:size(S)[2]
        # each col should have only one 1 as this corresponds to selecting a row
        sum_to_one = sum(S[:, j]) == 1
        non_negative = sum(S[:, j] .>= 0) == size(A)[2]
        correct_position = (S[sampler.col_sampled[j], j] == 1)
        flag_row = flag_row && (sum_to_one && non_negative && correct_position)
    end
    @test flag_row

    flag_col = true
    for i in 1:size(S)[1]
        # each row should only have one 1 or all 0 as we sample w.o. replacement
        sum_to_one = sum(S[i, :]) == 1 || sum(S[i, :]) == 0
        non_negative = sum(S[i, :] .>= 0) == block_size
        flag_col = flag_col && (sum_to_one && non_negative)
    end
    @test flag_col

    # second iteration; test if resampled
    old_col = copy(sampler.col_sampled)
    sample_sketch_2 = RLinearAlgebra.sample(sampler, A, b, x0, 2)
    @test old_col != sampler.col_sampled

    # test changing weights
    probability = repeat([0.], outer = size(A)[2])
    probability[1:20] .= 1/20
    block_size = 20

    sampler = LinSysBlkColSelectWoReplacement(block_size = block_size, probability = probability)
    
    # sample
    S, AS, res, grad = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check probability initializations
    @test size(sampler.S) == (size(A)[2], block_size)
    @test size(sampler.col_sampled) == (block_size, )
    @test isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[2])

    # check that no rows greater than 20 were sampled
    @test sum(sampler.col_sampled .<= 20) == length(sampler.col_sampled)

    # test error checking in keyword constructor
    # weights do not add to 1.
    sampler = LinSysBlkColSelectWoReplacement(block_size = 2, probability = [1., 1.])
    @test_throws DomainError RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # weights are not non-negative
    sampler = LinSysBlkColSelectWoReplacement(block_size = 2, probability = [2., -1.])
    @test_throws DomainError RLinearAlgebra.sample(sampler, A, b, x0, 1)


    # weights do not form a valid distribution over cols
    ncol = size(A)[2] - 50
    probability = Weights(repeat([1/ncol], outer = ncol)) # a vector of length 50
    sampler = LinSysBlkColSelectWoReplacement(block_size = 2, probability = probability)
    @test_throws DimensionMismatch RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # not enough non-zero probabilities
    ncol = size(A)[2]
    probability = Weights(repeat([0.], outer = ncol))
    probability[1] = 1. # only one non-zero probability but block_size > 1
    sampler = LinSysBlkColSelectWoReplacement(block_size = 2, probability = probability)
    @test_throws DimensionMismatch RLinearAlgebra.sample(sampler, A, b, x0, 1)
    @test size(probability)[1] == size(A)[2] # ignores the first dim mismatch error

    # initialize sampler with arbitrary values
    block_size = 10
    sampler = LinSysBlkColSelectWoReplacement(block_size, nothing, collect(1:10000), collect(1:10000), ones(10000,10000))
    
    # first iteration
    sample_sketch = RLinearAlgebra.sample(sampler, A, b, x0, 1)

    # check initializations
    @test size(sampler.S) == (size(A)[2], block_size)
    @test size(sampler.col_sampled) == (block_size, )
    @test sampler.probability == repeat([1/size(A)[2]], outer = size(A)[2]) && isa(sampler.probability, Weights)
    @test sampler.population == collect(1:size(A)[2])

    # check output
    @test length(sample_sketch) == 4
    S, AS, res, grad = sample_sketch[1], sample_sketch[2], sample_sketch[3], sample_sketch[4]

    # check sketched matrix and residual is correct
    @test norm(AS - A * S) < eps() * 1e2
    @test norm(AS - A[:, sampler.col_sampled]) < eps() * 1e2
    @test norm(A * S - A[:, sampler.col_sampled]) < eps() * 1e2
    @test norm(res - (A * x0 - b)) < eps() * 1e2
    @test norm(grad - (S' * A' * (A * x0 - b))) < eps() * 1e2

    # check that sketched matrix has correct structure
    flag_row = true
    for j in 1:size(S)[2]
        # each col should have only one 1 as this corresponds to selecting a row
        sum_to_one = sum(S[:, j]) == 1
        non_negative = sum(S[:, j] .>= 0) == size(A)[2]
        correct_position = (S[sampler.col_sampled[j], j] == 1)
        flag_row = flag_row && (sum_to_one && non_negative && correct_position)
    end
    @test flag_row

    flag_col = true
    for i in 1:size(S)[1]
        # each row should only have one 1 or all 0 as we sample w.o. replacement
        sum_to_one = sum(S[i, :]) == 1 || sum(S[i, :]) == 0
        non_negative = sum(S[i, :] .>= 0) == block_size
        flag_col = flag_col && (sum_to_one && non_negative)
    end
    @test flag_col

end 

end # End Module
