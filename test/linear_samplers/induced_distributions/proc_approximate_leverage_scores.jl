# Date: 01/16/2025
# Author: Christian Varner
# Purpose: Test cases for approximate leverage scores

module ProceduralApproximateLeverageScores

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Procedural -- Approximate Leverage Scores" begin

    # test definition
    @test isdefined(RLinearAlgebra, :approximate_leverage_score_distribution)
    @test isdefined(RLinearAlgebra, :_approximate_leverage_score)

    # test cases for _approximate_leverage_score
    row = 100
    col = 10
    sampler_row = RLinearAlgebra.LinSysBlkRowSRHT(;block_size = 10)
    sampler_col = RLinearAlgebra.LinSysBlkColSRHT()


    let sampler_row = sampler_row, sampler_col = sampler_col, row = row, col = col
        function row_sampler_function(A) 
            _, SA, _ = RLinearAlgebra.sample(sampler_row, A, randn(row), randn(col), 1)
            return SA
        end

        function col_sampler_function(A)
            _, AS, _, _ = RLinearAlgebra.sample(sampler_col, A, randn(row), randn(col), 1)
            return AS
        end

        A = randn(row, col)
        dist = RLinearAlgebra._approximate_leverage_score(A, 
            row_sampler_function,
            col_sampler_function)

        # distribution characteristics
        @test length(dist) == row
        @test sum(dist) ≈ 1.0
        @test sum(dist .>= 0) == row

        # TODO - test if the correct method is being applied
    end

    # test for approximate_leverage_score -- row_distribution = true
    row = 100
    col = 10
    sampler_row = RLinearAlgebra.LinSysBlkRowSRHT(;block_size = 10)
    sampler_col = RLinearAlgebra.LinSysBlkColSRHT()


    let sampler_row = sampler_row, sampler_col = sampler_col, row = row, col = col
        function row_sampler_function(A) 
            _, SA, _ = RLinearAlgebra.sample(sampler_row, A, randn(row), randn(col), 1)
            return SA
        end

        function col_sampler_function(A)
            _, AS, _, _ = RLinearAlgebra.sample(sampler_col, A, randn(row), randn(col), 1)
            return AS
        end

        A = randn(row, col)
        dist = RLinearAlgebra.approximate_leverage_score_distribution(A, 
            row_sampler_function,
            col_sampler_function,
            true)

        # distribution characteristics
        @test length(dist) == row
        @test sum(dist) ≈ 1.0
        @test sum(dist .>= 0) == row

        # TODO - test if the correct method is being applied
    end

    # test for approximate_leverage_score -- row_distribution = false
    row = 10
    col = 100
    sampler_row = RLinearAlgebra.LinSysBlkRowSRHT(;block_size = 10)
    sampler_col = RLinearAlgebra.LinSysBlkColSRHT()


    let sampler_row = sampler_row, sampler_col = sampler_col, row = row, col = col
        function row_sampler_function(A) 
            _, SA, _ = RLinearAlgebra.sample(sampler_row, A, randn(col), randn(row), 1)
            return SA
        end

        function col_sampler_function(A)
            _, AS, _, _ = RLinearAlgebra.sample(sampler_col, A, randn(col), randn(row), 1)
            return AS
        end

        A = randn(row, col)
        dist = RLinearAlgebra.approximate_leverage_score_distribution(A, 
            row_sampler_function,
            col_sampler_function,
            false)

        # distribution characteristics
        @test length(dist) == col
        @test sum(dist) ≈ 1.0
        @test sum(dist .>= 0) == col

        # TODO - test if the correct method is being applied
    end

end

end