# Date: 01/09/2025
# Author: Christian Varner
# Purpose: Test creating a distribution by leverage scores

module ProceduralLeverageScoreDistribution

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

@testset "Leverage Score Distribution -- Procedural" begin

    # test definition
    @test isdefined(RLinearAlgebra, :leverage_score_distribution)

    # error testing
    dim1 = 10
    dim2 = 5
    let dim1 = dim1, dim2 = dim2
        A = randn(dim1, dim2) ## 10, 5
        @test_throws AssertionError RLinearAlgebra.leverage_score_distribution(A, false)

        A = randn(dim2, dim1) ## 5, 10
        @test_throws AssertionError RLinearAlgebra.leverage_score_distribution(A, true)
    end

    # test distribution output
    let dim1 = dim1, dim2 = dim2
        # test for row distribution
        A = randn(dim1, dim2)
        dist = RLinearAlgebra.leverage_score_distribution(A, true)

        # test characteristics
        @test typeof(dist) == Vector{Float64}
        @test sum(dist) ≈ 1
        @test sum(dist .>= 0) == dim1
        @test length(dist) == dim1

        # test correct values
        Q1 = Matrix(qr(A).Q)
        for i in 1:dim1
            @test dist[i] ≈ norm(view(Q1, i, :))^2 / norm(Q1)^2
        end

        # test for column distribution
        A = randn(dim2, dim1)
        dist = RLinearAlgebra.leverage_score_distribution(A, false)

        # test characteristics
        @test typeof(dist) == Vector{Float64}
        @test sum(dist) ≈ 1
        @test sum(dist .>= 0) == dim1
        @test length(dist) == dim1

        Q1 = Matrix(qr(A').Q)
        for i in 1:dim2
            @test dist[i] ≈ norm(view(Q1, i, :))^2 / norm(Q1)^2
        end
    end

end
    
end