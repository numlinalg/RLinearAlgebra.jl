# This file is part of RLinearAlgebra.jl
# Date: 07/30/2024
# Author: Christian Varner
# Purpose: Test the distribution creating procedure in
# distributions/col_dist_leverage_score.jl

module ProceduralTestColDistLeverageScore

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Col Distribution using Leverage Scores -- Procedural" begin

    # test the type
    @test supertype(ColDistLeverageScore) == ColDistribution

    # testing context
    A = randn(100, 5)

    # get distribution
    dist_type = ColDistLeverageScore()
    col_dist = RLinearAlgebra.getDistribution(dist_type, A)

    # row_dist is indeed a valid probability distribution
    @test length(col_dist) == size(A)[2]
    @test sum(col_dist) ≈ 1
    @test sum(col_dist .>= 0) == size(col_dist)[1]
    @test isa(col_dist, Weights)

    # distribution is initialized correctly
    Q1 = Matrix(qr(A').Q)
    normalizing_constant = norm(Q1)^2
    flag = true
    for i in 1:size(Q1)[1]
       flag = flag & (col_dist[i]*normalizing_constant ≈ norm(Q1[i, :])^2 )
    end
    @test flag
end

end 