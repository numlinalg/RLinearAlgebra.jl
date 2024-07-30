# This file is part of RLinearAlgebra.jl
# Date: 07/30/2024
# Author: Christian Varner
# Purpose: Test the distribution creating procedure in
# distributions/row_dist_leverage_score.jl

module ProceduralTestRowDistLeverageScore

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Row Distribution using Leverage Scores -- Procedural" begin

    # test the type
    @test supertype(RowDistLeverageScore) == RowDistribution

    # testing context
    A = randn(100, 5)

    # get distribution
    dist_type = RowDistLeverageScore()
    row_dist = RLinearAlgebra.getDistribution(dist_type, A)

    # row_dist is indeed a valid probability distribution
    @test length(row_dist) == size(A)[1]
    @test sum(row_dist) ≈ 1
    @test sum(row_dist .>= 0) == size(row_dist)[1]
    @test isa(row_dist, Weights)

    # distribution is initialized correctly
    Q1 = Matrix(qr(A).Q)
    normalizing_constant = norm(Q1)^2
    flag = true
    for i in 1:size(Q1)[1]
       flag = flag & (row_dist[i]*normalizing_constant ≈ norm(Q1[i, :])^2 )
    end
    @test flag
end

end 