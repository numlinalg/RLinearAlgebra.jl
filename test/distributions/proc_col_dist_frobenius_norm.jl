# This file is part of RLinearAlgebra.jl
# Date: 07/30/2024
# Author: Christian Varner
# Purpose: Test the distribution creating procedure in
# distributions/col_dist_frobenius_norm.jl

module ProceduralTestColDistFrobeniusNorm

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Column Distribution using Frobenius Norm -- Procedural" begin

    # test the supertype
    @test supertype(ColDistFrobeniusNorm) == ColDistribution

    # testing context
    A = randn(5,100)

    # get the distribution
    dist_type = ColDistFrobeniusNorm()
    col_dist = RLinearAlgebra.getDistribution(dist_type, A)

    # col_dist is indeed a probability distributio over columns of A
    @test length(col_dist) == size(A)[2]
    @test sum(col_dist) ≈ 1
    @test sum(col_dist .>= 0) == size(col_dist)[1]
    @test isa(col_dist, Weights)

    # distribution is initialized correctly
    flag = true
    normalizing_constant = norm(A)^2
    for i in 1:size(A)[2]
        flag = flag & (col_dist[i] * normalizing_constant ≈ norm(A[:, i])^2)
    end
    @test flag

end

end # end module