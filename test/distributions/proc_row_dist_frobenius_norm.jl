# This file is part of RLinearAlgebra.jl
# Date: 07/30/2024
# Author: Christian Varner
# Purpose: Test the distribution creating procedure in
# distributions/row_dist_frobenius_norm.jl

module ProceduralTestRowDistFrobeniusNorm

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Row Distribution using Frobenius Norm -- Procedural" begin

    # test the type
    @test supertype(RowDistFrobeniusNorm) == RowDistribution

    # testing context
    A = randn(100, 5)

    # get distribution
    dist_type = RowDistFrobeniusNorm()
    row_distribution = RLinearAlgebra.getDistribution(dist_type, A)

    # row_distribution is indeed a probability distribution over rows of A
    @test length(row_distribution) == size(A)[1]
    @test sum(row_distribution) ≈ 1
    @test sum(row_distribution .>= 0) == size(row_distribution)[1]
    @test isa(row_distribution, Weights)

    # distribution is initialized correctly
    flag = true
    normalizing_constant = norm(A)^2
    for i in 1:size(A)[1]
        flag = flag & (row_distribution[i] * normalizing_constant ≈ norm(A[i, :])^2)
    end
    @test flag

end

end # end module