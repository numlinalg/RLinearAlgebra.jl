# Date: 08/06/2024
# Author: Christian Varner
# Purpose: Test the procedure that initializes
# a distribution using the frobenius norm of rows and columns
# in src/distributions/frobenius_norm.jl

module ProceduralTestDistFrobeniusNorm

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Distribution using Frobenius Norm -- Procedural" begin

  # function to test if the distribution is correctly initialized
  function testDistribution(dist::Vector{Float64}, B::AbstractMatrix)
    flag = true
    normalizing_constant = norm(B)^2
    for i in 1:size(B)[1]
      flag = flag & (isapprox(dist[i] * normalizing_constant, norm(B[i, :])^2 ) )
    end
    return flag
  end

  # test the type
  @test supertype(DistFrobeniusNorm) == Distribution
  
  ###################################################
  # Tests for Constructor
  ###################################################
  # Test the constructor
  dist = DistFrobeniusNorm(Left)
  @test eltype(dist) == Left
  @test dist.dist == zeros(1)
  @test dist.initialized_storage == false

  dist = DistFrobeniusNorm(Right)
  @test eltype(dist) == Right
  @test dist.dist == zeros(1)
  @test dist.initialized_storage == false

  # test custom initializations
  dist_initialization = randn(100)
  dist = DistFrobeniusNorm(Left, dist = dist_initialization, flag = false)
  @test eltype(dist) == Left
  @test dist.dist == dist_initialization
  @test dist.initialized_storage == false

  dist = DistFrobeniusNorm(Left, dist = dist_initialization, flag = true)
  @test eltype(dist) == Left
  @test dist.dist == dist_initialization
  @test dist.initialized_storage == true

  dist = DistFrobeniusNorm(Right, dist = dist_initialization, flag = false)
  @test eltype(dist) == Right
  @test dist.dist == dist_initialization
  @test dist.initialized_storage == false

  dist = DistFrobeniusNorm(Right, dist = dist_initialization, flag = true)
  @test eltype(dist) == Right
  @test dist.dist == dist_initialization
  @test dist.initialized_storage == true
  ###################################################
  # End Test for Constructor
  ###################################################

  ###################################################
  # Test for getDistribution!
  # initialize! calls getDistribution!
  ###################################################
  # get the distribution - Left
  A = randn(100,50)
  dist = DistFrobeniusNorm(Left, dist = dist_initialization, flag = false)
  RLinearAlgebra.initialize!(dist, A)

  # test that the distribution was correctly initialized
  @test size(dist.dist)[1] == 100
  @test dist.initialized_storage == true
  @test isapprox(sum(dist.dist), 1)
  @test sum(dist.dist .>= 0) == size(dist.dist)[1]
  @test testDistribution(dist.dist, A)

  # get the distribution - Right
  dist = DistFrobeniusNorm(Right, dist = dist_initialization, flag = false)
  RLinearAlgebra.initialize!(dist, A)

  # test that the distribution was correctly initialized
  @test size(dist.dist)[1] == 50
  @test dist.initialized_storage == true
  @test isapprox(sum(dist.dist), 1)
  @test sum(dist.dist .>= 0) == size(dist.dist)[1]
  @test testDistribution(dist.dist, A')

  # get the distribution when dist is initialized to correct length -- left
  dist = DistFrobeniusNorm(Left, dist = randn(100), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test size(dist.dist)[1] == 100
  @test dist.initialized_storage == true
  @test isapprox(sum(dist.dist), 1)
  @test sum(dist.dist .>= 0) == size(dist.dist)[1]
  @test testDistribution(dist.dist, A)

  # get the distribution when dist is initialized to correct length -- right
  dist = DistFrobeniusNorm(Right, dist = randn(50), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test size(dist.dist)[1] == 50
  @test dist.initialized_storage == true
  @test isapprox(sum(dist.dist), 1)
  @test sum(dist.dist .>= 0) == size(dist.dist)[1]
  @test testDistribution(dist.dist, A')

  ###################################################
  # End of Test for getDistribution!
  ###################################################
end

end # end of module
