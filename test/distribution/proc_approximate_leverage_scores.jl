# Date: 08/07/2024
# Author: Christian Varner
# Purpose: Test creating a distribution with approximate leverage
# scores implemented in src/distributions/approximate_leverage_scores.jl

module ProceduralTestDistApproximateLeverageScore

using Test, RLinearAlgebra, Random, LinearAlgebra

@testset "Distribution by Approximate Leverage Scores -- Procedural" begin
  
  @test supertype(DistApproximateLeverageScore) == Distribution

  ####################################################################
  # Test the constructor
  ####################################################################
  # test the constructor
  dist = DistApproximateLeverageScore(Left, randn(2,2), randn(2,2))
  @test eltype(dist) == Left

  dist = DistApproximateLeverageScore(Right, randn(2,2), randn(2,2))
  @test eltype(dist) == Right

  dist = DistApproximateLeverageScore(SketchDirection, randn(2,2), randn(2,2))
  @test eltype(dist) == SketchDirection

  # test with different inputs
  dist_init = randn(200)
  dist = DistApproximateLeverageScore(Left, randn(2,2), randn(2,2), dist = dist_init, flag = false)
  @test dist.dist == dist_init
  @test dist.initialized_storage == false
  @test eltype(dist) == Left
  
  dist = DistApproximateLeverageScore(Left, randn(2,2), randn(2,2), dist = dist_init, flag = true)
  @test dist.dist == dist_init
  @test dist.initialized_storage == true
  @test eltype(dist) == Left
  
  dist = DistApproximateLeverageScore(Right, randn(2,2), randn(2,2), dist = dist_init, flag = false)
  @test dist.dist == dist_init
  @test dist.initialized_storage == false
  @test eltype(dist) == Right
  
  dist = DistApproximateLeverageScore(Right, randn(2,2), randn(2,2), dist = dist_init, flag = true)
  @test dist.dist == dist_init
  @test dist.initialized_storage == true
  @test eltype(dist) == Right
  ####################################################################
  # End of tests for the constructor
  ####################################################################

  ####################################################################
  # Test the getDistribution! function
  # initialize! calls getDistribution!
  ####################################################################
  # TODO: implement some test cases please


  ####################################################################
  # End of tests for the getDistribution! function
  # initialize! calls getDistribution!
  ####################################################################
end

end # end module
