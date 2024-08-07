# Date: 08/06/2024
# Author: Christian Varner
# Purpose: Test creating a distribution with leverage scores
# of a matrix implemented in src/distributions/leverage_scores.jl

module ProceduralTestDistLeverageScore

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

Random.seed!(1010)

@testset "Distribution by Leverage Scores -- Procedural" begin

  # test the type
  @test supertype(DistLeverageScore) == Distribution

  # test the constructor
  dist = DistLeverageScore(Left)
  @test eltype(dist) == Left

  dist = DistLeverageScore(Right)
  @test eltype(dist) == Right
  
  dist = DistLeverageScore(SketchDirection)
  @test eltype(dist) == SketchDirection

  # test with different inputs
  dist_init = randn(100)
  dist = DistLeverageScore(Left, dist = dist_init, flag = false)

  dist = DistLeverageScore(Left, dist = dist_init, flag = true)

  dist = DistLeverageScore(Right, dist = dist_init, flag = false)

  dist = DistLeverageScore(Right, dist = dist_init, flag = true)


end

end # end module
