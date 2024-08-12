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
  
  function isDistribution(vector::Vector{Float64}, correct_size::Int64)
    flag = true
    flag = flag & (isapprox(sum(vector), 1))
    flag = flag & (sum(vector .>= 0) == size(vector)[1])
    flag = flag & (size(vector)[1] == correct_size)
    return flag
  end
  
  # testing context
  A = randn(100, 50)
  Π_1 = randn(50, 100)
  Π_2 = randn(50, 25)
  
  # create distribution - Left - taking rows
  dist = DistApproximateLeverageScore(Left, Π_1, Π_2)
  RLinearAlgebra.initialize!(dist, A)
  @test isDistribution(dist.dist, 100)
  
  # create distribution - Right
  A = randn(50, 100)
  dist = DistApproximateLeverageScore(Right, Π_1, Π_2)
  RLinearAlgebra.initialize!(dist, A)
  @test isDistribution(dist.dist, 100)

  # create distribution - Left - storage initialized
  A = randn(100, 50)
  Π_1 = randn(50, 100)
  Π_2 = randn(50, 25)
  dist = DistApproximateLeverageScore(Left, Π_1, Π_2, dist = randn(100), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test isDistribution(dist.dist, 100)
  @test dist.initialized_storage

  # create distribution - Right - storage initialized
  A = randn(50, 100)
  dist = DistApproximateLeverageScore(Right, Π_1, Π_2, dist = randn(100), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test isDistribution(dist.dist, 100)
  @test dist.initialized_storage

  # test error checking
  dist = DistApproximateLeverageScore(Left, Π_1, Π_2, dist = randn(100), flag = true)
  try
    RLinearAlgebra.initialize!(dist, A) # sketched matrix has Π_1 has too few rows
  catch err
    @test isa(err, BoundsError)
  end
  
  A = randn(100, 200)
  Π_1 = randn(200, 100)
  Π_2 = randn(200, 50)
  dist = DistApproximateLeverageScore(Left, Π_1, Π_2, dist = randn(200), flag = true)
  try
    RLinearAlgebra.initialize!(dist, A) # A has too many columns
  catch err
    @test isa(err, DomainError)
  end 
  
  # checking to make sure distribution is correct
  A = randn(100, 50)
  Π_1 = Matrix{Float64}(I, 100, 100)
  Π_2 = Matrix{Float64}(I, 50, 50)
  dist = DistApproximateLeverageScore(Left, Π_1, Π_2)
  RLinearAlgebra.initialize!(dist, A) # should just do SVD on A and retrieve U

  res = svd(A; full = true)
  Omega = res.U[:,1:50]
  nc = norm(Omega)^2

  flag = true
  for i in 1:size(A)[1]
    flag = flag & (isapprox(dist.dist[i] * nc, norm(Omega[i, :])^2))
  end
  @test flag

  ####################################################################
  # End of tests for the getDistribution! function
  # initialize! calls getDistribution!
  ####################################################################
end

end # end module
