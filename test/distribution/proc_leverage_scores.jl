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
  
  ################################################################
  # Test the constructor
  ################################################################
  # test the constructor
  dist = DistLeverageScore(Left)
  @test eltype(dist) == Left

  dist = DistLeverageScore(Right)
  @test eltype(dist) == Right
  
  dist = DistLeverageScore(SketchDirection)
  @test eltype(dist) == SketchDirection

  # test with different inputs
  dist_init = randn(200)
  dist = DistLeverageScore(Left, dist = dist_init, flag = false)
  @test dist.dist == dist_init
  @test dist.initialized_storage == false
  @test eltype(dist) == Left

  dist = DistLeverageScore(Left, dist = dist_init, flag = true)
  @test dist.dist == dist_init
  @test dist.initialized_storage == true
  @test eltype(dist) == Left

  dist = DistLeverageScore(Right, dist = dist_init, flag = false)
  @test dist.dist == dist_init
  @test dist.initialized_storage == false
  @test eltype(dist) == Right

  dist = DistLeverageScore(Right, dist = dist_init, flag = true)
  @test dist.dist == dist_init
  @test dist.initialized_storage == true
  @test eltype(dist) == Right
  ###############################################################
  # End of Test the constructor
  ################################################################

  ################################################################
  # Test the getDistribution! function
  # initialize! calls getDistribution!
  ################################################################
  A = randn(100, 50)
  Q1 = Matrix(qr(A).Q) # thin QR decomposition
  Q2 = Matrix(qr(A').Q)
  
  """
  Test whether or not the vector is a probability vector of size correct_size.

  It will test:
    - if it is the correct size
    - if all elements are greater than or equal to 0
    - if the elements sum to 1
  """
  function isDistribution(vector::Vector{Float64}, correct_size::Int64)
    flag = true
    flag = flag & (isapprox(sum(vector), 1))
    flag = flag & (sum(vector .>= 0) == size(vector)[1])
    flag = flag & (size(vector)[1] == correct_size)
    return flag
  end
  
  """
  Test whether or not the distribution was initialized using the leverage scores
  of the matrix Q.
  """
  function initializedCorrectly(vector::Vector{Float64}, Q::AbstractArray)
    flag = true
    @assert size(vector)[1] == size(Q)[1]
    nc = norm(Q)^2
    for i in 1:size(Q)[1]
      flag = flag & ( isapprox(vector[i]*nc, norm(Q[i, :])^2) )
    end
    
    return flag
  end

  # test using default constructor - Left
  dist = DistLeverageScore(Left)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 100) 
  @test initializedCorrectly(dist.dist, Q1)

  # test using default constructor - Right
  dist = DistLeverageScore(Right)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 50)
  @test initializedCorrectly(dist.dist, Q2)

  # test using different initializations of the parameters - Left
  dist = DistLeverageScore(Left, dist = dist_init, flag = false)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 100)
  @test initializedCorrectly(dist.dist, Q1)

  # test using different initializations of the parameters - Right
  dist = DistLeverageScore(Right, dist = dist_init, flag = false)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 50)
  @test initializedCorrectly(dist.dist, Q2)

  # test initialization procedure when buffer array has been initialized - Left
  dist = DistLeverageScore(Left, dist = randn(100), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 100)
  @test initializedCorrectly(dist.dist, Q1)

  # test initialization procedure when buffer array has been initialized - Left
  dist = DistLeverageScore(Right, dist = randn(50), flag = true)
  RLinearAlgebra.initialize!(dist, A)
  @test dist.initialized_storage
  @test isDistribution(dist.dist, 50)
  @test initializedCorrectly(dist.dist, Q2)
  ################################################################
  # End of Test the getDistribution! function
  # initialize! calls getDistribution!
  ################################################################

end

end # end module
