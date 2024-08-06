# Date: 08/06/2024
# Author: Christian Varner
# Purpose: Test the functionality of the file src/distribution.jl


module ProceduralTestDistribution

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "Distributions -- Procedural" begin

  # check that abstract types are defined
  @test @isdefined SketchDirection
  @test @isdefined Left
  @test @isdefined Right
  @test @isdefined Distribution
  @test @isdefined DistDefault

  @test :dist in fieldnames(DistDefault)
  @test :initialized_storage in fieldnames(DistDefault)

  # make sure type information is correct
  @test supertype(Left) == SketchDirection
  @test supertype(Right) == SketchDirection

  # test eltype for abstract type Distribution
  dist = Distribution{Left}
  @test eltype(dist) == Left
  
  dist = Distribution{Right}
  @test eltype(dist) == Right

  dist = Distribution{SketchDirection}
  @test eltype(dist) == SketchDirection
  
  # test eltype for struct DistDefault
  dist = DistDefault{Left}(zeros(1), false)
  @test eltype(dist) == Left

  dist = DistDefault{Right}(zeros(1), false)
  @test eltype(dist) == Right

  dist = DistDefault{SketchDirection}(zeros(1), false)
  @test eltype(dist) == SketchDirection

  # test DistDefault parameters

  # test initialize!
  A = randn(100, 50)
  
  # check left
  dist = DistDefault{Left}(zeros(1), false)
  @test dist.dist == zeros(1)
  @test dist.initialized_storage == false

  RLinearAlgebra.initialize!(dist, A)
  @test size(dist.dist)[1] == 100
  @test dist.dist == zeros(100)
  @test dist.initialized_storage == true

  # check right
  dist = DistDefault{Right}(zeros(1), false)
  @test dist.dist == zeros(1)
  @test dist.initialized_storage == false

  RLinearAlgebra.initialize!(dist, A)
  @test size(dist.dist)[1] == 50
  @test dist.dist == zeros(50)
  @test dist.initialized_storage == true

  # check when initialized_storage == true
  nonsense = randn(10)
  dist = DistDefault{Left}(nonsense, true)
  @test dist.dist == nonsense
  @test dist.initialized_storage == true

  RLinearAlgebra.initialize!(dist, A) # nothing should happen
  @test dist.dist == nonsense
  @test dist.initialized_storage == true

  dist = DistDefault{Right}(nonsense, true)
  @test dist.dist == nonsense
  @test dist.initialized_storage == true

  RLinearAlgebra.initialize!(dist, A) # nothing should happen
  @test dist.dist == nonsense
  @test dist.initialized_storage == true



end

end # End Module
