# Date: 08/06/2024
# Author: Christian Varner
# Purpose: Test the procedure that initializes
# a distribution using the frobenius norm of rows and columns
# in src/distributions/frobenius_norm.jl

module ProceduralTestDistFrobeniusNorm

using Test, RLinearAlgebra, Random, LinearAlgebra, StatsBase

@testset "Distribution using Frobenius Norm -- Procedural" begin

  # test the type
  @test supertype(DistFrobeniusNorm) == Distribution

end

end # end of module
