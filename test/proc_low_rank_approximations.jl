# This file is part of RLinearAlgebra.jl

module ProceduralTestLowRankApproximations

using Test, RLinearAlgebra

@testset "Low Rank Approximation Abstractions" begin

    # Verify parent types
    @test supertype(RangeFinderMethod) == ApproxMethod 
    @test supertype(IntDecompMethod) == ApproxMethod
    @test supertype(NystromMethod) == ApproxMethod

end

end # End module
