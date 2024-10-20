# This file is part of RLinearAlgebra.jl

module ProceduralTestLowRankApproximationError

using Test, RLinearAlgebra

@testset "Low Rank Approximation Error Abstractions" begin

    # Verify parent types
    @test supertype(RangeFinderError) == ApproxErrorMethod 

end

end # End module
