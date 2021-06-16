# This file is part of RLinearAlgebra.jl

module ProceduralTestLinSysStopCriterion

using Test, RLinearAlgebra

@testset "Linear System Stopping Criteria Abstractions" begin

    # Verify export
    @test @isdefined LinSysStopCriterion
    
end

end # End module
