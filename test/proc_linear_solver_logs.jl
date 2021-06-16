# This file is part of RLinearAlgebra.jl

module ProceduralTestLinSysSolverLog

using Test, RLinearAlgebra

@testset "Linear System Log Abstractions" begin

    # Verify export
    @test @isdefined LinSysSolverLog

end

end # End Module
