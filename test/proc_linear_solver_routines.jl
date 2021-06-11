# This file is part of RLinearAlgebra.jl

module ProceduralTestLinSysSolveRoutine

using Test, RLinearAlgebra

@testset "Linear System Solver Routine Abstractions" begin

    # Verify parent types
    @test supertype(LinSysVecRowProjection) == LinSysSolveRoutine
    @test supertype(LinSysVecColProjection) == LinSysSolveRoutine
    @test supertype(LinSysBlkRowProjection) == LinSysSolveRoutine
    @test supertype(LinSysBlkColProjection) == LinSysSolveRoutine
    @test supertype(LinSysPreconKrylov) == LinSysSolveRoutine

end

end # End module
