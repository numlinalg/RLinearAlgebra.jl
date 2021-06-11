# This file is part of RLinearAlgebra.jl

module ProceduralTestLinSysSolveRoutine

@testset "Linear System Solver Routine Abstractions"

    # Verify parent types
    @test supertype(LinSysVecRowProjection) == LinSysSolveRoutine
    @test supertype(LinSysVecColProjection) == LinSysSolveRoutine
    @test supertype(LinSysBlkRowProjection) == LinSysSolveRoutine
    @test supertype(LinSysBlkColProjection) == LinSysSolveRoutine
    @test supertype(LinSysPreconKrylov) == LinSysKrylovPrecon

end # End module
