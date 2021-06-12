# This file is part of RLinearAlgebra.jl

module ProceduralTestLSStopMaxIterations

using Test, RLinearAlgebra

@testset "LS Max Iteration Stop -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSStopMaxIterations) == LinSysStopCriterion

    # Verify check_stop_criterion functionality
    log = LSLogFull()
    stop = LSStopMaxIterations(10)

    log.iterations = 0
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 10
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

    log.iterations = 11
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false
end

end # End module
