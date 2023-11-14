# This file is part of RLinearAlgebra.jl

module ProceduralTestLSStopMaxIterations

using Test, RLinearAlgebra

@testset "LS Threshold Stop -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSStopThreshold) == LinSysStopCriterion

    # Verify check_stop_criterion functionality
    log = LSLogFull()
    stop = LSStopThreshold(10, 1e-10)
    
    #Test max iteration stopping
    log.iterations = 0
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 10
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

    log.iterations = 11
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    #Verify threshold stopping
    log.resid_hist = [1, 1e-11]

    log.iterations = 1
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 2
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

end

end # End module
