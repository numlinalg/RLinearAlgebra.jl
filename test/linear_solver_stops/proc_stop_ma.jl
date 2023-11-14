# This file is part of RLinearAlgebra.jl

module ProceduralTestLSStopMA

using Test, RLinearAlgebra

@testset "LS Moving Average Stop -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSStopMA) == LinSysStopCriterion

    # Verify check_stop_criterion functionality
    log = LSLogFullMA()
    stop = LSStopMaxIterations(10, 1e-10, 1.1, .9, .01, .01)

    log.iterations = 0
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 10
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

    log.iterations = 11
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    #Verify threshold stopping
    log.resid_hist = [1, 1e-11, 1e-11]
    log.iota_hist = [1, 1e-11, 1e-22]
    log.sigma2 = 1
    log.MAInfo.lambda = 15

    log.iterations = 1
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 2
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 3
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true
end

end # End module
