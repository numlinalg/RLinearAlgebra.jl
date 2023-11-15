# This file is part of RLinearAlgebra.jl

module ProceduralTestLSStopMA

using Test, RLinearAlgebra

@testset "LS Moving Average Stop -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSStopMA) == LinSysStopCriterion

    # Verify check_stop_criterion functionality
    log = LSLogFullMA()
    stop = LSStopMA(2, 1e-10, 1.1, .9, .01, .01)
    log.resid_hist = [1, 1, 1]
    log.iota_hist = [1, 1, 1]
    log.max_dimension = 100

    log.iterations = 0
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 2 
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

    log.iterations = 3
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    #Verify threshold stopping
    log.resid_hist = [1, 1e-11, 1e-11]
    log.iota_hist = [1, 1e-11, 1e-32]
    log.sigma2 = 1
    log.ma_info.lambda = 15
    stop = LSStopMA(4, 1e-10, 1.1, .9, .01, .01)

    log.iterations = 1
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 2
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iterations = 3
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true
end

end # End module
