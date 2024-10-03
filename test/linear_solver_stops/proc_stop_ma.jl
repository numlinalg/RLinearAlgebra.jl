
module ProceduralTestLSStopMA

using Test, RLinearAlgebra

@testset "LS Moving Average Stop -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSStopMA) == LinSysStopCriterion

    # Verify check_stop_criterion functionality
    log = LSLogMA()
    stop = LSStopMA(2, 1e-10, 1.1, .9, .01, .01)
    log.resid_hist = [1, 1, 1]
    log.iota_hist = [1, 1, 1]
    log.dist_info.dimension = 100
    log.dist_info.sampler = LinSysVecRowDetermCyclic

    log.iteration = 0
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iteration = 2 
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true

    log.iteration = 3
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    #Verify threshold stopping
    log.resid_hist = [1, 1e-11, 1e-11]
    log.iota_hist = [1, 1e-11, 1e-32]
    log.dist_info.sigma2 = 1
    log.ma_info.lambda = 15
    stop = LSStopMA(4, 1e-10, 1.1, .9, .01, .01)

    log.iteration = 1
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iteration = 2
    @test RLinearAlgebra.check_stop_criterion(log, stop) == false

    log.iteration = 3
    @test RLinearAlgebra.check_stop_criterion(log, stop) == true
end

end # End module
