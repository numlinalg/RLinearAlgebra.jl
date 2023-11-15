# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogFullMA

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LS Full Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogFullMA) == LinSysSolverLog

    # Verify Required Fields
    @test :iterations in fieldnames(LSLogFullMA)
    @test :converged in fieldnames(LSLogFullMA)

    # Verify log_update initialization
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA()


        RLinearAlgebra.log_update!(log, sampler, z, (A[1,:],b[1]), 0, A, b)

        @test length(log.resid_hist) == 1
        @test log.resid_hist[1] == norm(dot(A[1,:], z) - b[1])^2
        @test norm(log.iota_hist[1] - norm(dot(A[1,:], z) - b[1])^4) < 1e-15
        @test log.iterations == 0
        @test log.converged == false
    end

    # Verify moving average 
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA(lambda2 = 2)
        samp = (A[1,:], b[1])


        # Test moving average of log
        for i = 1:10
            samp = (A[1,:], b[1])
            RLinearAlgebra.log_update!(log, sampler, x + i*(z-x), samp, i, A, b)
        end
        #compute sampled residuals
        obs_res = [abs(dot(A[1,:],x + i*(z-x)) - b[1])^2 for i = 1:10]
        obs_res2 = [abs(dot(A[1,:],x + i*(z-x)) - b[1])^4 for i = 1:10]
        @test length(log.resid_hist) == 10
        @test norm(log.resid_hist - vcat(obs_res[1:2], 
                                         [(obs_res[i] + obs_res[i-1])/2 for i = 3:10])) < 1e-15
        @test norm(log.iota_hist - vcat(obs_res2[1:2], 
                                        [(obs_res2[i] + obs_res2[i-1])/2 for i = 3:10])) < 1e-15
        @test log.iterations == 10
        @test log.converged == false
        
        #Test uncertainty set 
        Uncertainty_set = get_uncertainty(log)
        @test length(Uncertainty_set[1]) == 10
        #If you undo the steps of the interval calculation should be 1
        @test norm((Uncertainty_set[2] - log.resid_hist) ./ sqrt.(2 * Base.log(2/.05) * log.iota_hist * 
                    log.sigma2 .* (1 .+ Base.log.(log.width_hist)) ./  log.width_hist) .- 1) < 1e-15
    end

end

end # End Module
