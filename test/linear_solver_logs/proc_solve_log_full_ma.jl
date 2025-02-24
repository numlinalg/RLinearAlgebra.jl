# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogFullMA

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LS Full MA Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogFullMA) == LinSysSolverLog

    # Verify Required Fields
    @test :iterations in fieldnames(LSLogFullMA)
    @test :converged in fieldnames(LSLogFullMA)

    # Initialize data
    A = rand(2,2)
    x = rand(2)
    b = A * x
    z = rand(2)
    
    # Verify log_update initialization
    let A = A, x = x, b = b, z = z
        
        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA()


        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.resid_hist) == 1
        @test norm(log.resid_hist[1] - norm(A * z - b)^2) < 1e2 * eps()
        @test log.iterations == 0
        @test log.converged == false
    end

    # Verify late moving average 
    let A = A, x = x, b = b, z = z

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA(lambda2 = 2)

        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)
        # Test moving average of log
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + (i+1)*(z-x), (), i, A, b)
        end
        #compute sampled residuals
        obs_res = [norm(A*(x + (i+1)*(z-x)) - b)^2 for i = 0:10]
        obs_res2 = [norm(A*(x + (i+1)*(z-x)) - b)^4 for i = 0:10]
        @test length(log.resid_hist) == 11
        @test norm(log.resid_hist[3:11] - vcat(obs_res[3], 
                                         [(obs_res[i] + obs_res[i-1])/2 for i = 4:11])) < 1e2 * eps()
        @test log.iterations == 10
        @test log.converged == false
    end

    # Verify early moving average
    let A = A, x = x, b = b, z = z

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA(lambda1 = 2, lambda2 = 10)

        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)
        # Test moving average of log when the residual only decreases to not trigger switch
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + .3^(i+1)*(z-x), (), i, A, b)
        end
        #compute sampled residuals
        obs_res = [norm(A*(x + .3^(i+1)*(z-x)) - b)^2 for i = 0:10]
        @test length(log.resid_hist) == 11
        @test norm(log.resid_hist[3:11] - vcat( [(obs_res[i] + obs_res[i-1])/2 for i = 3:11])) < 1e2 * eps()
        @test log.iterations == 10
        @test log.converged == false
    
    end
  
    # Verify collection rate
    let A = A, x = x, b = b, z = z

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA(collection_rate = 3)

        # Test initialization of log
        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.resid_hist) == 1
        @test norm(log.resid_hist[1] - norm(A * z - b)^2) < 1e2 * eps()
        @test log.iterations == 0
        @test log.converged == false

        # Test collection rate of log
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + i*(z-x), (), i, A, b)
        end

        @test length(log.resid_hist) == 4 # Record at 0, 3, 6, 9
        @test log.iterations == 10
        @test log.converged == false
    end

end

end # End Module
