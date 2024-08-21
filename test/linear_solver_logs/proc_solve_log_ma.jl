# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogMA

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LS MA Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogMA) == LinSysSolverLog

    # Verify Required Fields
    @test :iteration in fieldnames(LSLogMA)
    @test :converged in fieldnames(LSLogMA)

    # Verify log_update initialization
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        logger = LSLogMA()
<<<<<<< HEAD

        # Test the error message on the get_uncertainty function
        @test_throws ArgumentError("The SE constants are empty, please set them in dist_info field of LSLogMA first.") get_uncertainty(logger) 
        # Test warning message in default SEconstant look up
=======
        @test_throws ArgumentError("The SE constants are empty, please set them in dist_info field of LSLogMA first.") get_uncertainty(logger) 
>>>>>>> 93056cf (removed sampler lines from code)

        RLinearAlgebra.log_update!(logger, sampler, z, (A[1, :], b[1]), 0, A, b)

        @test length(logger.resid_hist) == 1
<<<<<<< HEAD
        @test norm(logger.resid_hist[1] - 2 * norm(dot(A[1, :], z) - b[1])^2) < 1e2 * eps()
        @test norm(logger.iota_hist[1] - 4 * norm(dot(A[1, :], z) - b[1])^4) < 1e2 * eps()
=======
        @test logger.resid_hist[1] == norm(A * z - b)^2
        @test norm(logger.iota_hist[1] - norm(A * z - b)^4) < 1e2 * eps()
>>>>>>> 93056cf (removed sampler lines from code)
        @test logger.iteration == 0
        @test logger.converged == false
        
        struct MadeUpSampler <: LinSysSampler  
        end

        @test_logs (:warn, "No constants defined for method of type Main.ProceduralTestLSLogMA.MadeUpSampler. By default we set sigma2 to 1 and scaling to 1.") RLinearAlgebra.get_SE_constants!(logger, MadeUpSampler)
        @test_throws ArgumentError("`sampler` is not of type `LinSysBlkColSampler`, `LinSysVecColSampler`, `LinSysBlkRowSampler`, or `LinSysVecRowSampler`") RLinearAlgebra.log_update!(logger, MadeUpSampler(), z, (A[1, :], b[1]), 0, A, b)
    end

    # Verify late moving average 
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        logger = LSLogMA(lambda2 = 2)
        samp = (A[1,:], b[1])

        RLinearAlgebra.log_update!(logger, sampler, z, (A[1,:],b[1]), 0, A, b)
        # Test moving average of log
        for i = 1:10
            samp = (A[1,:], b[1])
            RLinearAlgebra.log_update!(logger, sampler, x + (i+1)*(z-x), samp, i, A, b)
        end
        #compute sampled residuals
        obs_res = 2 .* [abs(dot(A[1,:],x + (i+1)*(z-x)) - b[1])^2 for i = 0:10]
        obs_res2 = 4 .* [abs(dot(A[1,:],x + (i+1)*(z-x)) - b[1])^4 for i = 0:10]
        @test length(logger.resid_hist) == 11
        @test norm(logger.resid_hist[3:11] - vcat(obs_res[3], 
                                         [(obs_res[i] + obs_res[i-1])/2 for i = 4:11])) < 1e2 * eps()
        @test norm(logger.iota_hist[3:11] - vcat(obs_res2[3], 
                                        [(obs_res2[i] + obs_res2[i-1])/2 for i = 4:11])) < 1e2 * eps()
        @test logger.iteration == 10
        @test logger.converged == false
        
        #Test uncertainty set 
        Uncertainty_set = get_uncertainty(logger)
        @test length(Uncertainty_set[1]) == 11
        #If you undo the steps of the interval calculation should be 1
        @test norm((Uncertainty_set[2] - logger.resid_hist) ./ sqrt.(2 * log(2/.05) * logger.iota_hist * 
                    logger.dist_info.sigma2 .* (1 .+ log.(logger.lambda_hist)) ./  logger.lambda_hist) .- 1) < 1e2 * eps()
    end
    # Verify early moving average
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        logger = LSLogMA(lambda1 = 2,
                     lambda2 = 10)
        samp = (A[1,:], b[1])

        RLinearAlgebra.log_update!(logger, sampler, z, (A[1,:],b[1]), 0, A, b)
        # Test moving average of log when the residual only decreases to not trigger switch
        for i = 1:10
            samp = (A[1,:], b[1])
            RLinearAlgebra.log_update!(logger, sampler, x + .3^(i+1)*(z-x), samp, i, A, b)
        end
        #compute sampled residuals
        obs_res = 2 .* [abs(dot(A[1,:],x + .3^(i+1)*(z-x)) - b[1])^2 for i = 0:10]
        obs_res2 = 4 .* [abs(dot(A[1,:],x + .3^(i+1)*(z-x)) - b[1])^4 for i = 0:10]
        @test length(logger.resid_hist) == 11
        @test norm(logger.resid_hist[3:11] - vcat( [(obs_res[i] + obs_res[i-1])/2 for i = 3:11])) < 1e2 * eps()
        @test norm(logger.iota_hist[3:11] - vcat( [(obs_res2[i] + obs_res2[i-1])/2 for i = 3:11])) < 1e2 * eps()
        @test logger.iteration == 10
        @test logger.converged == false
        
        #Test uncertainty set 
        Uncertainty_set = get_uncertainty(logger)
        @test length(Uncertainty_set[1]) == 11
        #If you undo the steps of the interval calculation should be 1
        @test norm((Uncertainty_set[2] - logger.resid_hist) ./ sqrt.(2 * log(2/.05) * logger.iota_hist * 
                    logger.dist_info.sigma2 .* (1 .+ log.(logger.lambda_hist)) ./  logger.lambda_hist) .- 1) < 1e2 * eps()
    end
    
   


end

end # End Module