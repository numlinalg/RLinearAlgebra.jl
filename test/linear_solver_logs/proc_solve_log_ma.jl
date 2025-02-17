# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogMA

using Test, RLinearAlgebra, Random, LinearAlgebra
# Used for collect all the samplers
include("logs_helpers/linear_samplers_collector.jl")

Random.seed!(1010)

@testset "LS MA Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogMA) == LinSysSolverLog

    # Verify Required Fields
    @test :iterations in fieldnames(LSLogMA)
    @test :converged in fieldnames(LSLogMA)

    # Initialize data
    A = rand(2,2)
    x = rand(2)
    b = A * x
    z = rand(2)

    # Verify log_update initialization
    let A = A, x = x, b = b, z = z

        sampler = LinSysVecRowOneRandCyclic()
        logger = LSLogMA()

        # Test the error message on the get_uncertainty function
        @test_throws ArgumentError("The SE constants are empty, please set them in dist_info field of LSLogMA first.") get_uncertainty(logger) 
        # Test warning message in default SEconstant look up

        RLinearAlgebra.log_update!(logger, sampler, z, (A[1, :], b[1]), 0, A, b)

        @test length(logger.resid_hist) == 1
        @test norm(logger.resid_hist[1] - 2 * norm(dot(A[1, :], z) - b[1])^2) < 1e2 * eps()
        @test norm(logger.iota_hist[1] - 4 * norm(dot(A[1, :], z) - b[1])^4) < 1e2 * eps()
        @test logger.iterations == 0
        @test logger.converged == false
        
        struct MadeUpSampler <: LinSysSampler  
        end

        @test_logs (:warn, "No constants defined for method of type Main.ProceduralTestLSLogMA.MadeUpSampler. By default we set sigma2 to 1 and scaling to 1.") RLinearAlgebra.get_SE_constants!(logger, MadeUpSampler)
        @test_throws ArgumentError("`sampler` is not of type `LinSysBlkColSampler`, `LinSysVecColSampler`, `LinSysBlkRowSampler`, or `LinSysVecRowSampler`") RLinearAlgebra.log_update!(logger, MadeUpSampler(), z, (A[1, :], b[1]), 0, A, b)
    end

    # Verify late moving average 
    let A = A, x = x, b = b, z = z

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
        @test_skip norm(logger.iota_hist[3:11] - vcat(obs_res2[3], 
                                         [(obs_res2[i] + obs_res2[i-1])/2 for i = 4:11])) < 1e2 * eps()
        @test logger.iterations == 10
        @test logger.converged == false
        
        #Test uncertainty set 
        Uncertainty_set = get_uncertainty(logger)
        @test length(Uncertainty_set[1]) == 11
        #If you undo the steps of the interval calculation should be 1
        @test norm((Uncertainty_set[2] - logger.resid_hist) ./ sqrt.(2 * log(2/.05) * logger.iota_hist * 
                    logger.dist_info.sigma2 .* (1 .+ log.(logger.lambda_hist)) ./  logger.lambda_hist) .- 1) < 1e2 * eps()
    end


    # Verify early moving average
    let A = A, x = x, b = b, z = z

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
        @test logger.iterations == 10
        @test logger.converged == false
        
        #Test uncertainty set 
        Uncertainty_set = get_uncertainty(logger)
        @test length(Uncertainty_set[1]) == 11
        #If you undo the steps of the interval calculation should be 1
        @test norm((Uncertainty_set[2] - logger.resid_hist) ./ sqrt.(2 * log(2/.05) * logger.iota_hist * 
                    logger.dist_info.sigma2 .* (1 .+ log.(logger.lambda_hist)) ./  logger.lambda_hist) .- 1) < 1e2 * eps()
    end

    # Verify it can work with all types of samplers
    # Vector samplers 
    let A = A, x = x, b = b, z = z
        # sampler_types = collect_samplers("vec")
        # samplers = [T() for T in sampler_types]
        samplers = [ LinSysVecRowDetermCyclic(),
                     LinSysVecRowHopRandCyclic(),
                     LinSysVecRowOneRandCyclic(),
                     LinSysVecRowPropToNormSampler(),
                     LinSysVecRowSVSampler(),
                     LinSysVecRowRandCyclic(),
                     LinSysVecRowUnidSampler(),
                     LinSysVecRowUnifSampler(),
                     LinSysVecRowGaussSampler(),
                     LinSysVecRowSparseUnifSampler(),
                     LinSysVecRowSparseGaussSampler(),
                     LinSysVecRowMaxResidual(),
                     LinSysVecRowMaxDistance(),
                     LinSysVecRowResidCyclic(),
                     LinSysVecRowDistCyclic(),
                     LinSysVecColDetermCyclic(),
                     LinSysVecColOneRandCyclic()
                    ]

        for sampler in samplers
            logger = LSLogMA(lambda2 = 2)
            # Ensure no warnings are thrown (valid constants exist)
            
            if typeof(sampler) <: Union{RLinearAlgebra.LinSysVecRowUnifSampler, RLinearAlgebra.LinSysVecRowSparseUnifSampler}
                @test_logs (:warn, "No constants defined for method of type $(typeof(sampler)). By default we set sigma2 to 1 and scaling to 1.") RLinearAlgebra.log_update!(logger, sampler, z, (A[1, :], b[1]), 0, A, b)
            end
        end
    end

    # # Block samplers
    # let
    #     A = rand(10, 5) 
    #     x = rand(5)  
    #     b = A * x         
    #     z = rand(5)

    #     sampler_types = collect_samplers("blk")
    #     samplers = [T() for T in sampler_types]
    #     types_to_add = [LinSysBlkRowCountSketch, LinSysBlkRowSelectWoReplacement, LinSysBlkColCountSketch, LinSysBlkColSelectWoReplacement]

    #     for BlkSamplerType in types_to_add
    #         if !any(T === BlkSamplerType for T in sampler_types)
    #             push!(samplers, BlkSamplerType())
    #         end
    #     end
        
    #     block_size = 2
    #     block_indices = 1:block_size     
    #     residual_block = A[block_indices, :] * z - b[block_indices] 

    #     for sampler in samplers
    #         _ = RLinearAlgebra.sample(sampler, A, b, x, 1)
    #         logger = LSLogMA(lambda2 = 2)
    #         @test_logs (:warn, "No constants defined for method of type $(typeof(sampler)). By default we set sigma2 to 1 and scaling to 1.") RLinearAlgebra.log_update!(logger, sampler, z, (A[1:2, 1:2], b[1:2], residual_block), 0, A, b)
    #     end
    # end
 


end

end # End Module
