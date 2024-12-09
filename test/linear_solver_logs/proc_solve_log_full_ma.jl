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

    # Verify log_update initialization
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA()


        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.resid_hist) == 1
        @test norm(log.resid_hist[1] - norm(A * z - b)^2) < 1e2 * eps()
        @test norm(log.iota_hist[1] - norm(A * z - b)^4) < 1e2 * eps()
        @test log.iterations == 0
        @test log.converged == false
    end
    
    # Verify collection rate
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFullMA(collection_rate = 3)

        # Test initialization of log
        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.resid_hist) == 1
        @test norm(log.resid_hist[1] - norm(A * z - b)^2) < 1e2 * eps()
        @test norm(log.iota_hist[1] - norm(A * z - b)^4) < 1e2 * eps()
        @test log.iterations == 0
        @test log.converged == false

        # Test collection rate of log
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + i*(z-x), (), i, A, b)
        end

        obs_res = [norm(A*(x + i*(z-x)) - b)^2 for i in [1, 3, 6, 9]]
        obs_res2 = [norm(A*(x + i*(z-x)) - b)^4 for i in [1, 3, 6, 9]]
        norm(log.resid_hist[2:4] - vcat( [(obs_res[i] + obs_res[i-1])/2 for i = 2:4]))

        @test length(log.resid_hist) == 4 # Record at 0, 3, 6, 9
        @test norm()
        @test norm(log.resid_hist - [1.0, 3.0, 6.0, 9.0] * norm(A * z - b)^2) < 1e-15
        @test log.iterations == 10
        @test log.converged == false
    end

end

end # End Module
