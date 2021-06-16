# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogOracle

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LS Oracle Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogOracle) == LinSysSolverLog

    # Verify Required Fields
    @test :iterations in fieldnames(LSLogOracle)
    @test :converged in fieldnames(LSLogOracle)

    # Verify log_update initialization
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogOracle(x)


        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.error_hist) == 1
        @test log.error_hist[1] == norm(z - x)
        @test length(log.resid_hist) == 1
        @test log.resid_hist[1] == norm(A * z - b)
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
        log = LSLogOracle(x, 3)

        # Test initialization of log
        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.error_hist) == 1
        @test log.error_hist[1] == norm(z - x)
        @test length(log.resid_hist) == 1
        @test log.resid_hist[1] == norm(A * z - b)
        @test log.iterations == 0
        @test log.converged == false

        # Test collection rate of log
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + i*(z-x), (), i, A, b)
        end

        @test length(log.error_hist) == 4 # Record at 0, 3, 6, 9
        @test norm(log.error_hist - [1.0, 3.0, 6.0, 9.0] * norm(z - x)) < 1e-15
        @test length(log.resid_hist) == 4 # Record at 0, 3, 6, 9
        @test norm(log.resid_hist - [1.0, 3.0, 6.0, 9.0] * norm(A * z - b)) < 1e-15
        @test log.iterations == 10
        @test log.converged == false
    end


end


end # End Module
