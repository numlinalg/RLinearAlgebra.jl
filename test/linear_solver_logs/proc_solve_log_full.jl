# This file is part of RLinearAlgebra.jl

module ProceduralTestLSLogFull

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LS Full Log -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LSLogFull) == LinSysSolverLog

    # Verify Required Fields
    @test :iterations in fieldnames(LSLogFull)
    @test :converged in fieldnames(LSLogFull)

    # Verify log_update initialization
    let
        A = rand(2,2)
        x = rand(2)
        b = A * x
        z = rand(2)

        sampler = LinSysVecRowOneRandCyclic()
        log = LSLogFull()


        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

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
        log = LSLogFull(3)

        # Test initialization of log
        RLinearAlgebra.log_update!(log, sampler, z, (), 0, A, b)

        @test length(log.resid_hist) == 1
        @test log.resid_hist[1] == norm(A * z - b)
        @test log.iterations == 0
        @test log.converged == false

        # Test collection rate of log
        for i = 1:10
            RLinearAlgebra.log_update!(log, sampler, x + i*(z-x), (), i, A, b)
        end

        @test length(log.resid_hist) == 4 # Record at 0, 3, 6, 9
        @test norm(log.resid_hist - [1.0, 3.0, 6.0, 9.0] * norm(A * z - b)) < 1e-15
        @test log.iterations == 10
        @test log.converged == false
    end

end

end # End Module
