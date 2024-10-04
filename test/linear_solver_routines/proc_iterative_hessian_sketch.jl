# This file is part of RLinearAlgebra.jl
# Date: 06/20/2024
# Author: Christian Varner
# Purpose: Test the iterative hessian sketch procedure
# found in iterative_hessian_sketch.jl

module ProceduralTestIterativeHessianSketch

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Iterative Hessian Sketch -- Procedural" begin

    # Supertype
    @test supertype(IterativeHessianSketch) == LinSysSolveRoutine

    # Testing context
    Random.seed!(1010)

    A = rand(10,5)
    x = rand(5)
    b = A * x

    
    ########################## Test one iteration of the algorithm initialized as intended.
    ########################## Sketch size is >= number of columns in A.
    rsub = IterativeHessianSketch(A, b, nothing, nothing) # btilde, step

    # verify correct assignment
    @test rsub.A == A
    @test rsub.b == b

    # Verify that buffer arrays are nothing
    @test isnothing(rsub.step)
    @test isnothing(rsub.btilde)

    # one step of the algorithm
    x0 = zeros(5)
    S = randn(5, length(b))
    RLinearAlgebra.rsubsolve!(rsub, x0, (S, S * A, S * A * x0 - S * b), 1)

    # check to make sure btilde and step get initialized to the correct length
    @test length(rsub.btilde) == 5
    @test length(rsub.step) == 5

    # test to make sure rsub.btilde is correctly initialized.
    @test size(S)[1] * A' * (b - A * zeros(5)) ≈ rsub.btilde

    # check to make sure that the inner problem is solved correctly (correct step)
    R = qr(S * A).R
    @test norm(R' * R * rsub.step - size(S)[1] * A' * (b - A * zeros(5))) < 1e-10

    # check update is correct
    @test x0 == zeros(5) + rsub.step
    ##########################
    ##########################

    ########################## Test one step of the algorithm when values step and btilde
    ########################## are not set as intended. Sketch size >= number of columns in A.
    btilde_random = randn(123)
    step_random = randn(123)
    rsub = IterativeHessianSketch(A, b, step_random, btilde_random)    

    # test user inputed values
    @test size(rsub.btilde)[1] == 123
    @test size(rsub.step)[1] == 123 
    @test rsub.btilde == btilde_random
    @test rsub.step == step_random

    # one step of the algorithm
    x0 = zeros(5)
    S = randn(5, length(b))
    RLinearAlgebra.rsubsolve!(rsub, x0, (S, S * A, S * A * x0 - S * b), 1)

    # User inputed values should be rewritted on the first iteration
    @test length(rsub.btilde) == 5
    @test length(rsub.step) == 5

    # test to make sure rsub.btilde is correctly initialized.
    @test size(S)[1] * A' * (b - A * zeros(5)) ≈ rsub.btilde 
    
    # check to make sure that the inner problem is solved correctly (correct step)
    R = qr(S * A).R
    @test norm(R' * R * rsub.step - size(S)[1] * A' * (b - A * zeros(5))) < 1e-10

    # check update is correct
    @test x0 == zeros(5) + rsub.step
    ##########################
    ##########################

    ########################## Sketch size >= number of columns
    ##########################
    # check final solutions
    rsub = IterativeHessianSketch(A, b, nothing, nothing)
    x0 = zeros(5)
    for i in 1:100
        S = randn(50, length(b))
        RLinearAlgebra.rsubsolve!(rsub, x0, (S, S * A, S * A * x0 - S * b), i)
    end

    ldivsol = zeros(5)
    LinearAlgebra.ldiv!(ldivsol, cholesky(A' * A), A' * b)
    @test norm(x0-ldivsol) < 1e-10
    ##########################
    ##########################

    ########################## One iteration of the algorithm when sketch size < number of columns
    ##########################
    rsub = IterativeHessianSketch(A, b, nothing, nothing) # btilde, step

    # one step of the algorithm
    x0 = zeros(5)
    S = randn(1, length(b))
    RLinearAlgebra.rsubsolve!(rsub, x0, (S, S * A, S * A * x0 - S * b), 1)

    step_buffer = zeros(5)
    B = (S*A)' * (S * A) 
    b_sub_system = (size(S)[1] * A' * (b - A * zeros(5)))

    @test norm(B' * B * rsub.step - B' * b_sub_system) < eps() * 1e4
    @test x0 ≈ zeros(5) + rsub.step
    @test_logs (:warn, "The sampler's block_size might be too small for sensible inner problem solution. Algorithm will continue by solving the least squares problem instead (Caution: No theory).") RLinearAlgebra.rsubsolve!(rsub, x0, (S, S * A, S * A * x0 - S * b), 1) 
    ##########################
    ##########################    

end

end # End module