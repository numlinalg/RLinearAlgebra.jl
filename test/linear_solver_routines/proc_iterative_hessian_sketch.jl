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

    rsub = IterativeHessianSketch(A,b,nothing,nothing) 

    # verify correct assignment
    @test rsub.A == A
    @test rsub.b == b

    # Verify that buffer arrays are nothing
    @test isnothing(rsub.step)
    @test isnothing(rsub.btilde)

    # one step of the algorithm
    x0 = zeros(5)
    S = randn(5,length(b))
    RLinearAlgebra.rsubsolve!(rsub, x0, (S, S*A, S*A*x0 - S*b), 1)

    # check to make sure btilde and step get initialized to the correct length
    @test length(rsub.btilde) == 5
    @test length(rsub.step) == 5

    # check to make sure that the inner problem is solved correctly
    @test norm(A'*S'*S*A*rsub.step - size(S)[1]*A'*(b-A*zeros(5)) ) < 1e-11

    # check update is correct
    @test x0 == zeros(5) + rsub.step

    # check final solutions
    rsub = IterativeHessianSketch(A,b,nothing,nothing)
    x0 = zeros(5)
    for i in 1:100
        S = randn(5*10,length(b))
        RLinearAlgebra.rsubsolve!(rsub, x0, (S, S*A, S*A*x0 - S*b), i)
    end
    ldivsol = zeros(5)
    LinearAlgebra.ldiv!(ldivsol,qr(A'*A),A'*b)
    @test norm(x0-ldivsol) < 1e-11

    # check to see that we used ldiv!
    x0 = zeros(5)
    S = randn(4,length(b))
    RLinearAlgebra.rsubsolve!(rsub, x0, (S, S*A, S*A*x0 - S*b), 1)

    step = zeros(5)
    LinearAlgebra.ldiv!(step, qr((S*A)'*S*A), size(S)[1]*A'*(b-A*zeros(5)))
    @test step == rsub.step
    
end

end # End module