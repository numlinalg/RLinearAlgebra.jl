# Date: 09/11/2024
# Author: Christian Varner
# Purpose: Test the arnoldi procedure
# in src/linear_solver_routines/arnoldi_solver.jl

module ProceduralTestArnoldiSolver

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Arnoldi Solver -- Procedural" begin

    # testing context
    Random.seed!(1010)

    # generate linear system
    A     = rand(10, 10)
    xstar = rand(10)
    b     = A * xstar
    x0    = zeros(10)
    r0    = b - A * x0

    # run randomized arnoldi iteration
    k = 5
    H, V = RLinearAlgebra.arnoldi_solver!(x0, A, b, k)

    # gather common matrices
    Vk = V[:, 1:k]
    Vk_tilde = V[:, 1:(k+1)] 

    Hk = H[1:k, 1:k]
    Hk_tilde = H[1:(k+1), 1:k]

    # Check Arnoldi Relation  
    @test norm( A * Vk - Vk_tilde * Hk_tilde ) < eps() * 1e3
    @test norm( Vk' * A * Vk - Hk ) < eps() * 1e3

    # Check that ΩVk is indeed orthonormal
    I = zeros(k, k)
    for i in 1:k
        I[i, i] = 1.0
    end
    @test norm( Vk' * Vk - I ) < eps() * 1e3

    # Check sketched Petrov-Galerkin condition (Ω*rk ⟂ Ω*Vk)
    rk = b - A * x0
    @test norm(Vk' * rk) < eps() * 1e3

    # Check that norm( (Ω * Vk)' * Ω * r0 )[1] = norm(r0)
    @test abs( ( Vk' * r0 )[1] - norm(r0) ) < eps() * 1e3

    # Check that the end result was correctly computed
    ek = zeros(k)
    ek[1] = norm(r0)
    @test norm(x0 - Vk * (Hk \ ek) ) < eps() * 1e3

end # end test

end # end module