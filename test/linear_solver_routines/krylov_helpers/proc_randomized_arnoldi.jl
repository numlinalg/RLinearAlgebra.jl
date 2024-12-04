# Date: 12/04/2024
# Author: Christian Varner 
# Purpose: Test the randomized arnoldi iteration

module ProceduralRandomizedArnoldi

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Randomized Arnoldi -- Procedural" begin

    Random.seed!(1010)

    # test to make sure it is defined
    @test isdefined(RLinearAlgebra, :randomized_arnoldi)

    #############################
    # Error Testing
    #############################
    
    # A is not square
    nrow = 100
    sketch_row = 50
    A = randn(nrow, nrow-1)
    q = randn(nrow)
    Omega = randn(sketch_row, nrow)
    k = 5
    @test_throws AssertionError RLinearAlgebra.randomized_arnoldi(A, q, Omega, k) 

    # q is wrong size
    A = randn(nrow, nrow)
    q = randn(nrow-1)
    Omega = randn(sketch_row, nrow)
    k = 10
    @test_throws AssertionError RLinearAlgebra.randomized_arnoldi(A, q, Omega, k)

    # Omega has the wrong number of columns
    A = randn(nrow, nrow)
    q = randn(nrow)
    Omega = randn(sketch_row, nrow - 1)
    k = 10
    @test_throws AssertionError RLinearAlgebra.randomized_arnoldi(A, q, Omega, k)

    # k is smaller than 1
    A = randn(nrow, nrow)
    q = randn(nrow)
    Omega = randn(sketch_row, nrow)
    k = 0
    @test_throws AssertionError RLinearAlgebra.randomized_arnoldi(A, q, Omega, k)

    #############################
    # Test output
    #############################

    A = randn(nrow, nrow)
    q = randn(nrow)
    Omega = randn(sketch_row, nrow) 
    k = rand(collect(1:sketch_row))
    output = RLinearAlgebra.randomized_arnoldi(A, q, Omega, k)

    # test return value length and type
    @test size(output, 1) == 3
    for i in 1:3
        @test typeof(output[i]) == Matrix{Float64}
    end
    Q, S, H = output[1], output[2], output[3]

    # test output shapes
    @test size(Q) == (nrow, k)
    @test size(S) == (sketch_row, k)
    @test size(H) == (k, k - 1)

    # test matrix conditions
    Q_prev = Q[:, 1:(k-1)]
    @test norm(A * Q_prev - Q * H) ≈ 0 atol = 1e-10
    
    H_prev = H[1:(k-1), 1:(k-1)]
    @test norm( (Omega * Q_prev)' * Omega * A * Q_prev - H_prev) ≈ 0 atol = 1e4

    # test that H has the correct structure
    for i in 3:k
        for j in 1:i-2
            @test H[i, j] == 0
        end
    end 

    # test that S is orthogonal
    eye = Matrix{Float64}(I, k, k)
    @test norm(S' * S - eye) ≈ 0 atol = 1e-10

    # test that Omega * Q is orthogonal
    ΩQ = Omega * Q
    @test norm(ΩQ' * ΩQ - eye) ≈ 0 atol = 1e-10

    # test that Omega * Q = S
    @test norm(ΩQ - S) ≈ 0 atol = 1e-10

end

end