# Date: 12/03/2024
# Author: Christian
# Purpose: Test the arnoldi iteration

module ProceduralArnoldi

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Arnoldi Iteration -- Procedural" begin

    Random.seed!(1010)

    #####################
    # test definition
    #####################
    @test isdefined(RLinearAlgebra, :arnoldi)

    #####################
    # test errors
    #####################
    
    # not square
    nrow = 100
    ncol = 50
    A = randn(nrow, ncol)
    q = randn(ncol)
    k = 10
    @test_throws AssertionError RLinearAlgebra.arnoldi(A, q, k)

    # q is wrong dimension
    A = randn(nrow, nrow)
    q = randn(nrow-1)
    k = 10
    @test_throws AssertionError RLinearAlgebra.arnoldi(A, q, k)

    # k is not greater than or equal to 1
    A = randn(nrow, nrow)
    q = randn(nrow)
    k = 0
    @test_throws AssertionError RLinearAlgebra.arnoldi(A, q, k)

    ###########################
    # Compute orthonormal basis
    # and check conditions
    ###########################
    A = randn(nrow, nrow)
    q = randn(nrow)
    k = rand(collect(1:nrow))
    Q, H = RLinearAlgebra.arnoldi(A, q, k)

    # matrix conditions
    Q_prev = Q[:, 1:(k-1)]
    @test A * Q_prev ≈ Q * H atol = 1e-10
    
    H_prev = H[1:(k-1), 1:(k-1)]
    @test Q_prev' * A * Q_prev ≈ H_prev atol = 1e-10

    # orthogonality
    eye = Matrix{Float64}(I, k, k)
    @test Q' * Q ≈ eye atol = 1e-10 
end

end