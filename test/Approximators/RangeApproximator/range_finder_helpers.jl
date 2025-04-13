module  RangeApproximator_helper
using Test, RLinearAlgebra, LinearAlgebra
using ..FieldTest
using ..ApproxTol

struct TestCompressorRecipe <: RLinearAlgebra.CompressorRecipe 
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

# Define a mul function for the test compressor
function RLinearAlgebra.mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::Main.RangeApproximator_helper.TestCompressorRecipe, 
    alpha::Float64, 
    beta::Float64
)
    mul!(C, A, S.op, alpha, beta)
end

mutable struct TestRangeApproximatorRecipe <: RLinearAlgebra.RangeApproximatorRecipe
    compressor::CompressorRecipe
    power_its::Int64
end
@testset "Range Finder Helpers" begin
    n_rows = 6
    n_cols = 4
    compression_dim = 2
    # Initialize the compressor with a Gaussian Sketch in this case scaling of matrix
    # is unimportant
    C = TestCompressorRecipe(n_cols, compression_dim, randn(n_cols, compression_dim)) 
    A = rand(n_rows, n_cols)
    @testset "Power Iteration Test" begin 
        # test with 0 power iterations
        let 
            # Start by testing the case with zero power iterations
            approx = TestRangeApproximatorRecipe(C, 0)
            # Produce the Q from the function call
            Q_func = RLinearAlgebra.rand_power_it(A, approx)
            Q_test = Array(qr(A*C.op).Q)
            Q_func ≈ Q_test
            # test the matrix is orthogonal
            gram_matrix = Q_func' * Q_func
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
        end

        # test with 3 power iterations
        let
            # Perform same test with 3 power iterations
            approx = TestRangeApproximatorRecipe(C, 3)
            Q_func = RLinearAlgebra.rand_power_it(A, approx)
            Q_test = Array(qr(A * A' * A * A' * A * A'* A * C.op).Q)
            @test Q_func ≈ Q_test
            # test the matrix is orthogonal
            gram_matrix = Q_func' * Q_func
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
        end
    
    end
    
    @testset "Subspace Iteration Test" begin 
        let 
            # Start by testing the case with zero power iterations
            approx = TestRangeApproximatorRecipe(C, 0)
            # Produce the Q from the function call
            Q_func = RLinearAlgebra.rand_subspace_it(A, approx)
            Q_test = Array(qr(A*C.op).Q)
            Q_func ≈ Q_test
            # test the matrix is orthogonal
            gram_matrix = Q_func' * Q_func
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
        end

        # test with multiple power iterations
        let
            approx = TestRangeApproximatorRecipe(C, 3)
            Q_func = RLinearAlgebra.rand_subspace_it(A, approx)
            # Perform the 3 subspace iterations which involves orthogonalzing everytime
            # we apply A or A' to the Q matrix
            Q1 = Array(qr(A * C.op).Q)
            Q2 = Array(qr(A' * Q1).Q)
            Q1 = Array(qr(A * Q2).Q)
            Q2 = Array(qr(A' * Q1).Q)
            Q1 = Array(qr(A * Q2).Q)
            Q2 = Array(qr(A' * Q1).Q)
            Q_test = Array(qr(A * Q2).Q)
            #Q_test = Array(qr(A * A' * A * A' * A * A'* A * C.op).Q)
            @test Q_func ≈ Q_test
            # test that the matrix is orthogonal 
            gram_matrix = Q_func' * Q_func
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end

        end
    
    end

end

end
