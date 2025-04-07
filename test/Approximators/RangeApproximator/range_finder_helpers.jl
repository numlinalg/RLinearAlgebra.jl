module  RangeApproximator_helper
using Test, RLinearAlgebra, LinearAlgebra
include("../../test_helpers/field_test_macros.jl")
include("../../test_helpers/approx_tol.jl")
struct TestCompressorRecipe <: CompressorRecipe 
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

# Define a mul function for the test compressor
function mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::CompressorRecipe, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, S.op, alpha, beta)
end

mutable struct TestRangeApproximator <: RangeApproximator
    compressor::CompressorRecipe
    power_its::Int64
end

@testset "Power Iteration Test" begin 
    let 
        n_rows = 6
        n_cols = 4
        compression_dim = 2
        # Initialize the compressor with a Gaussian Sketch in this case scaling of matrix
        # is unimportant
        C = TestCompressorRecipe(n_cols, compression_dim, randn(n_cols, compression_dim)) 
        A = rand(n_rows, n_cols)
        # Start by testing the case with zero power iterations
        approx = TestRangeApproximator(C, 0)
        # Produce the Q from the function call
        Q_func = rand_power_it(A, approx)
        Q_test = Array(qr(A*C.op).Q)
        Q_func ≈ Q_test

        # Perform same test with 3 power iterations
        approx.power_its = 3
        Q_func = rand_power_it(A, approx)
        Q_test = Array(qr(A * A' * A * A' * A * A'* A * C.op).Q)
        Q_func ≈ Q_test

    end

end

@testset "Subspace Iteration Test" begin 
    let 
        n_rows = 6
        n_cols = 4
        compression_dim = 2
        # Initialize the compressor with a Gaussian Sketch in this case scaling of matrix
        # is unimportant
        C = TestCompressorRecipe(n_cols, compression_dim, randn(n_cols, compression_dim)) 
        A = rand(n_rows, n_cols)
        # Start by testing the case with zero power iterations
        approx = TestRangeApproximator(C, 0)
        # Produce the Q from the function call
        Q_func = rand_subsapce_it(A, approx)
        Q_test = Array(qr(A*C.op).Q)
        Q_func ≈ Q_test

        # Perform same test with 3 subspace iterations
        approx.power_its = 3
        Q_func = rand_power_it(A, approx)
        # Perform the 3 subspace iterations which involves orthogonalzing everytime
        # we apply A or A' to the Q matrix
        Q1 = Array(qr(A * C.op).Q)
        Q2 = Array(qr(A' * Q1).Q)
        Q1 = Array(qr(A * Q2).Q)
        Q2 = Array(qr(A' * Q1).Q)
        Q1 = Array(qr(A * Q2).Q)
        Q2 = Array(qr(A' * Q1).Q)
        Q_test = Array(qr(A * Q2).Q)
        Q_test = Array(qr(A * A' * A * A' * A * A'* A * C.op).Q)
        Q_func ≈ Q_test

    end

end
end
