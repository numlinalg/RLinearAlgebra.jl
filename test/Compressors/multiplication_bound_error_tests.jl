
module compressor_bound_checking
using Test, RLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..ApproxTol
@testset "Multiplication Dimension Checks" begin
    # Testing Parameters 
    mutable struct TestCompressorRecipe <: CompressorRecipe
        n_rows::Int64
        n_cols::Int64
    end
    s = 3
    n_rows = 4
    n_cols = 5

    # S is 3 by 4
    S = TestCompressorRecipe(s, n_rows)
    
    ####################
    # Size methods 
    ####################
    let S = deepcopy(S), s=s, n_rows=n_rows, n_cols=n_cols

        # Get size 
        m, n = size(S)
        @test m == s
        @test n == n_rows

        # Get individual sizes 
        @test n_rows == size(S, 2)
        @test s == size(S, 1)

        # Make sure transpose size are opposite of original size
        @test size(S, 1) == size(S', 2)
        @test size(S, 2) == size(S', 1)
    end

    ########################
    # Matrix-Vector methods 
    ########################

    # Error for S*y -> x, when y has incorrect dimension 
    let S = deepcopy(S), s=s, n_cols=n_cols
        y = zeros(n_cols) # Should be n_rows 
        x = zeros(s)
        @test_throws DimensionMismatch RLinearAlgebra.vec_mul_dimcheck(x, S, y)
    end

    # Error for S*y -> x, when x has incorrect dimension 
    let S = deepcopy(S), s=s, n_rows=n_rows 
        y = zeros(n_rows)
        x = zeros(s+1)
        @test_throws DimensionMismatch RLinearAlgebra.vec_mul_dimcheck(x, S, y)
    end

    # Test the mat mat multiplication errors
    let
        # Here C has correct column dimension and row dimension A has incorrect row dim
        A = zeros(5, 2)
        C = zeros(3, 2)
        @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
    end

    let
        # C has the wrong row dimension but correct column dimension and A two correct 
        # dimensions
        A = zeros(4, 2)
        C = zeros(4, 2)
        @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
    end

    let
        # Here A has correct row and column dimensions and C has incorrect column 
        # dimension
        A = zeros(4, 2)
        C = zeros(3, 3)
        @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
    end

    # Test right multiplication using S' to avoid generating another S
    # S' has a dimension 4 by 3
    let
        # Here A has the correct row dimension and incorrect column dimension
        # C has the correct row and column dimensions
        A = zeros(4, 3)
        C = zeros(4, 3)
        @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
    end

    let
        # Here A has the correct row dimension and correct column dimension
        # C has the correct row and incorrect column dimensions
        A = zeros(4, 4)
        C = zeros(4, 2)
        @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
    end

    let
        # Here A has the correct row dimension and correct column dimension
        # C has the incorrect row and correct column dimensions
        A = zeros(4, 4)
        C = zeros(5, 3)
        @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
    end
end

end
