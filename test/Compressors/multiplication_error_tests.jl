
module compressor_bound_checking
    using Test, RLinearAlgebra
    import LinearAlgebra: mul!

    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")
    @testset "Multiplication Dimension Checks" begin 
        # Test the matrix multiplication assertions
        mutable struct TestCompressorRecipe <: CompressorRecipe
            n_rows::Int64
            n_cols::Int64
        end
        s = 3
        n_rows = 4
        n_cols = 5
        S = TestCompressorRecipe(s, n_rows)
        # Test the sizes
        let 
            m, n = size(S)
            @test m == s
            @test n == n_rows
            @test n_rows == size(S, 2)
            @test s == size(S, 1)
        end
    
        # Test the matvec multiplications errors
        let
            y = zeros(5)
            x = zeros(3)
            @test_throws DimensionMismatch RLinearAlgebra.vec_mul_dimcheck(x, S, y)
        end
    
        let
            y = zeros(4)
            x = zeros(2)
            @test_throws DimensionMismatch RLinearAlgebra.vec_mul_dimcheck(x, S, y)
            # Test that argument error is thrown when S is on the left
            @test_throws ArgumentError mul!(x, S, y, 1.0, 1.0)
        end
    
        # Test the mat mat multiplication errors
        let
            A = zeros(5, 2)
            C = zeros(3, 2)
            @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
        end
    
        let
            A = zeros(4, 2)
            C = zeros(4, 2)
            @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
        end
    
        let
            A = zeros(4, 2)
            C = zeros(3, 3)
            @test_throws DimensionMismatch RLinearAlgebra.left_mat_mul_dimcheck(C, S, A)
            # Test that argument error is thrown when S is on the right 
            @test_throws ArgumentError mul!(C, A, S, 1.0, 1.0)
        end
    
        let
            A = zeros(4, 3)
            C = zeros(4, 3)
            @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
        end
    
        let
            A = zeros(4, 4)
            C = zeros(4, 2) 
            @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
        end
    
        let
            A = zeros(4, 4)
            C = zeros(5, 3)
            @test_throws DimensionMismatch RLinearAlgebra.right_mat_mul_dimcheck(C, A, S')
        end

    end

end
