module compressor_abstract_dimchecks
using Test, RLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..ApproxTol

#####################
# Testing Parameters
##################### 
mutable struct TestCompressorRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
end
s = 3
n_rows = 4
n_cols = 5
S = TestCompressorRecipe(s, n_rows)

@testset "Multiplication Dimension Checks" begin
    ########################
    # Matrix-Vector methods 
    ########################

    # Error for S*y when y has incorrect dimension 
    let S=deepcopy(S), x=zeros(s), y=zeros(n_rows+1)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(x, S, y)
    end

    # Error for S*y -> x, when x has incorrect dimension 
    let S=deepcopy(S), y=zeros(n_rows), x=zeros(s+1) 
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(x, S, y)
    end
    
    # Correct for S*y -> x 
    let S=deepcopy(S), x=zeros(s), y = zeros(n_rows)
        @test isnothing(RLinearAlgebra.left_mul_dimcheck(x, S, y))
    end

    # Error for y'*S -> x, when y has incorrect dimension 
    let S=deepcopy(S), x=zeros(n_rows)', y=zeros(s+1)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(x, y', S)
    end

    # Error for y'*S -> x, when x has incorrect dimension 
    let S=deepcopy(S), x=zeros(n_rows+1)', y=zeros(s) 
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(x, y', S)
    end
    
    # Correct for y'*S -> x
    let S=deepcopy(S), x=zeros(n_rows), y=zeros(s) 
        @test isnothing(RLinearAlgebra.right_mul_dimcheck(x', y', S))
    end

    # Error for S'*y when y has incorrect dimension
    let S=deepcopy(S)', x=zeros(n_rows), y=zeros(s+1)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(x, S, y)
    end
    
    # Error for S'*y -> x, when x has incorrect dimension 
    let S=deepcopy(S)', x=zeros(n_rows+1), y=zeros(s)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(x, S, y)
    end

    # Correct for S'*y -> x 
    let S=deepcopy(S)', x=zeros(n_rows), y=zeros(s) 
        @test isnothing(RLinearAlgebra.left_mul_dimcheck(x, S, y))
    end

    # Error for y'*S' -> x, when y has incorrect dimension 
    let S=deepcopy(S)', x=zeros(s)', y=zeros(n_rows+1)'
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(x, y, S)
    end

    # Error for y'*S' -> x, when x has incorrect dimension 
    let S=deepcopy(S)', x=zeros(s+1)', y=zeros(n_rows)' 
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(x, y, S)
    end
    
    # Correct for y'*S' -> x
    let S=deepcopy(S)', x=zeros(s)', y=zeros(n_rows)' 
        @test isnothing(RLinearAlgebra.right_mul_dimcheck(x, y, S))
    end

    ########################
    # Matrix-Matrix methods 
    ########################
    #TODO: Matrix-Matrix methos dimchecks
    # Error for S*x
    # let S = deepcopy(S), 
    #     # Here C has correct column dimension and row dimension A has incorrect row dim
    #     A = zeros(5, 2)
    #     C = zeros(3, 2)
    #     @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    # end

    # let
    #     # C has the wrong row dimension but correct column dimension and A two correct 
    #     # dimensions
    #     A = zeros(4, 2)
    #     C = zeros(4, 2)
    #     @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    # end

    # let
    #     # Here A has correct row and column dimensions and C has incorrect column 
    #     # dimension
    #     A = zeros(4, 2)
    #     C = zeros(3, 3)
    #     @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    # end

    # # Test right multiplication using S' to avoid generating another S
    # # S' has a dimension 4 by 3
    # let
    #     # Here A has the correct row dimension and incorrect column dimension
    #     # C has the correct row and column dimensions
    #     A = zeros(4, 3)
    #     C = zeros(4, 3)
    #     @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S')
    # end

    # let
    #     # Here A has the correct row dimension and correct column dimension
    #     # C has the correct row and incorrect column dimensions
    #     A = zeros(4, 4)
    #     C = zeros(4, 2)
    #     @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S')
    # end

    # let
    #     # Here A has the correct row dimension and correct column dimension
    #     # C has the incorrect row and correct column dimensions
    #     A = zeros(4, 4)
    #     C = zeros(5, 3)
    #     @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S')
    # end
end

end