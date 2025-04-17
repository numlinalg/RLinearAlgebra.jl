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

    # Error S*A when A has incorrect dimensions
    let S=deepcopy(S), A=zeros(n_rows+1, n_cols), C=zeros(s, n_cols)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end 

    # Error S*A -> C when C has incorrect dimensions 
    let S=deepcopy(S), A=zeros(n_rows, n_cols), C=zeros(s+1, n_cols)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end
    let S=deepcopy(S), A=zeros(n_rows, n_cols), C=zeros(s, n_cols+1)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end

    # Correct dimensions for S*A -> C
    let S=deepcopy(S), A=zeros(n_rows, n_cols), C=zeros(s, n_cols)
        @test isnothing(RLinearAlgebra.left_mul_dimcheck(C, S, A))
    end

    # Error A*S when A has incorrect dimensions, n_cols has no meaning here 
    let S=deepcopy(S), A=zeros(n_cols, s+1), C=zeros(n_cols, n_rows)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end

    # Error A*S -> C when C has incorrect dimensions, n_cols has no meaning here 
    let S=deepcopy(S), A=zeros(n_cols, s), C=zeros(n_cols+1, n_rows)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end
    let S=deepcopy(S), A=zeros(n_cols, s), C=zeros(n_cols, n_rows+1)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end

    # Correct dimensions for A*S -> C
    let S=deepcopy(S), A=zeros(n_cols, s), C=zeros(n_cols, n_rows)
        @test isnothing(RLinearAlgebra.right_mul_dimcheck(C, A, S))
    end

    # Error A*S' when A has incorect dimensions 
    let S=deepcopy(S)', A=zeros(n_cols, n_rows+1), C=zeros(n_cols, s)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end

    # Error A*S' -> C when C has incorrect dimensions 
    let S=deepcopy(S)', A=zeros(n_cols, n_rows), C=zeros(n_cols+1, s)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end
    let S=deepcopy(S)', A=zeros(n_cols, n_rows), C=zeros(n_cols, s+1)
        @test_throws DimensionMismatch RLinearAlgebra.right_mul_dimcheck(C, A, S)
    end

    # Correct dimensions for A*S' -> C 
    let S=deepcopy(S)', A=zeros(n_cols, n_rows), C=zeros(n_cols, s)
        @test isnothing(RLinearAlgebra.right_mul_dimcheck(C, A, S))
    end

    # Error S'*A when A has incorrect dimensions 
    let S=deepcopy(S)', A=zeros(s+1, n_cols), C=zeros(n_rows, n_cols)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end

    # Error S'*A -> C when C has incorrect dimensions 
    let S=deepcopy(S)', A=zeros(s, n_cols), C=zeros(n_rows+1, n_cols)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end
    let S=deepcopy(S)', A=zeros(s, n_cols), C=zeros(n_rows, n_cols+1)
        @test_throws DimensionMismatch RLinearAlgebra.left_mul_dimcheck(C, S, A)
    end

    # Correct dimensions for S'*A' -> C
    let S=deepcopy(S)', A=zeros(s, n_cols), C=zeros(n_rows, n_cols)
        @test isnothing(RLinearAlgebra.left_mul_dimcheck(C, S, A))

    end
end

end