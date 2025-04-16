module compressor_abstract_multiplication
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
TestCompressorRecipe() = TestCompressorRecipe(s, n_rows)

@testset "Compressor Recipe Multiplication Errors" begin 
    # Test Set Parameters 
    x = randn(2)
    y = randn(2)
    A = randn(2, 2)
    C = randn(2, 2) 

    # Five argument muls 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe(), A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe(), 1.0, 1.0)
    @test_throws ArgumentError mul!(x, TestCompressorRecipe(), y, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe(), 1.0, 1.0)

    # Three argument muls 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe(), A)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe())
    @test_throws ArgumentError mul!(x, TestCompressorRecipe(), y)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe())

    # Binary muls 
    @test_throws ArgumentError TestCompressorRecipe()*A 
    @test_throws ArgumentError A*TestCompressorRecipe() 
    @test_throws ArgumentError TestCompressorRecipe()*y
    @test_throws ArgumentError y*TestCompressorRecipe()

    # Five argument muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()', 1.0, 1.0)
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x, 1.0, 1.0)
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe()', 1.0, 1.0)

    # Three arguments muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()')
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x)
    @test_throws ArgumentError mul!(y, x, TestCompressorRecipe()')

    # Binary muls 
    @test_throws ArgumentError TestCompressorRecipe()'*A 
    @test_throws ArgumentError A*TestCompressorRecipe()' 
    @test_throws ArgumentError TestCompressorRecipe()'*y
    @test_throws ArgumentError y*TestCompressorRecipe()' 
end

#################################
# Additional Testing Parameters 
#################################
mul!(C::AbstractArray, S::TestCompressorRecipe, A::AbstractArray, alpha::Number, 
    β::Number) = fill!(C, -1)
mul!(C::AbstractArray, A::AbstractArray, S::TestCompressorRecipe, alpha::Number, 
    β::Number) = fill!(C, -2)

#TODO: Correct Multiplication Checks 

end