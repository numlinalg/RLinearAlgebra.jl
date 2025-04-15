module compressor_abstract_types
using Test, RLinearAlgebra
import LinearAlgebra: mul!
using ..FieldTest
using ..ApproxTol
struct TestCompressor <: Compressor end
struct TestCompressorRecipe <: CompressorRecipe end

@testset "Compressor Abstract Types" begin
    @test isdefined(Main, :Compressor)
    @test isdefined(Main, :CompressorRecipe)
    @test isdefined(Main, :CompressorAdjoint)
    @test isdefined(Main, :Cardinality)
    @test isdefined(Main, :Left)
    @test isdefined(Main, :Right)
end

@testset "Compressor Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_compressor(TestCompressor(), A)
    @test_throws ArgumentError complete_compressor(TestCompressor(), A, b)
    @test_throws ArgumentError complete_compressor(TestCompressor(), x, A, b)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe())
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(), A)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(), A, b)
    @test_throws ArgumentError update_compressor!(TestCompressorRecipe(), x, A, b)
end

@testset "Compressor Recipe Multiplication Errors" begin 
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
    #   No specific 3 arg mul defined for next case but it is correctly handled
    @test_throws ArgumentError mul!(x, y, TestCompressorRecipe()) 

    # Five argument muls for adjoint 
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A, 1.0, 1.0)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()', 1.0, 1.0)
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x, 1.0, 1.0)

    # Three arguments muls for adjoint 
    @test_throws ArgumentError mul!(y, TestCompressorRecipe()', x)
    @test_throws ArgumentError mul!(C, TestCompressorRecipe()', A)
    @test_throws ArgumentError mul!(C, A, TestCompressorRecipe()')

    
end

end
