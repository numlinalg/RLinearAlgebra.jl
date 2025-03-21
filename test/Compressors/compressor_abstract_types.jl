module compressor_abstract_types
using Test, RLinearAlgebra
include("../test_helpers/field_test_macros.jl")
include("../test_helpers/approx_tol.jl")
struct TestCompressor <: Compressor end
struct TestCompressorRecipe <: CompressorRecipe end

@testset "Compressor Abstract Tyoes" begin
    @test isdefined(Main, :Compressor)
    @test isdefined(Main, :CompressorRecipe)
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

end
