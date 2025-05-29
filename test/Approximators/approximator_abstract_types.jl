module approximators_abstract_types
using Test, RLinearAlgebra
include("../test_helpers/field_test_macros.jl")
include("../test_helpers/approx_tol.jl")
struct TestCompressorRecipe <: CompressorRecipe end
struct TestApproximator <: Approximator end
struct TestApproximatorRecipe <: ApproximatorRecipe
    S::CompressorRecipe
end
TestApproximatorRecipe() = TestApproximatorRecipe(TestCompressorRecipe())
struct TestApproximatorError <: ApproximatorError end
struct TestApproximatorErrorRecipe <: ApproximatorErrorRecipe end

@testset "Approximator Abstract Types" begin
    @test isdefined(Main, :Approximator)
    @test isdefined(Main, :ApproximatorRecipe)
    @test isdefined(Main, :ApproximatorError)
    @test isdefined(Main, :ApproximatorErrorRecipe)
    @test isdefined(Main, :ApproximatorAdjoint)
end

@testset "Approximator Argument Errors" begin
    A = rand(2, 2)

    @test_throws ArgumentError rapproximate!(TestApproximatorRecipe(), A)
    @test_throws ArgumentError rapproximate(TestApproximator(), A)
end

# Test ApproximatorError argment error
@testset "ApproximatorError Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_approximator_error(
        TestApproximatorError(), TestApproximator(), A
    )
    @test_throws ArgumentError compute_approximator_error!(
        TestApproximatorErrorRecipe(), TestApproximatorRecipe(), A
    )
    @test_throws MethodError compute_approximator_error(
        TestApproximatorError(), TestApproximatorRecipe(), A
    )
end

end
