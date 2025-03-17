module logger_abstract_types
using Test, RLinearAlgebra
include("../../test_helpers/field_test_macros.jl")
include("../../test_helpers/approx_tol.jl")
struct TestLogger <: Logger end
struct TestLoggerRecipe <: LoggerRecipe end

@testset "Logger Abstract Types" begin
    @test isdefined(Main, :Logger)
    @test isdefined(Main, :LoggerRecipe)
end

@testset "Logger Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)

    @test_throws ArgumentError complete_logger(TestLogger(), A)
    @test_throws ArgumentError complete_logger(TestLogger(), A, b)
    @test_throws ArgumentError update_logger!(TestLoggerRecipe(), 1.0, 1)
end
end
