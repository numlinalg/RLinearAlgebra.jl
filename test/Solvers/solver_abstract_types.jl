module solver_abstract_types
using Test, RLinearAlgebra
include("../test_helpers/field_test_macros.jl")
include("../test_helpers/approx_tol.jl")

struct TestSolver <: Solver end
struct TestSolverRecipe <: SolverRecipe end
struct TestSolverError <: SolverError end
struct TestSolverErrorRecipe <: SolverErrorRecipe end
@testset "Solver Abstract Types" begin
    @test isdefined(Main, :Solver)
    @test isdefined(Main, :SolverRecipe)
    @test isdefined(Main, :SolverError)
    @test isdefined(Main, :SolverErrorRecipe)
end
# Test Solver argment error
@testset "Solver Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_solver(TestSolver(), x, A, b)
    @test_throws ArgumentError rsolve!(TestSolverRecipe(), x, A, b)
    @test_throws ArgumentError rsolve(TestSolver(), x, A, b)
end

# Test SolverError argment error
@testset "SolverError Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_solver_error(TestSolverError(), TestSolver(), A, b)
    @test_throws ArgumentError compute_solver_error(
        TestSolverErrorRecipe(), TestSolverRecipe(), x, A, b
    )
end
end
