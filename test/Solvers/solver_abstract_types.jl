module solver_abstract_types
using Test, RLinearAlgebra
using ..FieldTest
using ..ApproxTol
import Random:seed!
seed!(1232)

struct TestSolver <: Solver end
struct TestSolverRecipe <: SolverRecipe end
struct TestSolverError <: SolverError end
struct TestSolverErrorRecipe <: SolverErrorRecipe end
@testset "Solver Abstract Types" begin
    @test isdefined(Main, :Solver)
    @test isdefined(Main, :SolverRecipe)
end

# Test Solver argment error
@testset "Solver Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_solver(TestSolver(), x, A, b)
    @test_throws ArgumentError rsolve!(TestSolverRecipe(), x, A, b)
end

end
