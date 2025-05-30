module solver_abstract_types
using Test, RLinearAlgebra

###################################
# Initial Testing Parameters   
###################################
struct TestSolver <: Solver end
struct TestSolverRecipe <: SolverRecipe end

@testset "Solver Abstract Types" begin
    @test isdefined(Main, :Solver)
    @test isdefined(Main, :SolverRecipe)
end

@testset "Solver Interfaces Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_solver(TestSolver(), x, A, b)
    @test_throws ArgumentError rsolve!(TestSolverRecipe(), x, A, b)
end

###################################
# Updated Testing Parameters   
###################################


end
