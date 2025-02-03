module sub_solver_abstract_types 
    using Test, RLinearAlgebra
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")

    @test isdefined(Main, :SubSolver)
    @test isdefined(Main, :SubSolverRecipe)
end
