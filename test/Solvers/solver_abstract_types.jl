module solver_abstract_types 
    using Test, RLinearAlgebra
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")

    @testset "Solver Abstract Types" begin
       @test isdefined(Main, :Solver)
       @test isdefined(Main, :SolverRecipe)
       @test isdefined(Main, :SolverError)
       @test isdefined(Main, :SolverErrorRecipe)
    end

end
