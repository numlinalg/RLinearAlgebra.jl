@testset "SubSolver_LQ" begin
    using Test, RLinearAlgebra, Random, LinearAlgebra
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol
    let
	    Random.seed!(21321)
        n_rows = 3
        n_cols = 10
	    S1 = LQSolver()
	    @test typeof(S1) <: SubSolver
	    A = rand(n_rows, n_cols)
        # Because A gets overwritten copy it for the test
        C = deepcopy(A)
	    b = rand(n_rows)
	    S_method = complete_sub_solver(S1, A, b)
	    @test typeof(S_method) <: SubSolverRecipe
	    # Check that default values are correct
	    # and appropiate allocations have been made
	    @test S_method.A == A
        # Test the solver produces the correct solution
	    x = rand(n_cols)
        ldiv!(x, S_method, b)
        @test x ≈ lq(C) \ b
        # Now test update solver
        B = rand(n_rows, n_cols)
        update_sub_solver!(S_method, B)
        @test S_method.A ≈ B
    end

    let
	    Random.seed!(21321)
        n_rows = 1
        n_cols = 10
	    S1 = LQSolver()
	    @test typeof(S1) <: SubSolver
	    A = rand(n_cols)
        # Because A gets overwritten copy it for the test
        C = deepcopy(A)
	    b = rand()
	    S_method = complete_sub_solver(S1, A, b)
	    @test typeof(S_method) <: SubSolverRecipe
	    # Check that default values are correct
	    # and appropiate allocations have been made
	    @test S_method.A == A
        # Test the solver produces the correct solution
	    x = rand(n_cols)
        d = deepcopy(x)
        ldiv!(x, S_method, b)
        @test x ≈ (b - dot(A,d)) / dot(A,A) * A
        # Now test update solver
        B = rand(n_cols)
        update_sub_solver!(S_method, B)
        @test S_method.A ≈ B
    end

end
