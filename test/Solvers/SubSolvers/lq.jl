@testset "LQ SubSolver Tests" begin
    using Test, RLinearAlgebra, Random, LinearAlgebra
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol
    
    A = rand(3, 10)
    b = rand(3)
    @testset "LQ SubSolver" begin
        @test supertype(LQSolver) == SubSolver
        @test fieldnames(LQSolver) == ()
        @test fieldtypes(LQSolver) == ()
        # Checks that the compressor produces a SubSolver
        ss = LQSolver()
        @test typeof(ss) <: SubSolver
    end

    @testset "LQ SubSolver Recipe" begin
        @test supertype(LQSolverRecipe) == SubSolverRecipe
        @test fieldnames(LQSolverRecipe) == (:A, )
        @test fieldtypes(LQSolverRecipe) == (AbstractArray,)
    end
    
    @testset "LQ SubSolver: Complete SubSolver" begin
        let
            ss = LQSolver()
            ss_recipe = complete_sub_solver(ss, A)
            # test the attributes of the outputs of the complete function
            @test typeof(ss_recipe) == LQSolverRecipe{Matrix{Float64}}
            @test ss_recipe.A == A
        end

    end

    @testset "LQ SubSolver: Update SubSolver" begin
        let 
            B = rand(10, 3)
            ss = LQSolver()
            ss_recipe = complete_sub_solver(ss, A)
            update_sub_solver!(ss_recipe, B)
            @test ss_recipe.A == B
        end

    end

    @testset "LQ SubSolver: ldiv!" begin
        # begin by testing the matrix case
        let  
            ss = LQSolver()
            ss_recipe = complete_sub_solver(ss, A)
            x_sol = rand(10)
            x_true = lq(A) \ b
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

        # test the solver when using vectors
        let  
            a = rand(10)
            b = rand()
            ss = LQSolver()
            ss_recipe = complete_sub_solver(ss, a)
            x_sol = rand(10)
            xc = deepcopy(x_sol)
            x_true = (b - dot(a, xc)) / dot(a, a) * a 
            ldiv!(x_sol, ss_recipe, b)
            @test x_sol ≈ x_true
        end

    end

end
