module gradient_error 
using Test, RLinearAlgebra, Random
import LinearAlgebra: mul!, norm
using ..FieldTest
using ..ApproxTol
Random.seed!(1232)

mutable struct TestSolver <: Solver end

mutable struct TestSolverRecipe <: SolverRecipe
    residual_vec::AbstractVector
end

@testset "LS Gradient" begin
    @testset "LS Gradient: SolverError" begin
        # Verify Supertype
        @test supertype(LSgradient) == SolverError 

        # Verify fieldnames and types
        @test fieldnames(LSgradient) == ()
        @test fieldtypes(LSgradient) == ()
        # Verify the internal constructor

    end

    @testset "LS Gradient: SolverErrorRecipe" begin
        # Verify Supertype
        @test supertype(LSgradientRecipe) ==  SolverErrorRecipe

        # Verify fieldnames and types
        @test fieldnames(LSgradientRecipe) == (:gradient,)
        @test fieldtypes(LSgradientRecipe) == (AbstractVector,)
    end

    @testset "Residual: Complete error" begin
        for type in [Float32, Float64, ComplexF32, ComplexF64] 
            let n_rows = 4,
                n_cols = 6,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows),
                x = rand(type, n_cols),
                r = A*x - b,
                solver_rec = TestSolverRecipe(r),
                error_rec = complete_error(LSgradient(), TestSolver(), A, b)

                # Test the type
                @test typeof(error_rec) == LSgradientRecipe{typeof(b)}
                # Test type of residual vector
                @test eltype(error_rec.gradient) == type
                # Test residual vector to be all zeros
                @test error_rec.gradient == zeros(type, n_cols)
            end

        end

    end

    @testset "Residual: Compute Error" begin
        for type in [Float32, Float64] 
            let n_rows = 4,
                n_cols = 6,
                A = rand(type, n_rows, n_cols),
                b = rand(type, n_rows),
                x = rand(type, n_cols),
                r = A*x - b,
                solver_rec = TestSolverRecipe(r),
                solver = TestSolver(),
                error_rec = complete_error(LSgradient(), TestSolver(), A, b)

                # compute the error value
                err_val = compute_error(error_rec, solver_rec, A, b)
                # compute the gradient
                res = A' * r
                # compute norm squared of residual
                @test norm(res) ≈ err_val
            end

        end

    end

end

end
