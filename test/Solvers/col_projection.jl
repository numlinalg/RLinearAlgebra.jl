module col_projectionTest
using Test, RLinearAlgebra, LinearAlgebra
import RLinearAlgebra: complete_compressor, update_compressor!
import LinearAlgebra: mul!, norm
import Random: randn!
using ..FieldTest
using ..ApproxTol

# Define the test structures
##########################
# Compressors
##########################
mutable struct ColTestCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

ColTestCompressor() = ColTestCompressor(Right(), 5)

mutable struct ColTestCompressorRecipe <: CompressorRecipe 
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

function RLinearAlgebra.complete_compressor(
    comp::ColTestCompressor, 
    A::AbstractMatrix,
    b::AbstractVector 
)
    n_rows = size(A, 2)
    n_cols = comp.compression_dim
    # Make a gaussian compressor
    op = randn(n_rows, n_cols) ./ sqrt(n_rows)
    return ColTestCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

function RLinearAlgebra.update_compressor!(
    comp::ColTestCompressorRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector
)
    randn!(comp.op)
    comp.op ./= sqrt(comp.n_rows)
end

# Define a mul function for the test compressor
function RLinearAlgebra.mul!(
    C::AbstractArray,
    A::AbstractArray,
    S::Main.col_projectionTest.ColTestCompressorRecipe, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, S.op, alpha, beta)
end

##########################
# Error Method
##########################
mutable struct ColTestError <: RLinearAlgebra.SolverError
    g::Real
end

mutable struct ColTestErrorRecipe <: RLinearAlgebra.SolverErrorRecipe
    residual::Vector{Number}
end

ColTestError() = ColTestError(1.0) 

function RLinearAlgebra.complete_error(
    error::ColTestError, 
    solver::col_projection,
    A::AbstractMatrix, 
    b::AbstractVector
)
    return ColTestErrorRecipe(zeros(typeof(error.g), size(A, 1)))
end

function RLinearAlgebra.compute_error(error::ColTestErrorRecipe, solver, A, b)
    error.residual = A * solver.solution_vec - b
    return norm(error.residual)
end

##############################
# Residual-less Error Recipe
##############################
mutable struct ColTestErrorNoRes <: RLinearAlgebra.SolverError end

mutable struct ColTestErrorRecipeNoRes <: RLinearAlgebra.SolverErrorRecipe end

function RLinearAlgebra.complete_error(
    error::ColTestErrorNoRes, 
    solver::col_projection,
    A::AbstractMatrix, 
    b::AbstractVector 
)
    return ColTestErrorRecipeNoRes()
end

############################
# Loggers
############################
mutable struct ColTestLog <: Logger
    max_it::Int64
    g::Number
end

ColTestLog() = ColTestLog(5, 1.0) 

ColTestLog(max_it) = ColTestLog(max_it, 1.0)

mutable struct ColTestLogRecipe <: LoggerRecipe
    max_it::Int64
    hist::Vector{Real}
    thresh::Float64
    converged::Bool
end

function RLinearAlgebra.complete_logger(logger::ColTestLog)
    return ColTestLogRecipe(
        logger.max_it, 
        zeros(typeof(logger.g), logger.max_it), 
        logger.g, 
        false
    )
end

function RLinearAlgebra.update_logger!(logger::ColTestLogRecipe, err::Real, i::Int64)
    logger.hist[i] = err
    logger.converged = err < logger.thresh ? true : false
end

function RLinearAlgebra.reset_logger!(logger::ColTestLogRecipe)
    fill!(logger.hist, 0.0)
end

##############################
# Converged-less Logger
##############################
mutable struct ColTestLogNoCov <: Logger end

mutable struct ColTestLogRecipeNoCov <: LoggerRecipe end

function RLinearAlgebra.complete_logger(logger::ColTestLogNoCov)
    return ColTestLogRecipeNoCov()
end

##############################
# SubSolver
##############################
mutable struct ColTestSubSolver <: SubSolver end

mutable struct ColTestSubSolverRecipe <: SubSolverRecipe 
    A::AbstractMatrix
end

function RLinearAlgebra.complete_sub_solver(
    solver::ColTestSubSolver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    return ColTestSubSolverRecipe(A)
end

function RLinearAlgebra.update_sub_solver!(solver::ColTestSubSolverRecipe, A::AbstractMatrix)
    solver.A = A
end

function RLinearAlgebra.ldiv!(
    x::AbstractVector, 
    S::Main.col_projectionTest.ColTestSubSolverRecipe, 
    b::AbstractVector, 
)
    ldiv!(x, factorize(S.A), b)
end

#####################################
# Testing the functions
#####################################
@testset "col_projection" begin
    n_rows = 3
    n_cols = 10
    A = rand(n_rows, n_cols)
    xsol = rand(n_cols)
    b = A * xsol

    @testset "col_projection Technique" begin
        @test supertype(col_projection) == Solver

        # test fieldnames and types
        @test fieldnames(col_projection) == (:alpha, :compressor, :log, :error, :sub_solver)
        @test fieldtypes(col_projection) == (Float64, Compressor, Logger, SolverError, SubSolver)

        # test default constructor

        let solver = col_projection()
            @test solver.alpha == 1.0
            @test typeof(solver.compressor) == SparseSign 
            @test typeof(solver.compressor.cardinality) == Right 
            @test typeof(solver.log) == BasicLogger
            @test typeof(solver.error) == FullResidual
            @test typeof(solver.sub_solver) == QRSolver 
        end

        # test constructor
        let solver = col_projection(
            alpha = 2.0,
            compressor = ColTestCompressor(),
            log = ColTestLog(),
            error = ColTestError(),
            sub_solver = ColTestSubSolver()
        )

            @test solver.alpha == 2.0
            @test typeof(solver.compressor) == ColTestCompressor
            @test typeof(solver.log) == ColTestLog
            @test typeof(solver.error) == ColTestError
            @test typeof(solver.sub_solver) == ColTestSubSolver
        end 

        # Test that error gets returned with left compressor
        @test_logs (:warn, 
               "Compressor has cardinality `Left` but col_projection\
               compresses from the `Right`." 
        ) col_projection(
            alpha = 2.0,
            compressor = ColTestCompressor(Left(), 5),
            log = ColTestLog(),
            error = ColTestError(),
            sub_solver = ColTestSubSolver()
        )
    end

    @testset "col_projectionRecipe" begin
        @test supertype(col_projectionRecipe) == SolverRecipe

        # test fieldnames and types
        @test fieldnames(col_projectionRecipe) == (
            :S, 
            :log, 
            :error, 
            :sub_solver, 
            :alpha, 
            :compressed_mat, 
            :solution_vec, 
            :update_vec, 
            :mat_view, 
            :residual_vec
        )
        @test fieldtypes(col_projectionRecipe) == (
            CompressorRecipe,
            LoggerRecipe,
            SolverErrorRecipe,
            SubSolverRecipe,
            Float64, 
            AbstractArray, 
            AbstractVector, 
            AbstractVector, 
            SubArray, 
            AbstractVector,
        )  
    end

    @testset "col_projection: Complete Solver" begin
        # test error method with no residual error 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)

            comp = ColTestCompressor(Right(), comp_dim)
            log = ColTestLog()
            err = ColTestErrorNoRes()
            sub_solver = ColTestSubSolver()
            solver = col_projection(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )

            @test_throws ArgumentError(
                "ErrorRecipe $(typeof(ColTestErrorRecipeNoRes())) does not contain the \
                field 'residual' and is not valid for a col_projection solver."
            ) complete_solver(solver, x, A, b)
        end

        # test logger method with no converged field 
        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)

            comp = ColTestCompressor(Right(), comp_dim)
            log = ColTestLogNoCov()
            err = ColTestError()
            sub_solver = ColTestSubSolver()
            solver = col_projection(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )

            @test_throws ArgumentError(
                "LoggerRecipe $(typeof(ColTestLogRecipeNoCov())) does not contain \
                the field 'converged' and is not valid for a col_projection solver."
            ) complete_solver(solver, x, A, b)
        end

        let A = A,
            xsol = xsol,
            b = b,
            comp_dim = 2,
            alpha = 1.0,
            n_rows = size(A, 1),
            n_cols = size(A, 2),
            x = zeros(n_cols)

            comp = ColTestCompressor(Right(), comp_dim)
            log = ColTestLog()
            err = ColTestError()
            sub_solver = ColTestSubSolver()
            solver = col_projection(
                compressor = comp,
                log = log,
                error = err,
                sub_solver = sub_solver,
                alpha = alpha
            )

            solver_rec = complete_solver(solver, x, A, b)

            # test types of the contents of the solver
            @test typeof(solver_rec) == col_projectionRecipe{
                Float64, 
                Vector{Float64}, 
                Matrix{Float64}, 
                SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}, 
                Main.col_projectionTest.ColTestCompressorRecipe, 
                Main.col_projectionTest.ColTestLogRecipe, 
                Main.col_projectionTest.ColTestErrorRecipe, 
                Main.col_projectionTest.ColTestSubSolverRecipe
            }
            @test typeof(solver_rec.S) == ColTestCompressorRecipe
            @test typeof(solver_rec.log) == ColTestLogRecipe
            @test typeof(solver_rec.error) == ColTestErrorRecipe
            @test typeof(solver_rec.sub_solver) == ColTestSubSolverRecipe
            @test typeof(solver_rec.alpha) == Float64
            @test typeof(solver_rec.compressed_mat) == Matrix{Float64}
            @test typeof(solver_rec.solution_vec) == Vector{Float64}
            @test typeof(solver_rec.update_vec) == Vector{Float64}
            @test typeof(solver_rec.mat_view) <: SubArray
            @test typeof(solver_rec.residual_vec) == Vector{Float64}

            # Test sizes of vectors and matrices
            @test size(solver_rec.S) == (n_cols, comp_dim)
            @test size(solver_rec.compressed_mat) == (n_rows, comp_dim)
            @test size(solver_rec.update_vec) == (comp_dim,)

            # test values of entries
            solver_rec.alpha == alpha
            solver_rec.solution_vec == x
            solver_rec.update_vec == zeros(n_cols)
        end
        
    end
    # @testset "col_projection: Column Projection Update" begin
    #     # Begin with a test of an update when the block size is 1
    #     for type in [Float32, Float64, ComplexF32, ComplexF64]
    #         let A = rand(type, n_rows, n_cols),
    #             xsol = ones(type, n_cols),
    #             b = A * xsol,
    #             comp_dim = 1,
    #             alpha = 1.0,
    #             n_rows = size(A, 1),
    #             n_cols = size(A, 2),
    #             x = zeros(type, n_cols)

    #             comp = ColTestCompressor(Right(), comp_dim)
    #             log = ColTestLog()
    #             err = ColTestError()
    #             sub_solver = ColTestSubSolver()
    #             solver = col_projection(
    #                 compressor = comp,
    #                 log = log,
    #                 error = err,
    #                 sub_solver = sub_solver,
    #                 alpha = alpha
    #             )

    #             solver_rec = complete_solver(solver, x, A, b)

    #             # Sketch the matrix and vector
    #             As = A * solver_rec.compressor
    #             solver_rec.mat_view = view(As, :, 1:comp_dim)
    #             solver_rec.solution_vec = deepcopy(x) 
    #             solver_rec.residual_vec = b - As
    #             solver.solution_vec = x

    #             # compute comparison update
    #             sc = dot(solver.mat_view, residual_vec) / dot(As, As) * alpha
    #             test_sol = x - solver.S * sc

    #             # compute the update
    #             RLinearAlgebra.col_projection_update!(solver_rec)
    #             @test solver_rec.solution_vec ≈ test_sol
    #         end

    #     end

    # end

    # @testset "col_projection: Block Column Projection Update" begin
    #     # Begin with a test of an update when the block size is 2
    #     for type in [Float32, Float64, ComplexF32, ComplexF64]
    #         let A = rand(type, n_rows, n_cols),
    #             xsol = ones(type, n_cols),
    #             b = A * xsol,
    #             comp_dim = 1,
    #             alpha = 1.0,
    #             n_rows = size(A, 1),
    #             n_cols = size(A, 2),
    #             x = zeros(type, n_cols)

    #             comp = ColTestCompressor(Left(), comp_dim)
    #             log = ColTestLog()
    #             err = ColTestError()
    #             sub_solver = ColTestSubSolver()
    #             solver = col_projection(
    #                 compressor = comp,
    #                 log = log,
    #                 error = err,
    #                 sub_solver = sub_solver,
    #                 alpha = alpha
    #             )

    #             solver_rec = complete_solver(solver, x, A, b)

    #             # Sketch the matrix and vector
    #             sA = solver_rec.compressor * A 
    #             solver_rec.vec_view = view(sb, 1:comp_dim)
    #             solver_rec.mat_view = view(sA, 1:comp_dim, :)
    #             solver_rec.solution_vec = deepcopy(x) 

    #             # compute comparison update
    #             test_sol =  x + As \ (sb - sA * x)

    #             # compute the update
    #             RLinearAlgebra.col_update_block!(solver_rec)
    #             @test solver_rec.solution_vec ≈ test_sol
    #         end

    #    end

    #end

end

end