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
    return ColCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

function RLinearAlgebra.update_compressor!(
    comp::ColCompressorRecipe,
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
    S::Main.col_projectionTest.ColCompressorRecipe, 
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
# mutable struct KTestErrorNoRes <: RLinearAlgebra.SolverError end

# mutable struct KTestErrorRecipeNoRes <: RLinearAlgebra.SolverErrorRecipe end

# function RLinearAlgebra.complete_error(
#     error::KTestErrorNoRes, 
#     solver::Kaczmarz,
#     A::AbstractMatrix, 
#     b::AbstractVector 
# )
#     return KTestErrorRecipeNoRes()
# end

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
            :compressor, 
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

end