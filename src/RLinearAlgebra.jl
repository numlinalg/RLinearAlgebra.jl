module RLinearAlgebra
import Base.:*
import Base: transpose, adjoint
import LinearAlgebra: ldiv!, mul!, lmul!, dot, lq!, LQ, Adjoint
import StatsBase: sample!
import Random: bitrand, rand!, randn!
import SparseArrays: SparseMatrixCSC

# Include the files correspoding to the top-level techniques
include("Compressors.jl")
include("Solvers.jl")
include("Approximators.jl")

# Export Approximator types and functions
export Approximator, ApproximatorRecipe, ApproximatorAdjoint
export complete_approximator, update_approximator!, rapproximate, rapproximate!

# Export Compressor types and functions
export Compressor, CompressorRecipe, CompressorAdjoint
export Cardinality, Left, Right
export complete_compressor, update_compressor!
export Gaussian, GaussianRecipe
export SparseSign, SparseSignRecipe

# Export Solver types and functions
export Solver, SolverRecipe
export complete_solver, update_solver!, rsolve, rsolve!

# Export Logger types and functions
export Logger, LoggerRecipe
export MAInfo
export BasicLogger, BasicLoggerRecipe, MALogger, MALoggerRecipe
export complete_logger, update_logger!, reset_logger!, update_ma!
export threshold_stop

# Export SubSolver types
export SubSolver, SubSolverRecipe, ldiv!
export complete_sub_solver, update_sub_solver!
export LQSolver, LQSolverRecipe

# Export SolverError types and functions
export SolverError, SolverErrorRecipe
export complete_solver_error, compute_solver_error, compute_solver_error!

# Export ApproximatorError types and functions
export ApproximatorError, ApproximatorErrorRecipe
export complete_approximator_error, compute_approximator_error, compute_approximator_error!

end #module
