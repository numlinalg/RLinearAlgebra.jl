module RLinearAlgebra
import Base.:*
import Base: transpose, adjoint
import LinearAlgebra: Adjoint, axpby!, dot, ldiv!, lmul!, lq!, lq, LQ, mul!, norm, qr!
import StatsBase: sample, sample!, ProbabilityWeights, wsample!
import Random: bitrand, rand!, randn!
import SparseArrays: SparseMatrixCSC, sprandn, sparse

# Include the files correspoding to the top-level techniques
include("Compressors.jl")
include("Solvers.jl")
include("Approximators.jl")

# Export Approximator types and functions
export Approximator, ApproximatorRecipe, ApproximatorAdjoint
export rapproximate, rapproximate!, complete_approximator
export RangeApproximator, RangeApproximatorRecipe
export RangeFinder, RangeFinderRecipe

# Export Compressor types and functions
export Compressor, CompressorRecipe, CompressorAdjoint
export Cardinality, Left, Right, Undef
export complete_compressor, update_compressor!
export FJLT, FJLTRecipe
export Gaussian, GaussianRecipe
export SparseSign, SparseSignRecipe
export SRHT, SRHTRecipe
export CountSketch, CountSketchRecipe


# Export Distribution types and functions
export Distribution, DistributionRecipe
export complete_distribution, update_distribution!, sample_distribution!
export Uniform, UniformRecipe

# Export Solver types and functions
export Solver, SolverRecipe
export complete_solver, update_solver!, rsolve!

# Export Logger types and functions
export Logger, LoggerRecipe
export BasicLogger, BasicLoggerRecipe
export complete_logger, update_logger!, reset_logger!
export threshold_stop

# Export SubSolver types and functions
export SubSolver, SubSolverRecipe, ldiv!
export complete_sub_solver, update_sub_solver!
export LQSolver, LQSolverRecipe
export QRSolver, QRSolverRecipe

# Export SolverError types and functions
export SolverError, SolverErrorRecipe
export complete_error, compute_error
export FullResidual, FullResidualRecipe

# Export ApproximatorError types and functions
export ApproximatorError, ApproximatorErrorRecipe
export complete_approximator_error, compute_approximator_error, compute_approximator_error!

end #module
