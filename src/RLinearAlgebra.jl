    import Base.:* 
    import Base: transpose, adjoint 
    import LinearAlgebra: ldiv!, mul!, lmul!, dot, lq, LQ
    import StatsBase: sample!
    import Random: bitrand, seed!
   
    # Include the files correspoding to the top-level techniques
    include("Compressors.jl")
    include("Solvers.jl")
    include("Approximators.jl")

    # Export the complete_ functions 
    export complete_compressor, complete_solver, complete_approximator
    export complete_sub_solver, complete_error, complete_logger

    # Export the update_ functions
    export update_compressor!, update_logger!, update_sub_solver!

    # Export the Solve and Approximate functions
    export rsolve, rsolve!, rapproximate, rapproximate!

    # Export Approximator types
    export Approximator, ApproximatorRecipe, ApproximatorAdjoint
    export RangeFinder, RangeFinderRecipe
    export ErrorMethod, ProjectedError, ProjectedErrorRecipe

    # Export Compressor types
    export Compressor, CompressorRecipe, CompressorAdjoint
    export SparseSign, SparseSignRecipe 

    # Export Solver types
    export Solver, SolverRecipe, Kaczmarz, KaczmarzRecipe 

    # Export Logger types
    export  Logger, LoggerRecipe, BasicLogger, BasicLoggerRecipe

    # Export SubSolver types
    export SubSolver, SubSolverRecipe, LQSolver, LQSolverRecipe

    # Export SolverError types
    export SolverError, SolverErrorRecipe, FullResidual, FullResidualRecipe
    export CompressedResidual, CompressedResidualRecipe
