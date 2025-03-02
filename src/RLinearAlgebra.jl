module RLinearAlgebra
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
    export complete_compressor, complete_solver, complete_approximator, update_approximator!
    export complete_sub_solver, complete_approximator_error, complete_approximator_error!
    export complete_solver_error, complete_logger

    # Export the update_ functions
    export update_compressor!, update_logger!, update_sub_solver!

    # Export the compute error functions
    export compute_solver_error, compute_approximator_error, compute_approximator_error!

    # Export the Solve and Approximate functions
    export rsolve, rsolve!, rapproximate, rapproximate!

    # Export Approximator types
    export Approximator, ApproximatorRecipe, ApproximatorAdjoint
    export ApproximatorError, ApproximatorErrorRecipe

    # Export Compressor types
    export Compressor, CompressorRecipe, CompressorAdjoint

    # Export Solver types
    export Solver, SolverRecipe 

    # Export Logger types
    export  Logger, LoggerRecipe

    # Export SubSolver types
    export SubSolver, SubSolverRecipe

    # Export SolverError types
    export SolverError, SolverErrorRecipe
end #module
