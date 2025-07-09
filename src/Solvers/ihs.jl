"""
    IHS <: Solver

An implementation of the Iterative Hessian Sketch solver for solving over determined 
least squares problems (@cite)[pilanci2014iterative].
 
# Mathematical Description
Let ``A`` be an ``m \\times n`` matrix and consider the least square problem ``\\min_x 
\\|Ax - b \\|_2^2. Iterative Hessian Sketch 

# Fields
- `alpha::Float64`, a step size parameter.
- `compressor::Compressor`, a technique for forming the compressed linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError', a method for estimating the progress of the solver.

# Constructor
    function IHS(;
        compressor::Compressor = SparseSign(cardinality = Left()),
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        alpha::Float64 = 1.0
    )
## Keywords
- `compressor::Compressor`, a technique for forming the compressed linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError', a method for estimating the progress of the solver.
- `alpha::Float64`, a step size parameter.

# Returns
- A `IHS` object.
"""
mutable struct IHS <: Solver
    alpha::Float64
    log::Logger
    compressor::Compressor
    error::SolverError
    function IHS(alpha, log, compressor, error)
        if typeof(compressor.cardinality) != Left
            @warn "Compressor has cardinality `Right` but IHS compresses  from the  `Left`."
        end 

        new(alpha, log, compressor, error)
    end

end

function IHS(;
    compressor::Compressor = SparseSign(cardinality = Left()),
    log::Logger = BasicLogger(),
    error::SolverError = FullResidual(),
    alpha::Float64 = 1.0
)
    return IHS(
        alpha, 
        log,
        compressor,
        error
    )
end

"""
    IHSRecipe{
        Type<:Number, 
        LR<:LoggerRecipe,
        CR<:CompressorRecipe,
        ER<:ErrorRecipe,
        M<:AbstractArray, 
        MV<:SubArray, 
        V<:AbstractVector
    } <: SolverRecip

A mutable structure containing all information relevant to the Iterative Hessian Sketch 
solver. It is formed by calling the function `complete_solver` on a `IHS` solver, which 
includes all the user controlled parameters, the linear system matrix `A` and constant 
vector `b`.

# Fields
- `compressor::CompressorRecipe`, a technique for compressing the matrix ``A``.
- `logger::LoggerRecipe`, a technique for logging the progress of the solver.
- `error::SolverErrorRecipe`, a technique for estimating the progress of the solver.
- `compresed_mat::AbstractMatrix`, a buffer for storing the compressed matrix.
- `mat_view::SubArray`, a container for storing a view of the compressed matrix buffer.
- `residual_vec::AbstractVector`, a vector that contains the residual of the linear system 
    ``Ax-b``.
- `gradient_vec::AbstractVector`, a vector that contains the gradient of the least squares 
    problem, ``A^\\top(b-Ax)``.
- `update_vec::AbstractVector`, a vector that contains the update by solving the IHS 
    subproblem.
- `solution_vec::AbstractVector`, a vector storing the current IHS solution.
- `R::UpperTriangular`, a containter for storing the upper triangular portion of the R 
    factor from a QR factorization of `mat_view`. This is used to solve the IHS sub-problem.
"""
mutable struct IHSRecipe{
    Type<:Number, 
    LR<:LoggerRecipe,
    CR<:CompressorRecipe,
    ER<:SolverErrorRecipe,
    M<:AbstractArray, 
    MV<:SubArray, 
    V<:AbstractVector
} <: SolverRecipe
    log::LR
    compressor::CR
    error::ER
    alpha::Float64
    compressed_mat::M
    mat_view::MV
    residual_vec::V
    gradient_vec::V
    update_vec::V
    solution_vec::V
    R::UpperTriangular{Type, M}
end

function complete_solver(
    ingredients::IHS, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    compressor = complete_compressor(ingredients.compressor, x, A, b)
    logger = complete_logger(ingredients.log) 
    error = complete_error(ingredients.error, ingredients, A, b)
    sample_size::Int64 = compressor.n_rows
    rows_a, cols_a = size(A)
    # Check that required fields are in the types
    if !isdefined(error, :residual)
        throw(
            ArgumentError(
                "ErrorRecipe $(typeof(error)) does not contain the \
                field 'residual' and is not valid for an IHS solver."
            )
        )
    end

    if !isdefined(logger, :converged)
        throw(
            ArgumentError(
                "LoggerRecipe $(typeof(logger)) does not contain \
                the field 'converged' and is not valid for an IHS solver."
            )
        )
    end

    # Check that the sketch size is larger than the column dimension and return a warning
    # otherwise
    if cols_a > sample_size
        throw(
            ArgumentError(
                "Compression dimension not larger than column dimension this will lead to \
                singular QR decompositions, which cannot be inverted"
            )
        )
    end

    compressed_mat = zeros(eltype(A), sample_size, cols_a)
    res = zeros(eltype(A), rows_a) 
    grad = zeros(eltype(A), cols_a) 
    update_vec = zeros(eltype(A), cols_a) 
    solution_vec = x
    mat_view = view(compressed_mat, 1:sample_size, :)
    R = UpperTriangular(mat_view[1:cols_a, :])

    return IHSRecipe{
        eltype(compressed_mat),
        typeof(logger),
        typeof(compressor),
        typeof(error),
        typeof(compressed_mat),
        typeof(mat_view),
        typeof(update_vec)
    }(
        logger, 
        compressor, 
        error, 
        ingredients.alpha,
        compressed_mat, 
        mat_view, 
        res, 
        grad, 
        update_vec, 
        solution_vec, 
        R
    )
end

function rsolve!(solver::IHSRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    reset_logger!(solver.log)
    solver.solution_vec = x
    err = 0.0
    copyto!(solver.residual_vec, b)
    for i in 1:solver.log.max_it
        mul!(solver.residual_vec, A, solver.solution_vec, -1.0, 1.0)
        mul!(solver.gradient_vec, A', solver.residual_vec)
        err = compute_error(solver.error, solver, A, b)
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return solver.solution_vec, solver.log
        end

        # generate a new compressor
        update_compressor!(solver.compressor, x, A, b)
        # Based on the size of the compressor update views of the matrix
        rows_s, cols_s = size(solver.compressor)
        solver.mat_view = view(solver.compressed_mat, 1:rows_s, :)
        # Compress the matrix
        mul!(solver.mat_view, solver.compressor, A)
        # Update the subsolver 
        solver.R = UpperTriangular(qr!(solver.mat_view).R)
        # Compute first R' solver
        ldiv!(solver.update_vec, solver.R', solver.gradient_vec)
        # Compute second R Solve
        ldiv!(solver.gradient_vec, solver.R, solver.update_vec)
        # update the solution
        axpy!(solver.alpha, solver.gradient_vec, solver.solution_vec)
    end

    return solver.solution_vec, solver

end
