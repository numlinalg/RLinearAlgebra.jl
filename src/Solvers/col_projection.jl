"""
    col_projection <: Solver

An implementation of a column projection solver. Specifically, it is a solver that iteratively
    updates a solution by projection the solution onto a compressed rowspace of the linear 
    system.

# Mathmatical Description 

Let `A` be an `m \\times n` matrix and consider the consistent linear system `Ax = b`. 
When `n > m`, this system has infinitely many solutions, forming an affine subspace:

``x \\in {x \\in \\mathbb{R}^n : Ax = b}``.

Column projection methods iteratively refine the estimate `x` by solving compressed normal 
equations via a sketching matrix `S`. Letting ``\\tilde{A} = A S``, and the initial residual 
being ``Ax - b`.

Then the update is:

``\\Delta x = (\\tilde{A}^\\top \\tilde{A})^{-1} \\tilde{A}^\\top r``

``x_{+} = x - \\alpha S \\Delta x``

In the scalar sketch case (i.e., one column), the update simplifies to:

``x_{+} = x - \\alpha S \\cdot \\frac{\\langle A S, r \\rangle}{\\| A S \\|^2}``

where `\\alpha` is an over-relaxation parameter. The sketching matrix `S` can be random (e.g. 
SparseSign, Sampling) or deterministic. 

The residual is updated by:

``r_{+} = r - \\tilde{A} \\Delta x``

# Fields
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
affect convergence.
- `compressor::Compressor`, a technique for forming the compressed column space of the linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed column space.

# Constructor
    col_projection(;
        alpha::Float64 = 1.0,
        compressor::Compressor = SparseSign(cardinality=Right()), 
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        sub_solver::SubSolver = QRSolver(), 
    )
## Keywords
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
    affect convergence. By default this value is 1.
- `compressor::Compressor`, a technique for forming the compressed column space of the 
    linear system. By default it's SparseSign compressor.
- `log::Logger`, a technique for logging the progress of the solver. By default it's the 
    basic logger.
- `error::SolverError`, a method for estimating the progress of the solver. By default it's 
    the FullResidual error.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
    compressed rowspace. When the `compression_dim = 1` this is not used.

## Returns 
- A `col_projection` object.
"""
mutable struct col_projection <: Solver 
    alpha::Float64
    compressor::Compressor
    log::Logger
    error::SolverError
    sub_solver::SubSolver
    function col_projection(alpha, compressor, log, error, sub_solver) 
        if typeof(compressor.cardinality) != Right
            @warn "Compressor has cardinality `Left` but col_projection\
            compresses from the `Right`."
        end

        new(alpha, compressor, log, error, sub_solver)
    end

end

function col_projection(;
    alpha::Float64 = 1.0,
    compressor::Compressor = SparseSign(cardinality=Right()), 
    log::Logger = BasicLogger(),
    error::SolverError = FullResidual(),
    sub_solver::SubSolver = QRSolver(), 
)
    return  col_projection(
        alpha, 
        compressor, 
        log, 
        error, 
        sub_solver
    )
end

#------------------------------------------------------------------
"""
    col_projectionRecipe{
        T<:Number, 
        V<:AbstractVector,
        M<:AbstractArray, 
        MV<:SubArray,
        C<:CompressorRecipe, 
        L<:LoggerRecipe,
        E<:SolverErrorRecipe, 
        B<:SubSolverRecipe
    } <: SolverRecipe

A mutable structure containing all information relevant to the col_projection solver. It 
    is formed by calling the function `complete_solver` on `col_projection` solver, which 
    includes all the user controlled parameters, and the linear system matrix `A` and constant 
    vector `b`.

# Fields
- `S::CompressorRecipe`, a technique for forming the compressed column space of the 
  linear system.
- `log::LoggerRecipe`, a technique for logging the progress of the solver.
- `error::SolverErrorRecipe`, a method for estimating the progress of the solver.
- `sub_solver::SubSolverRecipe`, a technique to perform the projection of the solution 
  onto the compressed column space.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
  affect convergence.
- `compressed_mat::AbstractMatrix`, a matrix container for storing the compressed matrix. 
  Will be set to be the largest possible block size.
- `solution_vec::AbstractVector`, a vector container for storing the current solution 
  to the linear system.
- `update_vec::AbstractVector`, a vector container for storing the update to the solution.
- `mat_view::SubArray`, a container for storing a view of the compressed matrix container. 
  Using views here allows for variable block sizes.
- `residual_vec::AbstractVector`, a vector container for storing the residual at each 
  iteration.
"""
mutable struct col_projectionRecipe{
    T<:Number, 
    V<:AbstractVector,
    M<:AbstractArray, 
    MV<:SubArray,
    C<:CompressorRecipe, 
    L<:LoggerRecipe,
    E<:SolverErrorRecipe, 
    B<:SubSolverRecipe
   } <: SolverRecipe
    S::C
    log::L
    error::E
    sub_solver::B
    alpha::Float64
    compressed_mat::M
    solution_vec::V
    update_vec::V
    mat_view::MV
    residual_vec::V
end

#------------------------------------------------------------------
function complete_solver(
    solver::col_projection, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)

    # Dimension checking will be performed in the complete_compressor
    compressor = complete_compressor(solver.compressor, A, b)
    logger = complete_logger(solver.log)
    error = complete_error(solver.error, solver, A, b) 
    # Check that required fields are in the types
    if !isdefined(error, :residual)
        throw(
            ArgumentError(
                "ErrorRecipe $(typeof(error)) does not contain the \
                field 'residual' and is not valid for a col_projection solver."
            )
        )
    end

    if !isdefined(logger, :converged)
        throw(
            ArgumentError(
                "LoggerRecipe $(typeof(logger)) does not contain \
                the field 'converged' and is not valid for a col_projection solver."
            )
        )
    end
    # Assuming that max_it is defined in the logger
    alpha::Float64 = solver.alpha 
    # We assume the user is using compressors to only decrease dimension
    n_rows::Int64 = compressor.n_rows
    n_cols::Int64 = compressor.n_cols
    initial_size = n_rows 
    sample_size = n_cols  
    rows_a, cols_a = size(A)
    # Allocate the information in the buffer using the types of A and b
    compressed_mat = typeof(A)(undef, rows_a, sample_size) 
    residual_vec = typeof(b)(undef, rows_a)
    # Since sub_solver is applied to compressed matrices use here
    sub_solver = complete_sub_solver(solver.sub_solver, compressed_mat, residual_vec)
    mat_view = view(compressed_mat, :, 1:sample_size) 
    solution_vec = x  
    update_vec = typeof(x)(undef, n_cols)  
    return col_projectionRecipe{eltype(A), 
                        typeof(b), 
                        typeof(A), 
                        typeof(mat_view),
                        typeof(compressor),
                        typeof(logger),
                        typeof(error),
                        typeof(sub_solver)
                        }(compressor, 
                        logger, 
                        error,
                        sub_solver,
                        alpha,
                        compressed_mat,
                        solution_vec,
                        update_vec,
                        mat_view,
                        residual_vec
                        )
end

"""
    col_proj_update!(solver::col_projectionRecipe)

A function that performs the column projection update when the compression dimension 
    is one. If ``a`` is the resulting compression of the transpose of the coefficient matrix,
    and ``r`` is the current residual, then we perform the update:

``x = x - \\alpha S \\frac{\\langle a, r \\rangle}{\\|a\\|_2^2}``,

    where `S` is the compression operator and `a = A S`.

# Arguments
- `solver::col_projectionRecipe`, the solver information required for performing the update.

# Outputs
- returns `nothing`
"""
function col_proj_update!(solver::col_projectionRecipe)
    # one-dimensional subarray
    scaling = solver.alpha * dot(solver.mat_view, solver.residual_vec) 
    scaling /= dot(solver.mat_view, solver.mat_view)
    scaling_vec = fill(scaling, 1)
    # x_new = x_old - alpha * S * update_vec
    mul!(solver.solution_vec, solver.S, scaling_vec, -1.0, 1.0)
    # recompute the residual
    mul!(solver.residual_vec, solver.mat_view, scaling_vec, -1.0, 1.0)
    return nothing
end

"""
    col_proj_update_block!(solver::col_projectionRecipe)

A function that performs the column projection update when the compression dimension 
    is greater than 1. In the block case, where the compressed matrix is 
    ``\\tilde A = A S`` and the residual is ``r = b - A x``, we perform the update:

``x = x - \\alpha S (\\tilde A^\\top \\tilde A)^\\dagger \\tilde A^\\top r``,

    where `S` is the compression operator and the update projects the solution onto the 
    column space of the matrix `A`.

# Arguments
- `solver::col_projectionRecipe`, the solver information required for performing the update.

# Outputs
- returns `nothing`
"""
function col_proj_update_block!(solver::col_projectionRecipe)
    # update the subsolver and solve for update vector
    update_sub_solver!(solver.sub_solver, solver.mat_view)
    ldiv!(solver.update_vec, solver.sub_solver, solver.residual_vec)
    # x_new = x_old - alpha * S * update_vec
    mul!(solver.solution_vec, solver.S, solver.update_vec, -solver.alpha, 1.0)
    # recomputet the residual
    mul!(solver.residual_vec, solver.mat_view, solver.update_vec, -solver.alpha, 1.0)
    return nothing
end

function rsolve!(
    solver::col_projectionRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    # initialization
    reset_logger!(solver.log)
    solver.solution_vec = x
    # compute the residual b-Ax
    mul!(solver.residual_vec, A, solver.solution_vec)
    solver.residual_vec .-= b

    for i in 1:solver.log.max_it
        err = compute_error(solver.error, solver, A, b)
        println(err)
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return nothing
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.S, x, A, b)
        # based on size of new compressor update views of matrix
        # this should not result in new allocations
        rows_s, cols_s =  size(solver.S)  
        solver.mat_view = view(solver.compressed_mat, :, 1:cols_s)
        # compress the matrix
        mul!(solver.mat_view, A, solver.S) 
        
        # Solve the undetermined sketched linear system and update the solution
        if size(solver.mat_view, 2) == 1
            col_proj_update!(solver)
        else
            col_proj_update_block!(solver)
        end

    end

    return nothing 
end
