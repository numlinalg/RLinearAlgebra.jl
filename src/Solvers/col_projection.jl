"""
    col_projection <: Solver

An implementation of a column projection solver. Specifically, it is a solver that iteratively
    updates a solution by projection the solution onto a compressed rowspace of the linear 
    system.

# Fields
- `S::Compressor`, a technique for forming the compressed rowspace of the linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
compressed column space.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
affect convergence.
"""
#------------------------------------------------------------------
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

function col_proj_update!(solver::col_projectionRecipe)
    # one-dimensional subarray
    scaling = solver.alpha * dot(solver.mat_view, solver.residual_vec) 
    scaling /= dot(solver.mat_view, solver.mat_view)
    # x_new = x_old - alpha * S * update_vec
    mul!(solver.solution_vec, solver.S, scaling, -1.0, 1.0)
    # recompute the residual
    mul!(solver.residual_vec, solver.mat_view, scaling, -1.0, 1.0)
end

function col_proj_update_block!(solver::col_projectionRecipe)
    # update the subsolver and solve for update vector
    update_sub_solver!(solver.sub_solver, solver.mat_view)
    ldiv!(solver.update_vec, solver.sub_solver, solver.residual_vec)
    # x_new = x_old - alpha * S * update_vec
    mul!(solver.solution_vec, solver.S, solver.update_vec, -solver.alpha, 1.0)
    # recomputet the residual
    mul!(solver.residual_vec, solver.mat_view, solver.update_vec, -solver.alpha, 1.0)
end

#---------------------------------------------------------------
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
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return solver.solution_vec, solver.log[solver.log .> 0]
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.S, A, b, x)
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

    return solver.solution_vec, solver.log
end

#----------------------------------------------------------------------
"""
    LSgradient <: SolverErrorRecipe

A structure for the full gradient of ||b-Ax||_2^2, `A'(b-Ax)`.

# Fields
- `gradient::AbstractVector`, a container for the full gradient.
"""
struct LSgradient <: SolverError end

mutable struct LSgradientRecipe{V<:AbstractVector} <: SolverErrorRecipe
    gradient::V
end

function complete_error(error::LSgradient, A::AbstractMatrix, b::AbstractVector)
    return LSgradientRecipe{typeof(b)}(zeros(size(A,2)))
end

function compute_error(
        error::LSgradientRecipe, 
        solver::col_projectionRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    # copyto!(error.gradient, b)
    # coompute A'r
    mul!(error.gradient, A', solver.residual_vec) 
    
    return dot(error.gradient, error.gradient)
end

# """
#     CompressedLSgradientRecipe <: SolverErrorRecipe
# A structure for the compressed residual, ``.

# # Fields
# - `gradient::AbstractVector`, a container for the compressed residual, `S'A'r`.
# - `gradient_view::SubArray`, a view of the residual container to handle varying compression
# sizes.
# """

# struct compressedLSgradient <: SolverError

# end

# mutable struct compressedLSgradientRecipe{V<:AbstractVector, S<:SubArray} <: SolverErrorRecipe
#     gradient::V
#     gradient_view::S
# end

# function complete_error(error::compressedLSgradient, A::AbstractMatrix, b::AbstractVector)
#     gradient = zeros(size(b,1))
#     gradient_view = view(gradient, 1:1)
#     return CompressedLSgradientRecipe{typeof(gradient),typeof(gradient_view)}(gradient, gradient_view)
# end

# function compute_error(
#     error::compressedLSgradientRecipe, 
#     solver::col_projectionRecipe, 
#     A::AbstractMatrix, 
#     b::AbstractVector
# )::Float64
#     row_s = size(solver.S, 1)
#     error.gradient_view = view(error.gradient, 1:row_s)
#     mul!(error.gradient_view, solver.mat_view', solver.residual_vec, -1.0, 1.0) # r = 
#     return dot(error.gradient_view, error.gradient_view)
# end
