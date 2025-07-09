"""
    col_projection <: Solver

An implementation of a column projection solver. Specifically, it is a solver that iteratively
    updates a solution by projection the solution onto a compressed columnspace of the linear 
    system.

# Fields
- `S::Compressor`, a technique for forming the compressed columnspace of the linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
compressed rowspace.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
affect convergence.
"""
#------------------------------------------------------------------
mutable struct col_projection <: Solver 
    alpha::Float64
    S::Compressor
    log::Logger
    error::SolverError
    sub_solver::SubSolver
end

#------------------------------------------------------------------
function col_projection(;
    alpha::Float64 = 1.0,
    S::Compressor = SparseSign(), 
    log::Logger = BasicLogger(),
    error::SolverError = FullResidual(),
    sub_solver::SubSolver = QRSolver(), 
)

# Intialize the datatype setting unkown elements to empty versions of correct datatype
return  col_projection(alpha, S, log, error, sub_solver)
end

#------------------------------------------------------------------
mutable struct col_projectionRecipe{T<:Number, 
    V<:AbstractVector,
    M<:AbstractMatrix, 
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
compressed_vec::V
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
# Check the dimensions align


# Dimension checking will be performed in the complete_compressor
compressor = complete_compressor(solver.S, A, b)
logger = complete_logger(solver.log, A, b)
error = complete_error(solver.error, A, b) 
# Check that required fields are in the types
@assert isdefined(error, :residual) "ErrorRecipe $(typeof(error)) does not contain the field 'residual' and is not valid for a col_projection solver."
@assert isdefined(logger, :converged) "LoggerRecipe $(typeof(logger)) does not contain the field 'converged' and is not valid for a col_projection solver."
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
compressed_vec = typeof(b)(undef, rows_a) 
# Since sub_solver is applied to compressed matrices use here
sub_solver = complete_sub_solver(solver.sub_solver, compressed_mat, compressed_vec)
mat_view = view(compressed_mat, :, 1:sample_size)   ######################
solution_vec = x  
update_vec = typeof(x)(undef, n_cols)  
residual_vec = typeof(x)(undef, rows_a)
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
                       compressed_vec,
                       solution_vec,
                       update_vec,
                       mat_view,
                       residual_vec
                      )
end

#---------------------------------------------------------------
function rsolve!(
    solver::col_projectionRecipe, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
# initialization
solver.solution_vec = x
err = 0.0 
# compute the residual b-Ax
mul!(solver.residual_vec, A, x, -1, 0)
solver.residual_vec .+= b 
for i in 1:solver.log.max_it
    # generate a new version of the compression matrix
    update_compressor!(solver.S, A, b, x)
    # based on size of new compressor update views of matrix
    # this should not result in new allocations
    rows_s, cols_s =  size(solver.S)  
    solver.mat_view = view(solver.compressed_mat, :, 1:cols_s) ################## A_new is mxk
    # compress the matrix and constant vector
    mul!(solver.mat_view, A, solver.S)  ################## A_new=AS
    err = compute_error(solver.error, solver, A, b)
    # Update log adds value of err to log and checks stopping
    update_logger!(solver.log, err, i)
    if solver.log.converged
        return solver.solution_vec, solver.log[solver.log .> 0]
    end

    # sub-solver needs to designed for new compressed matrix
    update_sub_solver!(solver.sub_solver, solver.mat_view)
    # use sub-solver to find update the solution
    sub_solve!(solver.update_vec, solver.sub_solver, solver.residual_vec) #########
    # Using over-relaxation parameter, alpha, to update solution
    # x += alpha * S * update_vec
    mul!(solver.solution_vec, solver.S, solver.update_vec, solver.alpha, 1.0)
    # Compute the residual r-ASx
    mul!(solver.residual_vec, solver.mat_view, solver.solution_vec, -1.0, 1.0)  ### nothing changed
end

return solver.solution_vec, solver.log
end

#----------------------------------------------------------------------
"""
    FullResidual <: SolverErrorRecipe

A structure for the full residual, `b-Ax`.

# Fields
- `residual::AbstractVector`, a container for the residual `b-Ax`.
"""

struct LSgradient <: SolverError

end

mutable struct LSgradientRecipe{V<:AbstractVector} <: SolverErrorRecipe
    gradient::V
end

function complete_error(error::LSgradient, A::AbstractMatrix, b::AbstractVector)
    return LSgradientRecipe{typeof(b)}(zeros(size(b,1)))
end

function compute_error(
        error::LSgradientRecipe, 
        solver::col_projectionRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    copyto!(error.gradient, b)
    mul!(error.gradient, A', solver.residual_vec, -1.0, 1.0) ####r=b-A(Sx)
    
    return dot(error.gradient, error.gradient)
end

"""
    CompressedLSgradientRecipe <: SolverErrorRecipe

A structure for the compressed residual, `b-ASx`.

# Fields
- `gradient::AbstractVector`, a container for the compressed residual, `b-ASx`.
- `gradient_view::SubArray`, a view of the residual container to handle varying compression
sizes.
"""

struct compressedLSgradient <: SolverError

end

mutable struct compressedLSgradientRecipe{V<:AbstractVector, S<:SubArray} <: SolverErrorRecipe
    gradient::V
    gradient_view::S
end

function complete_error(error::compressedLSgradient, A::AbstractMatrix, b::AbstractVector)
    gradient = zeros(size(b,1))
    gradient_view = view(gradient, 1:1)
    return CompressedLSgradientRecipe{typeof(gradient),typeof(gradient_view)}(gradient, gradient_view)
end

function compute_error(
    error::compressedLSgradientRecipe, 
    solver::col_projectionRecipe, 
    A::AbstractMatrix, 
    b::AbstractVector
)::Float64
    rows_s = size(solver.S, 1)
    error.gradient_view = view(error.gradient, 1:rows_s)
    mul!(error.gradient_view, solver.mat_view', solver.residual_vec, -1.0, 1.0)
    return dot(error.gradient_view, error.gradient_view)
end
