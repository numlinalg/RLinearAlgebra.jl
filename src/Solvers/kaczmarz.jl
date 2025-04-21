"""
    Kaczmarz <: Solver

An implementation of a block Kaczmarz solver. Specifically, it is a solver that iteratively
    updates a solution by projection the solution onto a compressed rowspace of the linear 
    system.

# Fields
- `S::Compressor`, a technique for forming the compressed rowspace of the linear system.
- `log::Logger`, a technique for logging the progress of the solver.
- `error::SolverError`, a method for estimating the progress of the solver.
- `sub_solver::SubSolver`, a technique to perform the projection of the solution onto the 
compressed rowspace.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
affect convergence.
"""
mutable struct Kaczmarz <: ProjectionSolver 
    alpha::Float64
    S::Compressor
    log::Logger
    error::SolverError
    sub_solver::SubSolver
end


function Kaczmarz(;
        S::Compressor = SparseSign(), 
        log::Logger = BasicLogger(),
        error::SolverError = FullResidual(),
        sub_solver::SubSolver = LQSolver(),
        alpha::Float64 = 1.0
    )
    # Intialize the datatype setting unkown elements to empty versions of correct datatype
    return Kaczmarz(alpha,
                    S, 
                    log, 
                    error, 
                    sub_solver
                   )
end

"""
    KaczmarzRecipe{T<:Number, 
                        V<:AbstractVector,
                        M<:AbstractMatrix, 
                        VV<:SubArray,
                        MV<:SubArray,
                        C<:CompressorRecipe, 
                        L<:LoggerRecipe,
                        E<:SolverErrorRecipe, 
                        B<:SubSolverRecipe
                        } <: SolverRecipe

An mutable structure containing all information relevant to the kcazmarz solver. It is
formed by calling the function `complete_solver` on `Kaczmarz` datatype, which includes all
the user controllewd parameters, and the linear system matrix `A` and constant vector `b`.

# Fields
- `S::CompressorRecipe`, a technique for forming the compressed rowspace of the linear
system.
- `log::LoggerRecipe`, a technique for logging the progress of the solver.
- `error::SolverErrorRecipe`, a method for estimating the progress of the solver.
- `sub_solver::SubSolverRecipe`, a technique to perform the projection of the solution onto
the compressed rowspace.
- `alpha::Float64`, the over-relaxation parameter. It is multiplied by the update and can 
affect convergence.
- `compressed_mat::AbstractMatrix`, a matrix container for storing the compressed matrix. 
Will be set to be the largest possible block size.
- `compressed_vec::AbstractVector`, a vector container for storing the compressed constant
vector. 
Will be set to be the largest possible block size.
- `solution_vec::AbstractVector`, a vector container for storing the solution to the linear
system.
- `update_vec::AbstractVector`, a vector container for storing the update to the linear 
system.
- `mat_view::SubArray`, a container for storing a view of compressed matrix container. 
Using views here allows for variable block sizes.
- `vec_view::SubArray`, a container for storing a view of the compressed vector container.
Using views here allows for variable block sizes.
"""
mutable struct KaczmarzRecipe{T<:Number, 
                              V<:AbstractVector,
                              M<:AbstractArray, 
                              VV<:SubArray,
                              MV<:SubArray,
                              C<:CompressorRecipe, 
                              L<:LoggerRecipe,
                              E<:SolverErrorRecipe, 
                              B<:SubSolverRecipe
                             } <: ProjectionSolverRecipe
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
    vec_view::VV
end

function complete_solver(
        solver::Kaczmarz, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )
    # Dimension checking will be performed in the complete_compressor
    compressor = complete_compressor(solver.S, A, b)
    logger = complete_logger(solver.log, A, b)
    error = complete_error(solver.error, A, b) 
    # Check that required fields are in the types
    @assert isdefined(error, :residual) "ErrorRecipe $(typeof(error)) does not contain the \
    field 'residual' and is not valid for a kaczmarz solver."
    @assert isdefined(logger, :converged) "LoggerRecipe $(typeof(logger)) does not contain \
    the field 'converged' and is not valid for a kaczmarz solver."
    # Assuming that max_it is defined in the logger
    alpha::Float64 = solver.alpha 
    # We assume the user is using compressors to only decrease dimension
    n_rows::Int64 = compressor.n_rows
    n_cols::Int64 = compressor.n_cols
    sample_size = n_rows
    initial_size = n_cols
    rows_a, cols_a = size(A)
    # Allocate the information in the buffer using the types of A and b
    compressed_mat = zeros(eltype(A), sample_size, cols_a)
    compressed_vec = zeros(eltype(b), sample_size) 
    # Since sub_solver is applied to compressed matrices use here
    sub_solver = complete_sub_solver(solver.sub_solver, compressed_mat, compressed_vec)
    mat_view = view(compressed_mat, 1:sample_size, :)
    vec_view = view(compressed_vec, 1:sample_size)
    solution_vec = x
    update_vec = zeros(eltype(x), cols_a)
    return KaczmarzRecipe{eltype(A), 
                          typeof(b), 
                          typeof(A), 
                          typeof(vec_view),
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
                           vec_view
                          )
end


# If the compression dim is 1 exactly
function rsolve!(
        solver::KaczmarzRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
)
    solver.solution_vec = x
    err = 0.0
    for i in 1:solver.log.max_it
        err = compute_error(solver.error, solver, A, b)
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return solver.solution_vec, solver.log
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.S, A, b, x)
        mul!(solver.mat_view, solver.S, A)
        bi = dot(solver.S, b)
        # Compute the projection scaling (bi - dot(ai,x)) / ||ai||^2
        scaling = (dot(solver.mat_view, x) - bi) / dot(solver.mat_view, solver.mat_view)
        # adjust projection scaling by overelaxation scaling
        combinded_scaling = solver.alpha * scaling 
        # udpate the solution
        axpby!(combined_scaling, solver.mat_view, 1.0. solver.solution_vec)
    end

    return solver.solution_vec, solver.log
end

function rsolve!(
        solver::KaczmarzRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
)
    solver.solution_vec = x
    err = 0.0
    for i in 1:solver.log.max_it
        err = compute_error(solver.error, solver, A, b)
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return solver.solution_vec, solver.log
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.S, A, b, x)
        # based on size of new compressor update views of matrix
        # this should not result in new allocations
        rows_s, cols_s =  size(solver.S)
        solver.mat_view = view(solver.compressed_mat, 1:rows_s, :)
        solver.vec_view = view(solver.compressed_vec, 1:rows_s)
        # compress the matrix and constant vector
        mul!(solver.mat_view, solver.S, A)
        mul!(solver.vec_view, solver.S, b)
        # Compute the block residual
        mul!(solver.vec_view, solver.mat_view, solver.solution_vec, -1.0, 1.0)
        # sub-solver needs to designed for new compressed matrix
        update_sub_solver!(solver.sub_solver, solver.mat_view)
        # use sub-solver to find update the solution
        ldiv!(solver.update_vec, solver.sub_solver, solver.vec_view)
        # Using over-relaxation parameter, alpha, to update solution
        solver.solution_vec .+= solver.alpha .* solver.update_vec 
    end

    return solver.solution_vec, solver.log
end

"""
    FullResidual <: SolverErrorRecipe

A structure for the full residual, `b-Ax`.

# Fields
- `residual::AbstractVector`, a container for the residual `b-Ax`.
"""
struct FullResidual <: SolverError

end

mutable struct FullResidualRecipe{V<:AbstractVector} <: SolverErrorRecipe
    residual::V
end

function complete_error(error::FullResidual, A::AbstractMatrix, b::AbstractVector)
    return FullResidualRecipe{typeof(b)}(zeros(size(b,1)))
end

function compute_error(
        error::FullResidualRecipe, 
        solver::KaczmarzRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    copyto!(error.residual, b)
    mul!(error.residual, A, solver.solution_vec, -1.0, 1.0)
    return dot(error.residual, error.residual)
end

"""
    CompressedResidual <: ErrorRecipe

A structure for the compressed residual, `Sb-SAx`.

# Fields
- `residual::AbstractVector`, a container for the compressed residual, `Sb-SAx`.
- `residual_view::SubArray`, a view of the residual container to handle varying compression
sizes.
"""
struct CompressedResidual <: SolverError

end

mutable struct CompressedResidualRecipe{V<:AbstractVector, S<:SubArray} <: SolverErrorRecipe
    residual::V
    residual_view::S
end

function complete_error(error::CompressedResidual, A::AbstractMatrix, b::AbstractVector)
    residual = zeros(size(b,1))
    residual_view = view(residual, 1:1)
    return CompressedResidualRecipe{typeof(residual),typeof(residual_view)}(
        residual, 
        residual_view
        )

end

function compute_error(
        error::CompressedResidualRecipe, 
        solver::KaczmarzRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    rows_s = size(solver.S, 1)
    error.residual_view = view(error.residual, 1:rows_s)
    copyto!(error.residual_view, solver.vec_view)
    mul!(error.residual_view, solver.mat_view, solver.solution_vec, -1.0, 1.0)
    return dot(error.residual_view, error.residual_view)
end
