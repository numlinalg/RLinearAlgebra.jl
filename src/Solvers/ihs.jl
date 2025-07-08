mutable struct IHSRecipe{
    Type<:Number, 
    LR<:LoggerRecipe,
    CR<:CompressorRecipe,
    ER<:ErrorRecipe,
    M<:AbstractArray, 
    MV<:SubArray, 
    V<:AbstractVector
}
    logger::LR
    compressor::CR
    error::ER
    compressed_mat::M
    mat_view::MV
    residual_vec::V
    gradient_vec::V
    update_vec::V
    solution_vec::V
    R::UpperTriangular{T, M}
end

function complete_solver(
    ingredients::IHS, 
    x::AbstractVector, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    compressor = complete_compressor(ingredients.compressor, x, A, b)
    logger = complete_logger(ingredients.log) 
    error = complete_error(ingredients.error)
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
        @warn "Compression dimension not larger than column dimension this will lead to \
        singular QR decompositions, which cannot be inverted"
    end

    compressed_mat = zeros(ingredients.type, sample_size, cols_a)
    res = zeros(ingredients.type, rows_a) 
    grad = zeros(ingredients.type, cols_a) 
    update_vec = zeros(ingredients.type, cols_a) 
    solution_vec = x
    mat_view = view(compressed_mat, 1:sample_size, :)
    R = UpperTriangular(mat_view[1:cols_a])

    return IHSRecipe{
        ingredients.type,
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
    reset_logger!(solver.logger)
    solver.solution_vec = x
    err = 0.0
    for i in 1:solver.log.max_it
        mul!(solver.resdual_vec, A, solver.solution_vec, -1.0, 1.0)
        mul!(solver.gradient_vec, A', solver.res)
        err = compute_error(solver.error, solver, A, b)
        update_logger!(solver.logger, err, i)
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
        solver.R = UpperTriangular(qr!(solver.ma_view).R)
        # Compute first R' solver
        ldiv!(solver.update_vec, solver.R', solver.gradient_vec)
        # Compute second R Solve
        ldiv!(solver.gradient_vec, solver.R, solver.update_vec)
        # update the solution
        axpy!(1.0, solver.gradient_vec, 1.0, solver.solution_vec)
    end

    return solver.solution_vec, solver

end
