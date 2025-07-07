mutable struct IHSRecipe{M<:AbstractArray, MV<:SubArray, V<:AbstractVector}
    logger::LoggerRecipe
    compressor::CompressorRecipe
    sub_solver::SubSolverRecipe
    compressed_mat::M
    mat_view::MV
    res::V
    grad::V
    update_vec::V
    solution_vec::V
end

function complete_solver()

end

function rsolve!(solver::IHSRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    reset_logger!(solver.logger)
    solver.solution_vec = x
    err = 0.0
    for i in 1:solver.log.max_it
        mul!(solver.res, A, solver.solution_vec, -1.0, 1.0)
        mul!(solver.grad, A', solver.res)
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
        update_sub_solver!(solver.sub_solver, solver.mat_view)
        ldiv!(solver.update_vec, solver.sub_solver, solver.grad)
        axpy!(1.0, solver.update_vec, 1.0, solver.solution_vec)
    end

    return solver.solution_vec, solver

end
