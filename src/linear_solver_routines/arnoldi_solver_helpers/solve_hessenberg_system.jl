# This file is part of RLinearAlgebra.jl

"""
    form_and_solve_hessenberg_system!(x::AbstractVector, beta::Float64, H::AbstractMatrix, 
        V::AbstractMatrix, k::Int64)

Forms the constant vector `y`, solves `H[1:k, 1:k] z = y`, and updates the vector `x`
by adding `V[:, 1:k] * z`. This is an approximate solution to the original system of
equations `Ax = b`.

!!! Remark
    This function is used in `src/linear_solver_routines/arnoldi_solver.jl` and in
    `src/linear_solver_routines/randomized_arnoldi_solver.jl`
"""
function form_and_solve_hessenberg_system!(
    x::AbstractVector, 
    beta::Float64, 
    H::AbstractMatrix, 
    V::AbstractMatrix, 
    k::Int64)

    # form constant vector
    y = zeros(k)
    y[1] = beta

    # solve upper hessenberg system
    z = H[1:k, 1:k] \ y

    # form approximation to solution of original linear system
    x .+= V[:, 1:k] * z
end