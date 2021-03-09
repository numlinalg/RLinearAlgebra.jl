# Implementation of Blendenpik with Gaussian row mixing for the solution of least squares
# problem ||Ax - b||₂ where A has full column rank.
#
# This method for other row mixing strategies is described in
#
# Avron, Haim, Petar Maymounkov, and Sivan Toledo. "Blendenpik: Supercharging LAPACK's
# least-squares solver." SIAM Journal on Scientific Computing 32, no. 3 (2010): 1217-1236.
#
# January 2021

"""
    blendenpick_gauss(A, b; r)

Solves the least squares problem with coefficient `A` and constant `b`, where `A` has full
column rank, using the blendenpick method with Gaussian row mixing. The number of sampled
rows, `r`, defaults to the number of columns.
"""
function blendenpick_gauss(
    A::Matrix{T},  # Coefficient matrix of system
    b::Vector{T};  # Constant vector of system
    r::Int=size(A, 2) + 0,  # Size of row sample
    verbose::Bool=false  #Show stats from lsqr solver
) where T <: Real
    m = size(A, 1)              # Number of rows in A

    # Mix rows of a A with Gaussians to generate r by size(A, 2) matrix
    A_mixed = randn(r, m) * A

    # Generate preconditioner using R⁻ factor of qr decomposition of mixed matrix
    _, R = qr(A_mixed)
    Rinv = R \ I

    # Run lsqr on transformed systems
    y, stats = lsqr(A * Rinv, b)

    verbose && show(stats)

    # Recover and return solution to original system
    return Rinv * y
end
