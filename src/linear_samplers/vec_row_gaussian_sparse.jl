#This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowSparseGaussSampler <: LinSysVecRowSelect

A mutable structure that specifies sampling a proporition of the
rows of a linear system, scaling each using independent Gaussian 
random variables, and then taking their sum. The proportion of rows
sampled without replacement is given by (at most) `sparsity` which must be 
between 0 and 1 (not inclusive).

# Fields
- `sparsity::Float64`

# Constructors
- `LinSysVecRowSparseUnifSampler()` defaults the `sparsity` level to 0.2.
"""
mutable struct LinSysVecRowSparseGaussSampler <: LinSysVecRowSelect
    sparsity::Float64
    function LinSysVecRowSparseGaussSampler(sparsity::Float64)
        if sparsity <= 0.0 || sparsity >= 1.0
            DomainError(sparsity, "Must be strictly between 0.0 and 1.0") |>
                throw
        else
            return new(sparsity)
        end
    end
end

LinSysVecRowSparseGaussSampler() = LinSysVecRowSparseGaussSampler(0.2)

function sample(
    type::LinSysVecRowSparseGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # Determine indices of equations from the system to use
    numrows = floor(Int64, type.sparsity * size(A,1))
    indices = randperm(size(A,1))[1:numrows]

    # Generate random coefficients  
    u_vars = randn(numrows) 

    # Return linear combinations
    return A[indices,:]'*u_vars, dot(b[indices], u_vars)
end

#export LinSysVecRowSparseGaussSampler