# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowSparseUnifSampler <: LinSysVecRowSelect

A mutable structure that specifies sampling a proporition of the
rows of a linear system, scaling each using independent uniform 
random variables, and then taking their sum. The proportion of rows
sampled without replacement is given by (at most) `sparsity` which must be 
between 0 and 1 (not inclusive).

# Fields
- `sparsity::Float64`

# Constructors
- `LinSysVecRowSparseUnifSampler()` defaults the `sparsity` level to 0.2.
"""
mutable struct LinSysVecRowSparseUnifSampler <: LinSysVecRowSelect
    sparsity::Float64
    function LinSysVecRowSparseUnifSampler(sparsity::Float64)
        if sparsity <= 0.0 || sparsity >= 1.0
            DomainError(sparsity, "Must be strictly between 0.0 and 1.0") |>
                throw
        else
            return new(sparsity)
        end
    end
end

LinSysVecRowSparseUnifSampler() = LinSysVecRowSparseUnifSampler(0.2)

function sample(
    type::LinSysVecRowSparseUnifSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # Determine indices of equations from the system to use
    numrows = floor(Int64, type.sparsity * size(A,1))
    indices = randperm(size(A,1))[1:numrows]

    # Generate random coefficients  
    u_vars = rand(numrows) 

    # Return linear combinations
    return A[indices,:]'*u_vars, dot(b[indices], u_vars)
end

#export LinSysVecRowSparseUnifSampler