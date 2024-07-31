# This code was Written by Nathaniel Pritchard
"""
    LinSysBlkRowGaussSampler <: LinSysBlkRowSampler

A mutable structure with fields to handle Guassian row sketching where a Gaussian matrix
is multiplied by the matrix `A` from the left. 

# Fields
- `block_size::Int64`, Specifies the size of each block.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`, The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64`, The standard deviation of the sketch, is set to be sqrt(block_size/numberOfRows).

# Constructors
Calling `LinSysBlkRowGaussSampler()` defaults to setting `block_size` to 2.
"""
mutable struct LinSysBlkRowGaussSampler <: LinSysBlkRowSampler
    block_size::Int64
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysBlkRowGaussSampler(block_size) = LinSysBlkRowGaussSampler(block_size, nothing, 0.0)
LinSysBlkRowGaussSampler() = LinSysBlkRowGaussSampler(2, nothing, 0.0)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    m, n = size(A)
    if iter == 1
        @assert type.block_size > 0 "`block_size` must be positve."
        if type.block_size > m
            @warn "`block_size` should be less than row dimension"
        end
        # Scaling value for the matrix so it has expectation 1 when applied to unit vector
        type.scaling = sqrt(1 / type.block_size)
        # Allocate for the sketching matrix
        type.sketch_matrix = Matrix{Float64}(undef, type.block_size, m) 
    end
    
    # Fill the allocated matrix with N(0,1) entries
    randn!(type.sketch_matrix)
    # Scale the matrix to ensure it has expectation 1 when applied to unit vector
    type.sketch_matrix .*= type.scaling
    SA = type.sketch_matrix * A
    Sb = type.sketch_matrix * b
    # Residual of the linear system
    res = SA * x - Sb
    return type.sketch_matrix, SA, res
end
