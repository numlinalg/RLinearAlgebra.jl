
"""
    LinSysBlockRowSRHT <: LinSysBlkRowSampler

A mutable structure with fields to handle SRHT row sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `blockSize::Int64`, the size of blocks being chosen
- `paddedSize::Int64`, the size of the matrix when padded
- `block::Union{Vector{Int64}, Nothing}`, storage for block indices
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockRowSRHT()` defaults to setting `blockSize` to 2.
"""
mutable struct LinSysBlockRowSRHT <: LinSysBlkRowSampler
    blockSize::Int64
    paddedSize::Int64
    block::Union{Vector{Int64}, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockRowSRHT(blockSize) = LinSysBlockRowSRHT(
                                                   blockSize, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )
LinSysBlockRowSRHT() = LinSysBlockRowSRHT(2, 0, nothing, nothing, nothing, nothing, 0.0)

# Common sample interface for linear systems
function sample(
    type::LinSysBlockRowSRHT,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        type.paddedSize = m
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, m), 1) != 0
            type.paddedSize = Int64(2^(div(log(2, m),1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(type.paddedSize, n)
            type.bp = zeros(type.paddedSize)
            # Pad matrix and constant vector
            type.Ap[1:m, :] .= A
            type.bp[1:m] .= b
        else
            type.Ap = A
            type.bp = b
        end
        # Compute scaling and sign flips
        type.scaling = sqrt(type.blockSize / type.paddedSize)
        type.signs = bitrand(type.paddedSize)
        # Apply FWHT to padded matrix and vector
        fwht!(type.bp, signs = type.signs, scaling = type.scaling)
        for i = 1:n
            @views fwht!(type.Ap[:, i], signs = type.signs, scaling = type.scaling)
        end
        
        type.block = zeros(Int64, type.blockSize) 
    end

    type.block .= sample(1:type.paddedSize, type.blockSize, replace = false) 
    SA = type.Ap[type.block, :]
    Sb = type.bp[type.block]
    # Residual of the linear system
    res = SA * x - Sb
    return type.block, SA, res
end
