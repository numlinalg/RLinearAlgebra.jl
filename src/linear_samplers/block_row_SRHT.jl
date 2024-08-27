"""
    LinSysBlockRowSRHT <: LinSysBlkRowSampler

A mutable structure with fields to handle SRHT row sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `block_size::Int64`, the size of blocks being chosen
- `padded_size::Int64`, the size of the matrix when padded
- `block::Union{Vector{Int64}, Nothing}`, storage for block indices
- `hadamard::Union{AbstractMatrix, Nothing}`, storage for the hadamard matrix.
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockRowSRHT()` defaults to setting `block_size` to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockRowSRHT <: LinSysBlkRowSampler
    block_size::Int64
    padded_size::Int64
    block::Union{Vector{Int64}, Nothing}
    hadamard::Union{AbstractMatrix, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockRowSRHT(block_size) = LinSysBlockRowSRHT(
                                                   block_size, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )
LinSysBlockRowSRHT() = LinSysBlockRowSRHT(2, 0, nothing, nothing, nothing, nothing, nothing, 0.0)

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
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, m), 1) != 0
            type.padded_size = Int64(2^(div(log(2, m), 1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(type.padded_size, n)
            type.bp = zeros(type.padded_size)
            # Pad matrix and constant vector
            type.Ap[1:m, :] .= A
            type.bp[1:m] .= b
        else
            type.padded_size = m
            type.Ap = A
            type.bp = b
        end
        type.hadamard = hadamard(type.padded_size)
        # Compute scaling and sign flips
        type.scaling = sqrt(type.block_size / type.padded_size)
        type.signs = bitrand(type.padded_size)
        # Apply FWHT to padded matrix and vector
        fwht!(type.bp, signs = type.signs, scaling = type.scaling)
        for i = 1:n
            Av = view(type.Ap, :, i)
            # Perform the fast walsh hadamard transform and update the ith row of Ap
            fwht!(Av, signs = type.signs, scaling = type.scaling)
        end
        
        type.block = zeros(Int64, type.block_size) 
    end

    type.block .= randperm(type.padded_size)[1:type.block_size] 
    SA = type.Ap[type.block, :]
    Sb = type.bp[type.block]
    # Residual of the linear system
    res = SA * x - Sb
    return ((sgn .* type.hadamard) .* type.scaling)[type.block, :], SA, res
end
