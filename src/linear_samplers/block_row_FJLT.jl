"""
    LinSysBlockRowFJLT <: LinSysBlkRowSampler

A mutable structure with fields to handle FJLT row sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `block_size::Int64`, the size of the sketching dimension
- `sparsity::Float64`, the sparsity of the sampling matrix
- `padded_size::Int64`, the size of the matrix when padded
- `sampling_matrix::Union{SparseMatrixCSC, Nothing}`, storage for sparse sketching matrix 
- `hadamard::Union{AbstractMatrix, Nothing}`, storage for the hadamard matrix.
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockRowFJLT()` defaults to setting `sparsity` to .3 and the blocksize to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockRowFJLT <: LinSysBlkRowSampler
    block_size::Int64
    sparsity::Float64 
    padded_size::Int64
    sampling_matrix::Union{SparseMatrixCSC, Nothing}
    hadamard::Union{AbstractMatrix, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockRowFJLT(;blocksize = 2, sparsity = .3) = LinSysBlockRowFJLT(
                                                   blocksize,
                                                   sparsity, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )

# Common sample interface for linear systems
function sample(
    type::LinSysBlockRowFJLT,
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
            @views type.Ap[1:m, :] .= A
            @views type.bp[1:m] .= b
        else
            type.padded_size = m
            type.Ap = A
            type.bp = b
        end
        type.hadamard = hadamard(type.padded_size)
        # Compute scaling and sign flips
        type.scaling = sqrt(type.block_size / (type.padded_size * type.sparsity))
        type.signs = bitrand(type.padded_size)
        # Apply FWHT to padded matrix and vector
        fwht!(type.bp, signs = type.signs, scaling = type.scaling)
        for i = 1:n
            Av = view(type.Ap, :, i)
            # Perform the fast walsh hadamard transform and update the ith row of Ap
            @views fwht!(Av, signs = type.signs, scaling = type.scaling)
        end
        
    end

    type.sampling_matrix = sprandn(type.block_size, type.padded_size, type.sparsity) 
    SA = type.sampling_matrix * type.Ap
    Sb = type.sampling_matrix * type.bp
    # Residual of the linear system
    res = SA * x - Sb
    sgn = [type.signs[i] ? 1 : -1 for i in 1:type.padded_size]
    return type.sampling_matrix * (sgn .* type.hadamard) .* type.scaling, SA, res
end
