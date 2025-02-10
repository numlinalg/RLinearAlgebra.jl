"""
    Compressor

An abstract supertype for structures that contain user-controlled parameters corresponding
to techniques that compress a matrix.
"""
abstract type Compressor end

"""
   CompressorRecipe

An abstract supertype for structures that contain both the user-controlled
parameters in the `Compressor` and the memory allocations necessary for applying the 
compression technique to a particular linear system.
"""
abstract type CompressorRecipe end

# Function skeletons
"""
    complete_compressor(compressor::Compressor, A::AbstractMatrix, b::AbstractVector)

A function that uses the information in the `Compressor`, the matrix `A`, 
and the constrant vector `b` to form a `CompressorRecipe` that can then be multiplied with
matrices and vectors.

### Arguments
- `compress::Compressor`, a compressor object.
- `A::AbstractMatrix`, a matrix that the returned CompressorRecipe may be applied to.
- `b::AbstractVector`, a vector that the returned CompressorRecipe may be applied to.

### Outputs
- A `CompressorRecipe` that can be applied to matrices and vectors through the use of the 
    multiplication functions.
"""
function complete_compressor(compress::Compressor, A::AbstractMatrix, b::AbstractVector)
    return 
end

"""
    update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)

A function that updates a `CompressorRecipe` with new random components, possibly based on
information contained in `A::AbstractMatrix` and `b::AbstractMatrix`.

### Arguments
- `S::CompressorRecipe`, A preallocated CompressorRecipe.
- `A::AbstractMatrix`, a matrix that could be used to update the compressor.
- `b::AbstractVector`, a vector that could be used to update the compressor.

### Outputs
- Will generate an updated version of `S` based on the information obtained from A, b.
    For some compression techniques that are data oblivious this simply means generating new
    random entries in `S`.
"""
function update_compressor!(
        S::CompressorRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector, 
        x::AbstractVector
    )
    update_compressor!(S, A, b)
    return nothing
end

function update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)
    update_compressor!(S, A)
    return nothing
end
# Implement the * operator  for matrix matrix multiplication
function (*)(S::CompressorRecipe, v::AbstractVector)
    s_rows, s_cols = size(S)
    len_v = size(v, 1)
    @assert len_v == s_cols "Vector has $len_v entries and is not compatible with matrix with $s_cols columns."  
    output = zeros(s_rows)
    mul!(output, S, v, 1.0, 0.0)
    return output
end

function mul!(x::AbstractVector, S::CompressorRecipe, y::AbstractVector)
    mul!(x, S, y, 1.0, 0.0)
    return
end

# Implement the * operator for matrix matrix multiplication
# The left multiplication version
function (*)(S::CompressorRecipe, A::AbstractMatrix)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    @assert a_rows == s_cols "Matrix A has $a_rows rows while S has $s_cols columns."
    B = zeros(s_rows, a_cols)
    mul!(B, S, A, 1.0, 0.0)
    return B
end

function mul!(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)
    mul!(C, S, A, 1.0, 0.0)
end

# The right multiplication version
function (*)(A::AbstractMatrix, S::CompressorRecipe)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    @assert s_rows == a_cols "Matrix A has $a_cols cols while S has $s_rows rows."
    B = zeros(a_rows, s_cols)
    mul!(B, A, S, 1.0, 0.0)
    return B
end

function mul!(C::AbstractMatrix, A::AbstractMatrix, S::CompressorRecipe)
    mul!(C, A, S, 1.0, 0.0)
    return
end

# Now implement the size functions for Compressors
function Base.size(S::CompressorRecipe)
    return S.n_rows, S.n_cols
end

function Base.size(S::CompressorRecipe, dim::Int64)
    return dim == 1 ? S.n_rows : S.n_cols
end

# Wrapper functions for all compressors

#Define wrappers for the adjoint and transpose of a compressoor
"""
    CompressorAdjoint{S<:CompressorRecipe} <: CompressorRecipe

A structure for the adjoint of a compression recipe.

### Fields
- `Parent::CompressorRecipe`, the CompressorRecipe the adjoint is being applied to..
"""
struct CompressorAdjoint{S<:CompressorRecipe} <: CompressorRecipe
    parent::S
end

Adjoint(A::CompressorRecipe) = CompressorAdjoint{typeof(A)}(A)
adjoint(A::CompressorRecipe) = Adjoint(A)
# Undo the transpose
adjoint(A::CompressorAdjoint{<:CompressorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::CompressorRecipe) = Adjoint(A)
# Undo the transpose wrapper
transpose(A::CompressorAdjoint{<:CompressorRecipe}) = A.parent
# Implement the size functions for adjoints
function Base.size(S::CompressorAdjoint{<:CompressorRecipe})
    return S.parent.n_cols, S.parent.n_rows
end

function Base.size(S::CompressorAdjoint{<:CompressorRecipe}, dim::Int64)
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end


function mul!(
        C::AbstractMatrix, 
        S::CompressorAdjoint{<:CompressorRecipe}, 
        A::AbstractMatrix, 
        alpha, 
        beta
    )
    # To advoid memory allocations store mul! result in transpose of C i.e. C' = A' * S
    # this will give us C = S' * A as desired
    mul!(transpose(C), transpose(A), S.parent, alpha, beta)
    return
end

function mul!(
        C::AbstractMatrix, 
        A::AbstractMatrix, 
        S::CompressorAdjoint{<:CompressorRecipe}, 
        alpha, 
        beta
    )
    # To advoid memory allocations store mul! result in transpose of C i.e. C' = S * A'
    # this will give us C = A * S' as desired
    mul!(transpose(C), S.parent, transpose(A), alpha, beta)
    return
end

function mul!(
        x::AbstractVector, 
        S::CompressorAdjoint{<:CompressorRecipe}, 
        y::AbstractVector, 
        alpha, 
        beta
    )
    # Because the direction of multiplication is based on size compatability no transposing 
    n_rows, n_cols = size(S)
    S.parent.n_rows = n_rows
    S.parent.n_cols = n_cols
    mul!(x, S.parent, y, alpha, beta)
    # Return the sizes to the original values which is inverse order of size S
    S.parent.n_rows = n_cols
    S.parent.n_cols = n_rows
    return
end

# Dimension testing for Compressors 
"""
    left_mat_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)

Function to test the dimensions of the CompressorRecipe and matrices when applying a 
    compression matrix to a matrix from the left.

# Arguments
- `C::AbstractMatrix`, A matrix where the output will be stored.
- `S::CompressorRecipe`, The compression matrix information.
- `A::AbstractMatrix`, The matrix the compressor is being applied to from the left.

# Outputs
- Will assert an error if one of the relevant dimensions of the three inputs is incompatible 
    with the others.
"""
function left_mat_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    c_rows, c_cols = size(C)
    @assert a_rows == s_cols "Matrix A has $a_rows rows while S has $s_cols columns."
    @assert a_cols == c_cols "Matrix A has $a_cols columns while C has $c_cols columns."
    @assert c_rows == s_rows "Matrix C has $c_rows rows while S has $s_rows rows."
    return
end

"""
    right_mat_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix), S::CompressorRecipe

Function to test the dimensions of the CompressorRecipe and matrices when applying a 
    compression matrix to a matrix from the right. 

### Arguments
- `C::AbstractMatrix`, A matrix where the output will be stored.
- `S::CompressorRecipe`, The compression matrix information.
- `A::AbstractMatrix`, The matrix the compressor is being applied to from the right.

### Outputs
- Will assert an error if one of the relevant dimensions of the three inputs is incompatible
    with the others.
"""
function right_mat_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix, S::CompressorRecipe)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    c_rows, c_cols = size(C)
    @assert a_cols == s_rows "Matrix A has $a_cols columns while S has $s_rows rows."
    @assert c_cols == s_cols "Matrix C has $c_cols columns while S has $s_cols columns."
    @assert c_rows == a_rows "Matrix C has $c_rows rows while A has $a_rows rows."
    return
end

"""
    vec_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)

Function to test the dimensions of the CompressorRecipe and matrices when applying a 
    compression matrix to a vector.

### Arguments
- `x::AbstractVector`, A vector where the output will be stored.
- `S::CompressorRecipe`, The compression matrix information.
- `A::AbstractVector`, The vector that the compressor is being applied to.

### Outputs
- Will assert an error if one of the relevant dimensions of the three inputs is incompatible 
    with the others.
"""
function vec_mul_dimcheck(x::AbstractVector, S::CompressorRecipe, y::AbstractVector)
    s_rows, s_cols = size(S)
    len_y = size(y, 1)
    len_x = size(x, 1)
    @assert len_y == s_cols "Vector y is of dimension $len_y while S has $s_cols columns."
    @assert len_x == s_rows "Vector x is of dimension $len_x while S has $s_rows rows."
    return
end

###################################
# Include Compressor Files
###################################
include("Compressors/sparse_sign.jl")
