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

# Docstring Components
comp_arg_list = Dict{Symbol, String}(
    :compressor => "`compressor::Compressor`, a user-specified compression method.",
    :compressor_recipe => "`S::CompressorRecipe`, a fully initialized realization for a 
    compression method for a specific matrix or collection of matrices and vectors.",
    :A => "`A::AbstractMatrix`, a target matrix for compression.",
    :C => "`C::AbstractMatrix`, A matrix where the output will be stored.",
    :b => "`b::AbstractVector`, a possible target vector for compression.",
    :x => " x::AbstractVector`, a vector that ususally represents a current iterrate 
    typically used in a solver",
    :y => "`x::AbstractVector`, a vector.",
    :z => "`x::AbstractVector`, a vector."
)

comp_output_list = Dict{Symbol, String}(
    :compressor_recipe => "A `CompressorRecipe` object."
)

comp_method_description = Dict{Symbol, String}(
    :complete_compressor => "A function that generates a `CompressorRecipe` given the 
    arguments.",
    :update_compressor => "A function that updates the `CompressorRecipe` in place given 
    arguments.",
    :mul_check => "A function that checks the compatability of arguments for multiplcation",
)
# Function skeletons
"""
    complete_compressor(compressor::Compressor, A::AbstractMatrix)

$(comp_method_description[:complete_compressor])

### Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:A]) 

### Outputs
- $(comp_output_list[:compressor_recipe])
"""
function complete_compressor(compress::Compressor, A::AbstractMatrix)
    return 
end

"""
    complete_compressor(compressor::Compressor, A::AbstractMatrix, b::AbstractVector)

$(comp_method_description[:complete_compressor])

### Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:A]) 
- $(comp_arg_list[:b]) 

### Outputs
- $(comp_output_list[:compressor_recipe])
"""
function complete_compressor(compress::Compressor, A::AbstractMatrix, b::AbstractVector)
    # If this variant is not defined for a compressor call the one with input matrix A 
    return complete_compressor(comprees, A)
end

"""
    complete_compressor(
        compressor::Compressor, 
        A::AbstractMatrix, 
        b::AbstractVector, 
        x::AbstractVector
    )

$(comp_method_description[:complete_compressor])

### Arguments
- $(comp_arg_list[:compressor])
- $(comp_arg_list[:A]) 
- $(comp_arg_list[:b]) 
- $(comp_arg_list[:x]) 

### Outputs
- $(comp_output_list[:compressor_recipe])
"""
function complete_compressor(
            compress::Compressor, 
            A::AbstractMatrix, 
            b::AbstractVector,
            x::AbstractVector
    )
    # If this variant is not defined for a compressor call the one with input matrix A 
    return complete_compressor(comprees, A, b)
end

"""
    update_compressor!(S::CompressorRecipe, A::AbstractMatrix)

$(comp_method_description[:update_compressor])

### Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A]) 

### Outputs
- `Nothing` 
"""
function update_compressor!(S::CompressorRecipe, A::AbstractMatrix)
    return nothing
end

"""
    update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)

$(comp_method_description[:update_compressor])

### Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A]) 
- $(comp_arg_list[:b]) 

### Outputs
- `Nothing` 
"""
function update_compressor!(S::CompressorRecipe, A::AbstractMatrix, b::AbstractVector)
    update_compressor!(S, A)
    return nothing
end

"""
    update_compressor!(
        S::CompressorRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector,
        x::AbstractMatrix
    )

$(comp_method_description[:update_compressor])

### Arguments
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A]) 
- $(comp_arg_list[:b]) 
- $(comp_arg_list[:x]) 

### Outputs
- `Nothing` 
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

# Dimension testing for Compressors 
"""
    left_mat_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)

$(comp_method_description[:mul_check] * " from the left.") 

### Arguments
- $(comp_arg_list[:C]) 
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:A]) 

# Outputs
- `Nothing`. 
"""
function left_mat_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    c_rows, c_cols = size(C)
    if a_rows != s_cols
        throw(DimensionMismatch("Matrix A has $a_rows rows while S has $s_cols columns."))
    elseif a_cols != c_cols
        throw(
              DimensionMismatch("Matrix A has $a_cols columns while C has $c_cols columns\
                .")
             )
    elseif c_rows != s_rows
        throw(DimensionMismatch("Matrix C has $c_rows rows while S has $s_rows rows."))
    end

    return nothing
end

"""
    right_mat_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix), S::CompressorRecipe

$(comp_method_description[:mul_check] * " from the right.") 

### Arguments
- $(comp_arg_list[:C]) 
- $(comp_arg_list[:A]) 
- $(comp_arg_list[:compressor_recipe])

# Outputs
- `Nothing`. 
"""
function right_mat_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix, S::CompressorRecipe)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    c_rows, c_cols = size(C)
    if a_cols != s_rows
        throw(DimensionMismatch("Matrix A has $a_cols columns while S has $s_rows rows."))
    elseif c_cols != s_cols
        throw(DimensionMismatch("Matrix C has $c_cols columns while S has $s_cols columns\
            ."))
    elseif c_rows != a_rows
        throw(DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows."))
    end

    return nothing
end

"""
    vec_mul_dimcheck(z::AbstractMatrix, S::CompressorRecipe, y::AbstractMatrix)

$(comp_method_description[:mul_check] * " with a vector.") 

### Arguments
- $(comp_arg_list[:z]) 
- $(comp_arg_list[:compressor_recipe])
- $(comp_arg_list[:y]) 

# Outputs
- `Nothing`. 
"""
function vec_mul_dimcheck(x::AbstractVector, S::CompressorRecipe, y::AbstractVector)
    s_rows, s_cols = size(S)
    len_y = size(y, 1)
    len_x = size(x, 1)
    if len_y != s_cols
        throw(
            DimensionMismatch("Vector y is of dimension $len_y \
                              while S has $s_cols columns.")
             )
    elseif len_x != s_rows
        throw(
            DimensionMismatch("Vector x is of dimension $len_x\
            while S has $s_rows rows.")
            )
    end

    return nothing
end
# Implement the * operator  for matrix matrix multiplication
function mul!(
        C::AbstractArray,
        S::Union{CompressorRecipe, AbstractArray},
        A::Union{CompressorRecipe, AbstractArray},
        alpha::Float64,
        beta::Float64
    )
    # This is a default functionality that only runs when multiplication is not defined for
    # the CompressorRecipe, where it returns an error in the worse case
    if typeof(B) <: CompressorRecipe
        type_B = typeof(B)
        return ArgumentError("Multiplication not defined for $type_B.")
    elseif typeof(B) <: CompressorRecipe
        type_C = typeof(C)
        return ArgumentError("Multiplication not defined for $type_C.")
    end
    
    return nothing
end

function (*)(S::CompressorRecipe, v::AbstractVector)
    s_rows, s_cols = size(S)
    len_v = length(v)
    if len_v != s_cols
        throw(DimensionMismatch("Vector has $len_v entries while matrix has $s_cols//
            columns.")) 
    end
    output = zeros(s_rows)
    mul!(output, S, v, 1.0, 0.0)
    return output
end

function mul!(x::AbstractVector, S::CompressorRecipe, y::AbstractVector)
    mul!(x, S, y, 1.0, 0.0)
    return nothing
end

# Implement the * operator for matrix matrix multiplication
# The left multiplication version
function (*)(S::CompressorRecipe, A::AbstractMatrix)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if a_rows != s_cols 
        throw(DimensionMismatch("Matrix A has $a_rows rows while S has $s_cols columns."))
    end
    B = zeros(s_rows, a_cols)
    mul!(B, S, A, 1.0, 0.0)
    return B
end

function mul!(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)
    mul!(C, S, A, 1.0, 0.0)
    return nothing
end

# The right multiplication version
function (*)(A::AbstractMatrix, S::CompressorRecipe)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if s_rows != a_cols 
        throw(DimensionMismatch("Matrix A has $a_cols cols while S has $s_rows rows."))
    end

    B = zeros(a_rows, s_cols)
    mul!(B, A, S, 1.0, 0.0)
    return B
end

function mul!(C::AbstractMatrix, A::AbstractMatrix, S::CompressorRecipe)
    mul!(C, A, S, 1.0, 0.0)
    return nothing
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

CompressorAdjoint(A::CompressorRecipe) = CompressorAdjoint{typeof(A)}(A)
adjoint(A::CompressorRecipe) = CompressorAdjoint(A)
# Undo the transpose
adjoint(A::CompressorAdjoint{<:CompressorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::CompressorRecipe) = CompressorAdjoint(A)
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
    return nothing
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
    return nothing
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
    return nothing
end


###################################
# Include Compressor Files
###################################

