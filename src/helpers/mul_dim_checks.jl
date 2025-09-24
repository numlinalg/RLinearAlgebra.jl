###################################
# Docstring Components  
###################################
check_arg_list = Dict{Symbol,String}(
    :recipe => "`S::Union{CompressorRecipe, ApproximatorRecipe}`, a fully 
    initialized realization for a compression or approximator method for a 
    specific AbstractArray or operator.",
    :recipe_adjoint => "`S::Union{CompressorAdjoint, ApproximatorAdjoint}`, the 
    representation of an adjoint of a compression or approximator operator.",
    :A => "`A::AbstractArray`, a target AbstractArray for compression.",
    :C => "`C::AbstractArray`, a AbstractArray where the output will be stored.",
)


check_method_description = Dict{Symbol,String}(
    :mul_check => "A function that checks the compatibility of arguments for 
    multiplication",
)

############################################
# Compressor-Array Multiplication Dim Checks 
############################################

"""
    left_mul_dimcheck(C::AbstractMatrix, S::CompressorRecipe, A::AbstractMatrix)

$(check_method_description[:mul_check] * " from the left.")

# Arguments
- $(check_arg_list[:C])
- $(check_arg_list[:recipe])
- $(check_arg_list[:A])

# Returns 
- `nothing`

# Throws 
- `DimensionMismatch` if dimensions of arguments are not compatible for
    multiplication.
"""
function left_mul_dimcheck(
    C::AbstractArray, 
    S::Union{CompressorRecipe, ApproximatorRecipe},
    A::AbstractArray
)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    c_rows, c_cols = size(C, 1), size(C, 2)
    if a_rows != s_cols
        throw(
            DimensionMismatch("Matrix A has $a_rows rows while S has $s_cols columns.")
        )
    elseif a_cols != c_cols
        throw(
            DimensionMismatch("Matrix A has $a_cols columns while C has $c_cols columns.")
        )
    elseif c_rows != s_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while S has $s_rows rows.")
        )
    end

    return nothing
end

"""
    left_mul_dimcheck(C::AbstractMatrix, S::CompressorAdjoint, A::AbstractMatrix)

$(check_method_description[:mul_check] * " from the left.")

# Arguments
- $(check_arg_list[:C])
- $(check_arg_list[:recipe_adjoint])
- $(check_arg_list[:A])

# Returns 
- `nothing`

# Throws 
- `DimensionMismatch` if dimensions of arguments are not compatible for
    multiplication.
"""
function left_mul_dimcheck(
    C::AbstractArray, 
    S::Union{CompressorAdjoint, ApproximatorAdjoint},
    A::AbstractArray
)
    # Checks S' * A -> C via A * S' -> C' 
    right_mul_dimcheck(transpose(C), transpose(A), S.parent)
    return nothing 
end

"""
    right_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix, S::CompressorRecipe)

$(check_method_description[:mul_check] * " from the right.")

# Arguments
- $(check_arg_list[:C])
- $(check_arg_list[:A])
- $(check_arg_list[:recipe])

# Returns 
- `nothing`

# Throws 
- `DimensionMismatch` if dimensions of arguments are not compatible for
    multiplication.
"""
function right_mul_dimcheck(
    C::AbstractArray, 
    A::AbstractArray, 
    S::Union{CompressorRecipe, ApproximatorRecipe}
)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    c_rows, c_cols = size(C, 1), size(C, 2)
    if a_cols != s_rows
        throw(
            DimensionMismatch("Matrix A has $a_cols columns while S has $s_rows rows.")
        )
    elseif c_cols != s_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while S has $s_cols columns.")
        )
    elseif c_rows != a_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows.")
        )
    end

    return nothing
end

"""
    right_mul_dimcheck(C::AbstractMatrix, A::AbstractMatrix, S::CompressorAdjoint)

$(check_method_description[:mul_check] * " from the right.")

# Arguments
- $(check_arg_list[:C])
- $(check_arg_list[:A])
- $(check_arg_list[:recipe_adjoint])

# Returns 
- `nothing`

# Throws 
- `DimensionMismatch` if dimensions of arguments are not compatible for
    multiplication.
"""
function right_mul_dimcheck(
    C::AbstractArray,
    A::AbstractArray,
    S::Union{CompressorAdjoint, ApproximatorAdjoint}
)
    left_mul_dimcheck(transpose(C), S.parent, transpose(A))
    return nothing 
end 
