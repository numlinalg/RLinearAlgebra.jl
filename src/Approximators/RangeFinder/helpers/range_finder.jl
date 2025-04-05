"""
    rand_power_it(A::AbstractMatrix, approx::RangeFinderRecipe)

Function that performs the randomized rangefinder procedure presented in Algortihm 4.3 of 
.

INPUTS:
- `A::AbstractMatrix`, the matrix being approximated.
- `approx::RangeFinderRecipe`, a `RangeFinderRecipe` structure that contains the compressor
recipe.

# OUTPUTS
- `Q::AbstractMatrix`, an economical `Q` approximating the range of A.
"""

function rand_power_it(A::AbstractMatrix, approx::RangeFinderRecipe)
    S = approx.compressor
    a_rows, a_cols = size(A)
    s_rows, s_cols = size(comp_mat)
    type = eltype(A)
    compressed_mat = Matrix{type}(undef, a_rows, s_cols)
    mul!(compressed_mat, A, S)
    if approx.power_its > 0
        # If we are running power iterations an extra matrix is need to store multiplication
        # output
        buff_mat = Matrix{type}(undef, a_cols, s_cols) 
        for i in 1:approx.power_its
            # Perform the power iterations (AA^\top)^power_its (AS)
            mul!(buff_mat, A', compressed_mat)
            mul!(compressed_mat, A, buff_mat)
        end
    
    end
    
    # Return the economical qr of the matrix Q
    return Array(qr!(compressed_mat).Q)    
end


"""
    rand_subspace_it(A::AbstractMatrix, approx::RangeFinderRecipe)

Function that performs the randomized rangefinder procedure presented in Algortihm 4.4 of 
.

INPUTS:
- `A::AbstractMatrix`, the matrix being approximated.
- `approx::RangeFinderRecipe`, a `RangeFinderRecipe` structure that contains the compressor
recipe.

# OUTPUTS
- `Q::AbstractMatrix`, an economical `Q` approximating the range of A.
"""
function rand_subspace_it(A::AbstractMatrix, approx::RangeFinderRecipe)
    S = approx.compressor
    a_rows, a_cols = size(A)
    s_rows, s_cols = size(comp_mat)
    type = eltype(A)
    compressed_mat = Matrix{type}(undef, a_rows, s_cols)
    mul!(compressed_mat, A, S)
    Q = Array(qr!(compressed_mat).Q)
    if approx.power_its > 0
        # If we are running power iterations an extra matrix is need to store multiplication
        # output
        buff_mat = Matrix{type}(undef, a_cols, s_cols) 
        for i in 1:approx.power_its
            # Perform the power iterations based on the recusion Q_{i'} = qr(A'Q_{i-1}).Q 
            # Q_i = qr(A*Q_{i'}).Q this helps limit rounding errors
            mul!(buff_mat, A', Q)
            Q = Array(qr!(buff_mat).Q) 
            mul!(compressed_mat, A, buff_mat)
            Q = Array(qr!(compressed_mat).Q)
        end
    
    end
    
    # Return the economical qr of the matrix Q
    return Q    
end
