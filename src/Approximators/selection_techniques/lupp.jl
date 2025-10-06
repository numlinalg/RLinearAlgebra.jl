"""
    LUPP <: Selector

A `Selector` that implements LU with partial pivoting for selecting column indices from a 
matrix.

# Fields
- None
"""
mutable struct LUPP <: Selector

end

"""
    LUPPRecipe <: SelectorRecipe

A `SelectorRecipe` that contains all the necessary preallocations for selecting column 
indices from a matrix using LU with partial pivoting.

# Fields
- None
"""
mutable struct LUPPRecipe <: SelectorRecipe

end

function complete_selector(ingredients::LUPP)
    return LUPPRecipe()
end

function update_selector!(selector::LUPPRecipe)
    return nothing
end

function select_indices!(
    idx::AbstractVector,
    selector::LUPPRecipe, 
    A::AbstractMatrix,
    n_idx::Int64, 
    start_idx::Int64
)
    if n_idx > size(A, 2)
        throw(
            DimensionMismatch( 
                "`n_idx` cannot be larger than the number of columns in `A`."
            )
        )
    end

    if start_idx + n_idx - 1 > size(idx, 1)
        throw(
            DimensionMismatch( 
                "`start_idx` + `n_idx` - 1 cannot be larger than the lenght of `idx`."
            )   
        )
    end
    
    # because LUPP selects rows and selectors select columns we need to pivot on A'
    p = lu!(A').p
    idx[start_idx:start_idx + n_idx - 1] = p[1:n_idx]
    return nothing
end
