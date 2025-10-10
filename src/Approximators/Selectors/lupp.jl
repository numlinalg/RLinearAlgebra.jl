"""
    LUPP <: Selector

A `Selector` that implements LU with partial pivoting for selecting column indices from a 
matrix.

# Fields
- `compressor::Compressor`, the compression technique that will applied to the matrix, 
    before selecting indices.

# Constructor
    LUPP(;compressor = Identity())

## Keywords
- `compressor::Compressor`, the compression technique that will applied to the matrix, 
    before selecting indices. Defaults the `Identity` compressor.

## Returns
- Will return a `Selector` object.
"""
mutable struct LUPP <: Selector
    compressor::Compressor
end
 
function LUPP(;compressor = Identity()) 
    LUPP(compressor)
end

"""
    LUPPRecipe <: SelectorRecipe

A `SelectorRecipe` that contains all the necessary preallocations for selecting column 
indices from a matrix using LU with partial pivoting.

# Fields
- `compressor::Compressor`, the compression technique that will applied to the matrix, 
    before selecting indices.
- `SA::AbstractMatrix`, a buffer matrix for storing the sketched matrix.
"""
mutable struct LUPPRecipe <: SelectorRecipe
    compressor::CompressorRecipe
    SA::AbstractMatrix
end

function complete_selector(ingredients::LUPP, A::AbstractMatrix)
    compressor = complete_compressor(ingredients.compressor, A)
    n_rows, n_cols = size(compressor)
    SA = Matrix{eltype(A)}(undef, n_rows, n_cols)
    return LUPPRecipe(compressor, SA)
end

function update_selector!(selector::LUPPRecipe)
    update_compressor!(selector.compressor)
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

    if n_idx > selector.compressor.n_rows
        throw(
            DimensionMismatch( 
                "Must select fewer indices then the `compression_dim`."
            )   
        )

    end
    
    mul!(selector.SA, selector.compressor, A)
    # because LUPP selects rows and selectors select columns we need to pivot on A'
    p = lu!(selector.SA').p
    idx[start_idx:start_idx + n_idx - 1] = p[1:n_idx]
    return nothing
end
