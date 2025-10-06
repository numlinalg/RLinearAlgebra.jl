"""
    QRCP <: Selector

A `Selector` that implements QR with column norm pivoting for selecting column indices from  
a matrix.

# Fields
- None
"""
mutable struct QRCP <: Selector

end

"""
    QRCPRecipe <: SelectorRecipe

A `SelectorRecipe` that contains all the necessary preallocations for selecting column 
indices from a matrix using QR with column norm pivoting.

# Fields
- None
"""
mutable struct QRCPRecipe <: SelectorRecipe

end

function complete_selector(ingredients::QRCP)
    return QRCPRecipe()
end

function update_selector!(selector::QRCPRecipe)
    return nothing
end

function select_indices!(
    idx::AbstractVector,
    selector::QRCPRecipe, 
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
    
    p = qr!(A, ColumnNorm()).p
    # store newly selected indices at inputed index storage points 
    idx[start_idx:start_idx + n_idx - 1] = p[1:n_idx]
    return nothing
end
