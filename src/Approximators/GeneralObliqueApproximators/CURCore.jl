"""
    CURCore

An abstract type for the computation of the core linking matrix in a CUR decomposition.
"""
abstract type CURCore end

"""
    CURCoreRecipe

An abstract type for the recipes containing the preallocated information needed for 
the computation of the core linking matrix in a CUR decomposition.
"""
abstract type CURCoreRecipe end

"""
    CURCoreAdjoint{S<:CURCoreRecipe} <: CURCoreRecipe 

A structure for the adjoint of an `CURCoreRecipe`.

# Fields

  - `Parent::CURCoreRecipe`, the approximator that we compute the adjoint of.
"""
struct CURCoreAdjoint{S<:CURCoreRecipe} <: CURCoreRecipe
    parent::S
end

adjoint(A::CURCoreRecipe) = CURCoreAdjoint(A)
# Undo the transpose
adjoint(A::CURCoreAdjoint{<:CURCoreRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::CURCoreRecipe) = CURCoreAdjoint(A)
# Undo the transpose wrapper
transpose(A::CURCoreAdjoint{<:CURCoreRecipe}) = A.parent

###################################
# Size of CURCore 
###################################
function Base.size(S::CURCoreRecipe)
    return S.n_rows, S.n_cols
end

function Base.size(S::CURCoreRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.n_rows : S.n_cols
end

function Base.size(S::CURCoreAdjoint)
    return S.parent.n_cols, S.parent.n_rows
end

function Base.size(S::CURCoreAdjoint, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? S.parent.n_cols : S.parent.n_rows
end

