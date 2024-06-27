#
#
#
#
"""
  LinSysVecBlockGaussSampler <: LinSysVecRowSampler

An immutable structure with one field for the number of blocks to sample.
"""

struct LinSysVecBlockGaussSampler <: LinSysVecRowSampler
  blocksize::Int64
end

function sample(
  type::LinSysVecBlockGaussSampler,
  A::AbstractArray,
  b::AbstractVector,
  x::AbstractVector,
  iter::Int64
)
  S = randn(type.blocksize,length(b))
  SA = S*A
  return S, SA, SA*x - S*b  
end