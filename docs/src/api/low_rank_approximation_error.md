# Low Rank Approximation Error Methods

```@contents
Pages=["low_rank_approximation_error.md"]
```

## Abstract Types

```@docs
ApproxErrorMethod

RangeFinderError

```

## Approximate Function

```@docs
RLinearAlgebra.error_approximate!(::T where T<: ApproxErrorMethod, ::ApproxMethod, ::AbstractMatrix)
```
