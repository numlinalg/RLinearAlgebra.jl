# Approximators 
```@contents
Pages = ["approximators.md"]
```

## Abstract Types
```@docs
Approximator

ApproximatorRecipe

ApproximatorAdjoint

RLinearAlgebra.RangeApproximator

RangeApproximatorRecipe

CURCore

CURCoreRecipe

CURCoreAdjoint

```

## Range Approximator Structures
```@docs
RandSVD

RandSVDRecipe

RangeFinder

RangeFinderRecipe

```

## General Oblique Approximators
```@docs
CUR

CURRecipe
```
### CURCore Structures
```@docs
CrossApproximation

CrossApproximationRecipe
```

## Exported Functions
```@docs
complete_approximator

complete_core

update_core!

rapproximate

rapproximate!
```

## Internal Functions
```@docs
RLinearAlgebra.rand_power_it

RLinearAlgebra.rand_ortho_it
```
