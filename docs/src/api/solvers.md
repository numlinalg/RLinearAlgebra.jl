# Solvers  
```@contents
Pages = ["solvers.md"]
```

## Abstract Types
```@docs
Solver

SolverRecipe
```

## Solver Structures
```@docs
ColumnProjection

ColumnProjectionRecipe

IHS

IHSRecipe

Kaczmarz

KaczmarzRecipe
```

## Exported Functions
```@docs
complete_solver

rsolve!
```

## Internal Functions
```@docs
RLinearAlgebra.colproj_update!

RLinearAlgebra.colproj_update_block!

RLinearAlgebra.kaczmarz_update!

RLinearAlgebra.kaczmarz_update_block!

RLinearAlgebra.dotu
```
