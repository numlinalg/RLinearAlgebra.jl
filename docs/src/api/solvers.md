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
col_projection

col_projectionRecipe

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
RLinearAlgebra.col_proj_update!

RLinearAlgebra.col_proj_update_block!

RLinearAlgebra.kaczmarz_update!

RLinearAlgebra.kaczmarz_update_block!

RLinearAlgebra.dotu
```
