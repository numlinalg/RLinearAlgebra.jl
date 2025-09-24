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
RLinearAlgebra.kaczmarz_update!

RLinearAlgebra.kaczmarz_update_block!

RLinearAlgebra.dotu
```
