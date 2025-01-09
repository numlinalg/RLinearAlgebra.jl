# Linear Solver Helpers 

```@contents
Pages=["linear_solver_helpers.md"]
```

## Gentleman's Solver for Least Squares

### Abstract Types

```@docs
RLinearAlgebra.GentData
```


### Solver
```@docs
RLinearAlgebra.gentleman!
RLinearAlgebra.ldiv!
RLinearAlgebra.copy_block_from_mat!
RLinearAlgebra.reset_gent!
```

## Krylov Solvers

### Helpers
```@docs
RLinearAlgebra.mgs!
RLinearAlgebra.arnoldi
RLinearAlgebra.randomized_arnoldi
```
