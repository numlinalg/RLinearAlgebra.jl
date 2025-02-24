# Linear Solver Logs

```@contents
Pages = ["linear_solver_logs.md"]
```


## Abstract Types

```@docs
LinSysSolverLog
```

## Loggers

```@docs
LSLogOracle

LSLogFull

LSLogMA

LSLogFullMA
```

## Log Function

```@docs
RLinearAlgebra.log_update!(::Nothing,::LinSysSampler,::AbstractVector,::Tuple,::Int64,::Any,::Any)
```

## Log Specific Functions

```@docs
RLinearAlgebra.get_uncertainty(::LSLogMA; alpha = .05)
```

## Internal Data Structures

```@docs
RLinearAlgebra.MAInfo

RLinearAlgebra.SEDistInfo
```

## Internal Functions
```@docs
RLinearAlgebra.update_ma!

RLinearAlgebra.get_SE_constants!
```
