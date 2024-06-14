# Development Checklists 

```@contents
Pages=["checklists.md"]
```
The purpose of this page is to maintain checklists of tasks to complete when adding new methods to the library. These checklists are organized by method.

## Samplers
If you are implementing a sketching method for the library, make sure you have completed the following steps before making a pull request. There is a checklist for row and column samplers separately. If the method can be applied to both rows and columns then both checklists should be completed.

### Row Sampler
```
1. Implementation
- [] Create a file in the directory `linear_samplers`.
- [] Create a mutable struct for the sampling method.
- [] Create a constructor with default values for your struct.
- [] Create a function `sample` (a template can be found in src/linear_samplers.jl):
        INPUTS: 
        1. [] `type`, your mutable struct
        2. [] `A`, an `AbstractMatrix`
        3. [] `b`, an `AbstractVector`
        4. [] `x`, an `AbstractVector` 
        5. []  `iter` an `Int64`
        OUTPUTS:
        1. [] `S` the sampling matrix (Note: if you are only sampling row indices this should be a vector of those indicies.)
        2. [] `SA` the sampling matrix applied to A from the left.
        3. [] `res` the sketched residual `SA * x - S * b`
- [] Add an include("linear_samplers/[YOURFILE]") under the heading that best fits you method in src/linear_samplers.jl.
- [] Add import statements to src/RLinearAlgebra.jl with any functions from other packages that you use.
- [] Add your structure to the list of exported row samplers in src/RLinearAlgebra.jl.
- [] Add your structure to docs/src/api/linear_samplers.md under the appropiate heading. 
- [] Add a procedural test to test/linear_samplers.
2. Pull request
- [] Give a specific title to pull request.
- [] Lay out the features added to the pull request.
- [] Tag two people to review your pull request.
```

### Column Sampler
```
1. Implementation
- [] Create a file in the directory `linear_samplers`.
- [] Create a mutable struct for the sampling method.
- [] Create a constructor with default values for your struct.
- [] Create a function `sample` (a template can be found in src/linear_samplers.jl):
        INPUTS: 
        1. [] `type`, your mutable struct
        2. [] `A`, an `AbstractMatrix`
        3. [] `b`, an `AbstractVector`
        4. [] `x`, an `AbstractVector` 
        5. []  `iter` an `Int64`
        OUTPUTS:
        1. [] `S` the sampling matrix (Note: if you are only sampling column indices this should be a vector of those indicies.)
        2. [] `AS` the sampling matrix applied to A from the right.
        3. [] `res` the full residual `A * x - b` (Note the difference from the row methods).
        4. [] `grad` the sketched residual `SA'res`.
- [] Add an include("linear_samplers/[YOURFILE]") under the heading that best fits you method in src/linear_samplers.jl.
- [] Add import statements to src/RLinearAlgebra.jl with any functions from other packages that you use.
- [] Add your structure to the list of exported row samplers in src/RLinearAlgebra.jl.
- [] Add your structure to docs/src/api/linear_samplers.md under the appropiate heading. 
- [] Add a procedural test to test/linear_samplers.
2. Pull request
- [] Give a specific title to pull request.
- [] Lay out the features added to the pull request.
- [] Tag two people to review your pull request.
```

## Linear Solver Routines
Here we a checklist to make new linear solvers routines. These should be routines that generate a solution or an update to a solution
of a linear system using the sketched matrix information. They can be row sketching based, column sketching based, or both.
```
1. Implementation
- [] Create a file in the directory `linear_solver_routines`.
- [] Create a mutable struct for the solver.
- [] Create a constructor with default values for your struct.
- [] Create a function `rsubsolve!` (a template can be found in src/linear_solver_routines.jl):
        INPUTS: 
        1. [] `type`, your mutable struct
        2. [] `x`, an `AbstractVector` 
        3. [] `samp`, a tuple whose values differ between row and column routines.
            - [] Row routines should have `Tuple{U,V,W} where {U<:Union{AbstractVector,AbstractMatrix},
            V<:AbstractArray,W<:AbstractVector}`.
            - Column routines should have samp::Tuple{U,V,W,X} where `{U<:Union{AbstractVector,AbstractMatrix},
            V<:AbstractArray,W<:AbstractVector,X<:AbstractVector}`
        4. []  `iter` an `Int64`
        OUTPUTS:
        1. [] Should update the value of `x` but not return anything.
- [] Add an include("linear_solver_routines/[YOURFILE]") under the heading that best fits you method in src/linear_solver_routines.jl.
- [] Add import statements to src/RLinearAlgebra.jl with any functions from other packages that you use.
- [] Add your structure to the list of exported linear solver routines in src/RLinearAlgebra.jl.
- [] Add your structure to docs/src/api/linear_solver_routines.md under the appropiate heading. 
- [] Add a procedural test to test/linear_solver_routines.
2. Pull request
- [] Give a specific title to pull request.
- [] Lay out the features added to the pull request.
- [] Tag two people to review your pull request.
```

## Linear Solver Logs
Here we include checklist for implementations of techniques for logging the progress of linear solvers.
 As an example these would be routines that log the residual of a linear solver.
```
1. Implementation
- [] Create a file in the directory `linear_solver_logs`.
- [] Create a mutable struct for the logger.
- [] Create a constructor with default values for your struct.
- [] Create a function `log_update!` (a template can be found in src/linear_solver_logs.jl):
        INPUTS: 
        1. [] `log`, your mutable struct
        2. [] `sampler`, a sampling datastructure. 
        3. [] `x` an abstract vector
        3. [] `samp`, a tuple whose values differ between row and column routines.
            - [] Row routines should have `Tuple{U,V,W} where {U<:Union{AbstractVector,AbstractMatrix},
            V<:AbstractArray,W<:AbstractVector}`.
            - Column routines should have samp::Tuple{U,V,W,X} where `{U<:Union{AbstractVector,AbstractMatrix},
            V<:AbstractArray,W<:AbstractVector,X<:AbstractVector}`
        5. []  `iter` an `Int64`
        6. [] `A` an `AbstractArray`
        7. [] `b` an `AbstractVector`
        OUTPUTS:
        1. [] Should update your logging data structure.
- [] Add an include("linear_solver_logs/[YOURFILE]") under the heading that best fits you method in src/linear_solver_logs.jl.
- [] Add import statements to src/RLinearAlgebra.jl with any functions from other packages that you use.
- [] Add your structure to the list of exported linear solver logs in src/RLinearAlgebra.jl.
- [] Add your structure to docs/src/api/linear_solver_logs.md under the appropiate heading. 
- [] Add a procedural test to test/linear_solver_logs.
2. Pull request
- [] Give a specific title to pull request.
- [] Lay out the features added to the pull request.
- [] Tag two people to review your pull request.
```

## Linear Solver Stops
Here we include checklist for implementations of techniques for stopping conditions of the linear solvers. 
This can be routines that stop the solver after a number of iterations or some log related information.
```
1. Implementation
- [] Create a file in the directory `linear_solver_stops`.
- [] Create a mutable struct for the stopping method.
- [] Create a constructor with default values for your struct.
- [] Create a function `check_stop_critertion!` (a template can be found in src/linear_solver_stops.jl):
        INPUTS: 
        1. [] `log`, a generic log structure
        7. [] `stop` your stopping structure
        OUTPUTS:
        1. [] A boolean value indicating whether or not to stop. With `true` indicating that stopping 
        should occur. 
- [] Add an include("linear_solver_stops/[YOURFILE]") under the heading that best fits you method in src/linear_solver_stops.jl.
- [] Add import statements to src/RLinearAlgebra.jl with any functions from other packages that you use.
- [] Add your structure to the list of exported linear solver stopping criteria in src/RLinearAlgebra.jl.
- [] Add your structure to docs/src/api/linear_solver_stops.md under the appropiate heading. 
- [] Add a procedural test to test/linear_solver_stops.
2. Pull request
- [] Give a specific title to pull request.
- [] Lay out the features added to the pull request.
- [] Tag two people to review your pull request.
```
