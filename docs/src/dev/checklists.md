# Development Checklists 

```@contents
Pages=["checklists.md"]
```
The purpose of this page is to maintain checklists of tasks to complete when adding new 
methods to the library. These checklists are organized by method.

## Compressors 
If you are implementing a compression method for the library, make sure you have completed 
the following steps before making a pull request. 

```
1. Implementation
- [ ] Create a file in the directory `src/Compressors`.
- [ ] Create a Compressor structure with n_rows and n_cols as well as other user-controlled 
parameters.
- [ ] Create a constructor with keyword default values for your struct.
- [ ] Create a CompressorRecipe structure that uses the parameters from the Compressor 
structure to preallocate memory.
- [ ] Create a `complete_compressor` function that takes as inputs of the Compressor, A, x,b
and returns a CompressorRecipe
- [ ] Create a `update_compressor!` function that generates new random values for a random
components of the Compressor 
- [ ] a 5 input `mul!` function for  applying a compressor to a matrix from the left
- [ ] a 5 input `mul!` function for  applying a compressor to a matrix from the right 
- [ ] a 5 input `mul!` function for  applying a compressor to a vector
- [ ] a 5 input `mul!` function for  applying the adjoint of the compressor to a vector
- [ ] Add an include("Compressors/[YOURFILE]") at bottom of page 
- [ ] Add import statements to src/RLinearAlgebra.jl with any functions from other packages 
that you use.
- [ ] Add your Compressor and CompressorRecipe to src/RLinearAlgebra.jl.
- [ ] Add your Compressor and CompressorRecipe to docs/src/api/compressors.md 
under the appropiate heading. 
- [ ] Add a procedural test to test/linear_samplers. Be sure to check that the functions 
work as intended and all warnings/assertions are displayed.
2. Pull request
- [ ] Give a specific title to pull request.
- [ ] Lay out the features added to the pull request.
- [ ] Tag two people to review your pull request.
```

## Loggers 
If you are implementing a logging method for the library, make sure you have completed 
the following steps before making a pull request. In the following guides, `BasicLogger`
is used as an example.


1. Implementation
- Method's core codes (`src/Solvers/Loggers`):
    - [ ] Create a file in the directory `src/Solvers/Loggers`. For example, `src/Solvers/Loggers/basic_logger.jl`.
    - [ ] Create a `Logger` struct with `max_it`, `collection_rate`, `threshold_info`, `stopping_criterion`, any other method-needed parameters, and argument validations to check invalid inputs. For example, `BasicLogger<:Logger`.
    - [ ] Create a constructor with keyword default values for your `Logger` struct. For example, 
    ```
        BasicLogger(;
            max_it = 0, 
            collection_rate = 1, 
            threshold = 0.0,
            stopping_criterion = threshold_stop 
        ) = BasicLogger(max_it, collection_rate, threshold, stopping_criterion)
    ```
    - [ ] Add documentation for this `Logger` struct, with mainly 5 parts: brief introduction, fields introduction, constructor and its keywords, what the constructor returns, and what the argument validations throw. For more details, you can check `src/Solvers/Loggers/basic_logger.jl`.
    - [ ] Create a `LoggerRecipe` struct that uses the parameters from the `Logger` struct to preallocate memory. For example, `BasicLoggerRecipe{F<:Function} <: LoggerRecipe`.
    - [ ] Create a `complete_logger` function that takes the `logger` struct as an input, and returns the `LoggerRecipe` struct you defined last step. For example, `complete_logger(logger::BasicLogger)`.
    - [ ] Add documentation for this `LoggerRecipe` struct, with mainly 2 parts: brief introduction, fields introduction. For more details, you can check `src/Solvers/Loggers/basic_logger.jl`.
    - [ ] Create a `update_logger!` function to log the errors as the iteration goes on, and stop the logging with convergence status or the maximum iteration limit. For example, 
    ```
        update_logger!(logger::BasicLoggerRecipe, error::Float64, iteration::Int64)
    ```
    - [ ] Create a `reset_logger!` function to clean the history log information after convergence or exceed the maximum iteration. For example, `reset_logger!(logger::BasicLoggerRecipe)`.
    - [ ] Create a `threshold_stop` function as the convergent stopping criterion designed for your `Logger` struct. For example, `threshold_stop(log::BasicLoggerRecipe)`
    - [ ] Add documentation for this `threshold_stop` function, with mainly 3 parts: brief introduction, arguments introduction, and returns. For more details, you can check `src/Solvers/Loggers/basic_logger.jl`.
    - [ ] **Optional**: If you have any helper functions that needed for your implementation, please implement them in a folder at `src/Solvers/Loggers`.
- Package structure cooperation (`src/Solvers/Loggers.jl`, `src/RLinearAlgebra.jl`, `src/refs.bib`):  
    - [ ] Add an include("Loggers/[YOURFILE]") at bottom of the page, `src/Solvers/Loggers.jl`.
    - [ ] Add import statements to `src/RLinearAlgebra.jl` with any functions from other packages that you use.
    - [ ] Export your `Logger`, `LoggerRecipe`, any structs and functions you needed for your logger method to work in `src/RLinearAlgebra.jl`.
    - [ ] Add your `Logger`, `LoggerRecipe`, needed structs and functions to `docs/src/api.loggers.md`, under the appropriate heading.
    - [ ] If there are any new-added references, please add in `src/refs.bib`.
- Tests (`test/Solvers/Loggers`):
    - [ ] Add a procedural test to `test/Solvers/Loggers`. Be sure to check that the functions work as intended and all warnings/assertions are displayed. For example, `test/Solvers/Loggers/basic_logger.jl`.
2. Pull request
-   [ ] Give a specific title to pull request.
- [ ] Lay out the features added to the pull request.
- [ ] Tag two people to review your pull request.
- [ ] **Optional**: If possible, please also add Copilot as a reviewer and choose to adopt its suggestions if reasonable.
