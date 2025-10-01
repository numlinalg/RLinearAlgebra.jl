# Loggers checklist 
If you are implementing a logging method for the library, make sure you have completed 
the following steps before making a pull request. In the following guides, `BasicLogger`
method is used as an example.

```
## Implementation
1. Method's core codes (`src/Solvers/Loggers`):
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
2. Package structure cooperation (`src/Solvers/Loggers.jl`, `src/RLinearAlgebra.jl`, `src/refs.bib`):  
    - [ ] Add an include("Loggers/[YOURFILE]") at bottom of the page, `src/Solvers/Loggers.jl`.
    - [ ] Add import statements to `src/RLinearAlgebra.jl` with any functions from other packages that you use.
    - [ ] Export your `Logger`, `LoggerRecipe`, any structs and functions you needed for your logger method to work in `src/RLinearAlgebra.jl`.
    - [ ] Add your `Logger`, `LoggerRecipe`, needed structs and functions to `docs/src/api.loggers.md`, under the appropriate heading.
    - [ ] If there are any new-added references, please add in `src/refs.bib`.
3. Tests (`test/Solvers/Loggers`):
    - [ ] Add a procedural test to `test/Solvers/Loggers`. Be sure to check that the functions work as intended and all warnings/assertions are displayed. For example, `test/Solvers/Loggers/basic_logger.jl`.
    - [ ] After finish implementing, you can goes to the julia's package environment by type `]` in the julia command line and run `test` to test whether you can pass all the tests.

## Pull request
- [ ] Give a specific title to pull request.
- [ ] Lay out the features added to the pull request.
- [ ] Tag two people to review your pull request.
- [ ] **Optional**: If possible, please also add Copilot as a reviewer and choose to adopt its suggestions if reasonable.
```