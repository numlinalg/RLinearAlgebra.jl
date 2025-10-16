# Solving a Consistent Linear System

This guide demonstrates how to use `RLinearAlgebra.jl` package to find the solution to 
a **consistent linear system** of the form:

$$Ax = b$$

---
## Problem Setup and solve the system

Let's define a specific linear system $Ax = b$. 

To verify the accuracy of the final result, suppose that we know the true solution 
to the system, $x_{\text{true}}$, and then use it and a random generated 
matrix $A$ to generate the vector $b$.

To achieve this, we need to import the required libraries and create the matrix `A` 
and vector `b` as defined above. 
We will also set an initial guess, `x_init`, for the solver.

```julia
using RLinearAlgebra, LinearAlgebra
num_rows, num_cols = 1000, 20;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
x_init = zeros(Float64, num_cols);
b = A * x_true;
```

```@setup ConsistentExample
using RLinearAlgebra, LinearAlgebra
num_rows, num_cols = 1000, 20;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
x_init = zeros(Float64, num_cols);
b = A * x_true;
```

As simple as you can imagine, `RLinearAlgebra.jl` can solve this system in just a 
few lines of codes and high efficiency:

```@example ConsistentExample
logger = BasicLogger(max_it = 10)
kaczmarz_solver = Kaczmarz(log = logger)
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
rsolve!(solver_recipe, x_init, A, b)

solution = x_init;
println("Solution to the system: \n", solution)
```
Done! How simple it is!

let's check how close our calculated solution is to the known `x_true`. 
We can do this by calculating the Euclidean norm of the difference between the two vectors. 
A small error norm indicates a successful approximation.

```@example ConsistentExample
# We can inspect the logger's history to see the convergence
error_history = solver_recipe.log.hist;
println(" - Solver stopped at iteration: ", solver_recipe.log.iteration)
println(" - Final error: ", error_history[solver_recipe.log.record_location])

# Calculate the norm of the error
error_norm = norm(solution - x_true)
println(" - Norm of the error between the solution and x_true: ", error_norm)
```

<!-- TODO: Why the err_history is different from error norm -->

As you can see, by using the modular Kaczmarz solver, we were able to configure a 
randomized block-based method and find a solution vector that is very close to 
the true solution. 

Let's go line by line to see what are the codes doing.

---
## Steps to solve the problem

Here, we choose to use the [Kaczmarz solver](@ref Kaczmarz) to solve the problem. 
We can configure it by passing in "ingredient" objects for each of its main functions:
 compressing the system, logging progress, and checking for errors.


### Configure the logger

Start with only the simplest component, let's configure just the maximum iteration that 
our algorithm can go. The configuration is located in the [`logger`](@ref Logger)
structure, which is responsible to record the error history, and tell the 
solver when to stop. 
Here, we will use the [`BasicLogger`](@ref BasicLogger).

```julia
# Configure the maximum iteration to be 500
logger = BasicLogger(max_it = 500)
```

### Create the solver

Now, we assemble our configured components (`logger`) into the main 
Kaczmarz solver object. 
We will use the default compressor, logger and sub-solver.

```julia
# Create the Kaczmarz solver object by passing in the ingredients
kaczmarz_solver = Kaczmarz(log = logger)
```

Before we can run the solver, we must call [`complete_solver`](@ref complete_solver). 
This function takes the solver configurations and the specific problem data `A, b, x_init` 
and creates a [`KaczmarzRecipe`](@ref KaczmarzRecipe). 
The recipe pre-allocates all the necessary memory buffers for efficient computation.

```julia
# Create the solver recipe by combining the solver and the problem data
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
```

With the recipe fully prepared, we can now call [`rsolve!`](@ref rsolve!) 
to run the Kaczmarz algorithm.
The function will iterate until the stopping criterion in the `logger` is met.

### Solve the system using the solver

The [`rsolve!`](@ref rsolve!) function will modify `x_init` in-place, updating 
it with the calculated solution.

```julia
# Run the solver!
rsolve!(solver_recipe, x_init, A, b)

# The solution is now stored in the updated x_init vector
solution = x_init;
```


