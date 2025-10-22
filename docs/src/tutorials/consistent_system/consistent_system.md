# Solving a Consistent Linear System

This guide demonstrates how to use `RLinearAlgebra.jl` package to solve a 
**consistent linear system**—a system where at least one solution 
exists—expressed in the form:

$$Ax = b$$

We'll walk through setting up the problem, using a solver, and verifying the result.

---
## Problem setup and solve the system

First, let's define our linear system $Ax = b$.

To easily verify the accuracy of our solver, we'll construct a problem where the true 
solution, $x_{\text{true}}$, is known beforehand. We'll start by creating a random 
matrix $A$ and a known solution vector $x_{\text{true}}$. Then, we can generate the 
right-hand side vector $b$ by computing $b = Ax_{\text{true}}$.

The following Julia code imports the necessary libraries, sets up the dimensions, and 
creates $A$, $x_{\text{true}}$, and $b$. We also initialize a starting guess, `x_init`, 
for our iterative solver.

```julia
using LinearAlgebra
num_rows, num_cols = 1000, 20;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
x_init = zeros(Float64, num_cols);
b = A * x_true;
```

```@setup ConsistentExample
using LinearAlgebra
num_rows, num_cols = 1000, 20;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
x_init = zeros(Float64, num_cols);
b = A * x_true;
```

As simple as you can imagine, `RLinearAlgebra.jl` can solve this system in just a 
few lines of codes and high efficiency:

```@example ConsistentExample
using RLinearAlgebra
logger = BasicLogger(max_it = 300)
kaczmarz_solver = Kaczmarz(log = logger)
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
rsolve!(solver_recipe, x_init, A, b)

solution = x_init;
println("Solution to the system: \n", solution)
```
**Done! How simple it is!**


Let's check how close our calculated `solution` is to the known `x_true`. 
We can measure the accuracy by calculating the Euclidean norm of the difference 
between the two vectors. A small norm indicates that our solver found a good approximation.

```@example ConsistentExample
# Calculate the norm of the error
error_norm = norm(solution - x_true)
println(" - Norm of the error between the solution and x_true: ", error_norm)
```
As you can see, by using the modular Kaczmarz solver, we were able to configure a 
randomized block-based method and find a solution vector that is very close to 
the true solution. 


Let's break down the solver code line by line to understand what each part does.

---
## Codes breakdown

As shown in the code, we used the [`Kaczmarz` solver](@ref Kaczmarz). A key feature of 
**RLinearAlgebra.jl** is its modularity; you can customize the solver's behavior by passing
in different "component" objects for tasks, such as system compression, progress logging, 
and termination checks.

For this example, we kept it simple by only customizing the maximum iteration located 
in [`Logger`](@ref Logger) component. Let's break down each step.


### Configure the logger

We start with the simplest component: the [`Logger`](@ref Logger). The 
[`Logger`](@ref Logger) is 
responsible for tracking metrics (such as the error history) and telling the solver 
when to stop. For this guide, we use the default [`BasicLogger`](@ref BasicLogger) 
and configure 
it with a single stopping criterion: a maximum number of iterations.

```julia
# Configure the maximum iteration to be 300
logger = BasicLogger(max_it = 300)
```

### Create the solver

Before running the solver on our specific problem (`A, b, x_init`), we must prepare it 
using the  [`complete_solver`](@ref complete_solver) function. This function creates 
a [`KaczmarzRecipe`](@ref KaczmarzRecipe), which combines the solver 
configuration with the problem data.

Crucially, this "recipe" pre-allocates all necessary memory buffers, which is a 
key step for ensuring efficient and high-performance computation.

```julia
# Create the Kaczmarz solver object by passing in the ingredients
kaczmarz_solver = Kaczmarz(log = logger)
# Create the solver recipe by combining the solver and the problem data
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
```

### Solve the system using the solver

Finally, we call [`rsolve!`](@ref rsolve!) to execute the algorithm. The `!` at the end 
of the function name is a Julia convention indicating that the function will inplace 
update part of its arguments. In this case, `rsolve!` modifies `x_init` in-place, 
filling it with the final solution vector. The solver will iterate until a stopping 
criterion in the `logger` is met, i.e. iteration goes up to $300$.

```julia
# Run the inplace solver!
rsolve!(solver_recipe, x_init, A, b)

# The solution is now stored in the updated x_init vector
solution = x_init;
```











