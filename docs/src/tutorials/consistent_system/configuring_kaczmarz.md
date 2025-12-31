# Configuring a Generalized Kaczmarz Solver

In the previous guide, we showed how to solve a consistent linear system in just a few 
lines of code. That example used the default configurations of the 
[`Kaczmarz` solver](@ref Kaczmarz) solver [patel2023randomized](@cite).
However, the true power of **RLinearAlgebra.jl** lies in its high degree of modularity 
and flexibility. You can fine-tune the solver's behavior by combining different 
ingredients, like cooking a fine dish, to tackle specific challenges, improve 
performance, or implement more complex algorithms.

We will follow the design philosophy of **RLinearAlgebra.jl** by composing different 
modules ([`Compressor`](@ref Compressor), [`Logger`](@ref Logger), [`Solver`](@ref Solver), 
etc.) to customize a solver and solve the same consistent linear system, 

$$Ax = b.$$


```@setup ConsistentExample
# Import relevant libraries
using RLinearAlgebra, LinearAlgebra


# Define the dimensions of the linear system
num_rows, num_cols = 1000, 50

# Create the matrix A and a known true solution x_true
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);

# Calculate the right-hand side vector b from A and x_true
b = A * x_true;
```


---
## Configure [`Compressor`](@ref Compressor)

For large-scale problems, the matrix $A$ can be massive, slowing down iterative algorithms. 
We can use a randomized sketching technique to "compress" $A$ and $b$ to a lower dimension 
while preserving the essential information of the system, s.t. we can solve the system 
fast without the loss of accuracy.

Here, we'll configure a [`SparseSign`](@ref SparseSign) compressor as an example. 
This compressor generates a sparse matrix $S$, whose non-zero elements are +1 or -1 
(with scaling). 

### Configure the [`SparseSign`](@ref SparseSign) Compressor

We will configure a compression matrix that reduces the 1000 rows of our original 
system down to a more manageable $30$ rows.

```@example ConsistentExample
# The goal is to compress the 1000 rows of A to 30 rows
compression_dim = 30
# We want each row of the compression matrix S to have 5 non-zero elements
non_zeros = 5

# Configure the SparseSign compressor
sparse_compressor = SparseSign(
    cardinality=Left(),                 # S will be left-multiplied: SAx = Sb
    compression_dim=compression_dim,    # The compressed dimension (number of rows)
    nnz=non_zeros,                      # The number of non-zero elements per row in S
    type=Float64                        # The element type for the compression matrix
)
```

If the compression dimension of `30` rows is considered too large, it can be changed to `10` by updating the compressor configuration and rebuilding the recipe as follows:

```julia
# Change the dimension of the compressor. Similarly, you can use the same idea 
# for other configurations' changes.
sparse_compressor.compression_dim = 10;
```

```@setup ConsistentExample
sparse_compressor.compression_dim = 10;
```

The `sparse_compressor` is containing all the [`SparseSign`](@ref SparseSign) 
configurations that we need. 

---
### (Optional) Build [SparseSignRecipe](@ref SparseSignRecipe) and apply it to the system

While the solver can use the `sparse_compressor` to perform the compression method 
on-the-fly, we can stop here to configure other "ingredients". However, 
it can sometimes be useful to form the compression matrix and the compressed 
system explicitly to get an idea of your compression matrix. Therefore, we will 
continue playing with it.

After defining the compressor's parameters, we combine it with our matrix $A$ to 
create a [`SparseSignRecipe`](@ref SparseSignRecipe). This "recipe" 
pre-calculates the sparse matrix `S` and prepares everything needed for 
efficient compression.

```@example ConsistentExample
# Pass the compressor configuration and the original matrix A to
# create the final compression recipe.
S = complete_compressor(sparse_compressor, A)

# You can closely look at the compression recipe you created.
println("Configurations of compression matrix:")
println(" - Compression matrix is applied to left or right: ", S.cardinality)
println(" - Compression matrix's number of rows: ", S.n_rows)
println(" - Compression matrix's number of columns: ",  S.n_cols)
println(" - The number of nonzeros in each column (left)/row (right) of compression matrix: ",  S.nnz)
println(" - Compression matrix's nonzero entry values: ",  S.scale)
println(" - Compression matrix: ",  typeof(S.op), size(S.op))
```
We can use `*` to apply this sparse matrix `S` to the system.

```@example ConsistentExample
# Form the compressed system SAx = Sb
SA = S * A
Sb = S * b

println("Dimensions of the compressed system:")
println(" - Matrix SA: ", size(SA))
println(" - Vector Sb: ", size(Sb))
```


---
## Configure [`Logger`](@ref Logger)

To monitor the solver and control its execution, we will configure a 
[`BasicLogger`](@ref BasicLogger) . 
This object serves two purposes: tracking metrics (like the error history) and 
defining stopping rules.

We'll configure it to stop after a maximum of `500` iterations or if the residual 
error drops below `1e-6`. We will also set `collection_rate = 5` to record the 
error every $5$ iterations.

```@example ConsistentExample
# Configure the logger to control the solver's execution
logger = BasicLogger(
    max_it = 500,
    threshold = 1e-6,
    collection_rate = 5
)
```

### (Optional) Build the [BasicLoggerRecipe](@ref BasicLoggerRecipe)

Just like the compressor, this `logger` is just a set of instructions. 
To make it "ready" to store data, we could call [complete_logger](@ref complete_logger) 
on it:


```@example ConsistentExample
# We can create the recipe manually, though this is rarely needed
logger_recipe = complete_logger(logger)
```
This `logger_recipe` is the object that actually contains the hist vector for 
storing the error history, the current iteration, and the converged status.

Again, you almost never need to call `complete_logger` yourself. 
Because the solver (which we will configure next) or the [`rsolve!`](@ref rsolve!)
(the function that solves the system) 
handles it for us. When we call [`complete_solver`](@ref complete_solver) or the 
[`rsolve!`](@ref rsolve!), it will automatically find the 
`logger` inside the solver, call `complete_logger` on it, 
and store the resulting `logger_recipe` inside the final `solver_recipe`.

---




## Configure [`Solver`](@ref Solver)

Now we assemble our configured ingredients—the `sparse_compressor` and the `logger`—into 
the main [`Kaczmarz` solver](@ref Kaczmarz) object. For any component we don't specify, 
a default will be used. Here, we'll explicitly specify the [LQSolver](@ref LQSolver) 
as our sub-solver.

### Configure the [`Kaczmarz` solver](@ref Kaczmarz)
```@example ConsistentExample
# Create the Kaczmarz solver object by passing in the ingredients
kaczmarz_solver = Kaczmarz(
    compressor = sparse_compressor,
    log = logger,
    sub_solver = LQSolver()
)
```

### (Optional) create the [SolverRecipe](@ref SolverRecipe)

This is the step where everything comes together. 
Just as with the compressor, when we call 
[`complete_solver`](@ref complete_solver), it takes the
`kaczmarz_solver` configurations and the problem data, and then:

1. Pre-allocates memory for everything needed in the algorithm.

2. Finds the `sparse_compressor` config inside `kaczmarz_solver` and calls `complete_compressor` to create the `SparseSignRecipe`.

3. Finds the `logger` inside `kaczmarz_solver` and calls `complete_logger` to create the `BasicLoggerRecipe`.

4. Bundles all these "recipes" into a single, ready-to-use solver_recipe object.

```@example ConsistentExample
# Set the solution vector x (typically a zero vector)
solution = zeros(Float64, num_cols);
# Create the solver recipe by combining the solver and the problem data
solver_recipe = complete_solver(kaczmarz_solver, solution, A, b);
```
However, again, this step can also be skipped and directly pass the 
solver config `kaczmarz_solver` into the function that can solve the system.



---
## Solve and Verify the Result

### Solve the System

We call [`rsolve!`](@ref rsolve!) to run the [`Kaczmarz` solver](@ref Kaczmarz). 
The `!` in the name indicates that the function modifies its arguments in-place. 
Here, `solution` will be updated with the solution vector. 
The algorithm will run until a stopping criterion from our 
`logger` is met.

```@example ConsistentExample
# Set the solution vector x (typically a zero vector)
solution = zeros(Float64, num_cols);

# Run the solver!
_, solver_history = rsolve!(kaczmarz_solver, solution, A, b);

# The solution is now stored in the updated solution vector
solution
```

### Verify the result

Finally, let's check how close our calculated solution is to the known 
`x_true` and inspect the `logger` to see how the solver performed.

```@example ConsistentExample
# We can inspect the logger's history to see the convergence
error_history = solver_history.log.hist;
println(" - Solver stopped at iteration: ", solver_history.log.iteration)
println(" - Final residual error, ||Ax-b||_2: ", error_history[solver_history.log.record_location])

# Calculate the norm of the error
error_norm = norm(solution - x_true)
println(" - Norm of the error between the solution and x_true: ", error_norm)
```

As you can see, by composing different modules, we configured a randomized 
solver that found a solution vector very close to the true solution.