# Deeper Dive: Modular Components

In the previous guide, we showed how to solve a consistent linear system in just a few 
lines of codes. That example used the default configurations of the 
[`Kaczmarz` solver](@ref Kaczmarz) solver, which is highly effective for many standard 
problems.

However, the true power of **RLinearAlgebra.jl** lies in its high degree of modularity 
and flexibility. You can fine-tune the solver's behavior by combining different 
ingradients, like cooking a fine dish, to tackle specific challenges, improve 
performance, or implement more complex algorithms.



We will follow the design philosophy of **RLinearAlgebra.jl** by composing different 
modules ([`Compressor`](@ref Compressor), [`Logger`](@ref Logger), [`Solver`](@ref Solver), 
etc.) to customize a solver and solve the same consistent linear system, 

$$Ax = b.$$


```@setup ConsistentExample
# Import relevant libraries
using RLinearAlgebra, Random, LinearAlgebra


# Define the dimensions of the linear system
num_rows, num_cols = 1000, 20

# Create the matrix A and a known true solution x_true
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);

# Calculate the right-hand side vector b from A and x_true
b = A * x_true;

# Set an initial guess for the solution vector x (typically a zero vector)
x_init = zeros(Float64, num_cols);
```


---
## 1. Configure [`Compressor`](@ref Compressor)

For large-scale problems, the matrix $A$ can be massive, slowing down iterative algorithms. 
We can use a randomized sketching technique to "compress" $A$ and $b$ to a lower dimension 
while preserving the essential information of the system, s.t. we can solve the system 
fast without the loss of accuracy.

Here, we'll configure a [`SparseSign`](@ref SparseSign) compressor as an example. 
This compressor generates a sparse matrix $S$, whose non-zero elements are +1 or -1 
(with scaling). 

### (a) Configure the [`SparseSign`](@ref SparseSign) Compressor

We will configure a compression matrix that reduces the 1000 rows of our original 
system down to a more manageable 300 rows.

```@example ConsistentExample
# The goal is to compress the 1000 rows of A to 300 rows
compression_dim = 300
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

If the compression dimension of `300` rows is considered too large, it can be changed to `10` by updating the compressor configuration and rebuilding the recipe as follows:

```@example ConsistentExample
# Change the dimension of the compressor. Similarly, you can use the same idea 
# for other configurations' changes.
sparse_compressor.compression_dim = 10
```

The `sparse_compressor` is containing all the [`SparseSign`](@ref SparseSign) 
configurations that we need. 

While the solver can use the `sparse_compressor` to perform the compression method 
on-the-fly, we can stop here to configure other "ingradients". However, 
it can sometimes be useful to form the compression matrix and the compressed 
system explicitly to get an idea of your compression matrix. Therefore, we will 
continue playing with it.

---
### (b) Build the [`SparseSignRecipe`](@ref SparseSignRecipe)

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
println(" - Compression matrix: ",  S.op)
```

### (c) Apply the sparse sign matrix to the system

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
## 2. Configure [`Logger`](@ref Logger)

To monitor the solver and control its execution, we will configure a 
[`BasicLogger`](@ref BasicLogger) . 
This object serves two purposes: tracking metrics (like the error history) and 
defining stopping rules.

We'll configure it to stop after a maximum of `50` iterations or if the residual 
error drops below `1e-6`. We will also set `collection_rate = 5` to record the 
error every $5$ iterations.

```@example ConsistentExample
# Configure the logger to control the solver's execution
logger = BasicLogger(
    max_it = 50,
    threshold = 1e-6,
    collection_rate = 5
)
```

---
## 3. Configure [`Solver`](@ref Solver)

Now we assemble our configured ingradients—the `sparse_compressor` and the `logger`—into 
the main [`Kaczmarz` solver](@ref Kaczmarz) object. For any component we don't specify, 
a default will be used. Here, we'll explicitly specify the [LQSolver](@ref LQSolver) 
as our sub-solver.

### (a) Configure the [`Kaczmarz` solver](@ref Kaczmarz)
```@example ConsistentExample
# Create the Kaczmarz solver object by passing in the ingredients
kaczmarz_solver = Kaczmarz(
    compressor = sparse_compressor,
    log = logger,
    sub_solver = LQSolver()
)
```

### (b) Create the solver recipe

Just as with the compressor, we must call [`complete_solver`](@ref complete_solver) to 
create a final "recipe". This function takes the solver configuration and 
the specific problem data (`A, b, x_init`) and pre-allocates all memory needed 
for an efficient run.

```@example ConsistentExample
# Create the solver recipe by combining the solver and the problem data
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
```



---
## 4. Solve and Verify the Result

### (a) Solve the System

We call [`rsolve!`](@ref rsolve!) to run the [`Kaczmarz` solver](@ref Kaczmarz). 
The `!` in the name indicates that the function modifies its arguments in-place. 
Here, `x_init` will be updated with the solution vector. 
The algorithm will run until a stopping criterion from our 
`logger` is met.

```@example ConsistentExample
# Run the solver!
rsolve!(solver_recipe, x_init, A, b)

# The solution is now stored in the updated x_init vector
solution = x_init;
```

### (b). Verify the result

Finally, let's check how close our calculated solution is to the known 
`x_true` and inspect the `logger` to see how the solver performed.

```@example ConsistentExample
# We can inspect the logger's history to see the convergence
error_history = solver_recipe.log.hist;
println(" - Solver stopped at iteration: ", solver_recipe.log.iteration)
println(" - Final residual error, ||Ax-b||_2: ", error_history[solver_recipe.log.record_location])

# Calculate the norm of the error
error_norm = norm(solution - x_true)
println(" - Norm of the error between the solution and x_true: ", error_norm)
```

As you can see, by composing different modules, we configured a randomized 
solver that found a solution vector very close to the true solution.