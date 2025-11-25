# Linear Systems
Another task that one may wish to accomplish with randomized methods is solving linear
systems. Given a linear system consisting of a matrix ``A \in \mathbb{R}^{m \times n}`` and
constant vector ``b \in \mathbb{R}^m``, we aim to use randomized methods to find an 
``x^* \in \mathbb{R}^{n}`` taking one of two key forms:
(1) the consistent linear system,
``
    Ax^* = b
``
or 
(2) the least squares solution,
``
x^* = \min_x \|Ax - b\|.
``
The majority of methods in randomized linear algebra are designed for finding solutions to 
problems of the form of (2); however, there are a few key approaches that solve systems 
of the form of (1). 

RLinearAlgebra.jl allows you to solve systems of the form of (1) or (2) simply by calling
the function `rsolve!(solver, x, A, b)`. Here `A` is the matrix of the linear system, 
`b` is the constant vector, and `x` is the initialization of the solution vector that will
be replaced with solution produced by the solver. The `solver` variable of type 
`Solver` allows you to specify the algorithmic parameters specific to the solver 
you wish to use to find ``x^*``. 

These algorithmic parameters may require any of the following sub-structures:
1. `Compressor`, specifies how we will compress the system.
2. `SubSolver`, allows you to specify how they wish to 
    solve the compressed system. Typically, randomized methods compress the linear system
    then solve the compressed system. This parameter allows for the use of iterative 
    methods like Krylov solvers to solve the linear system instead of direct factorizations.
4. `ErrorMethod`, allows you to specify the technique that you wish to use to 
    determine a solver's progress. For example, you may wish to compute the residual at 
    every iteration using the `FullResidual` structure.
3. `Logger`, performs two key tasks: (1) it stores the information from 
    an error method in a `hist` field and (2) it determines whether an inputted stopping 
    criterion is satisfied.
Any particular solver in RLinearAlgebra.jl could have any subset of these structures, 
check the [Solvers API](@ref) for more details.

# Kaczmarz for consistent systems example
Kaczmarz is one randomized approach for solving consistent linear systems with a nice 
geometric interpretation. We can understand this geometric interpretation by first 
recognizing that when a linear system has solution, ``x^*``,  
``A[i, :] x^* = b[i]`` for every row of the linear system. Geometrically this means that
``x^*`` lies at the intersection of all row hyperplanes. Kaczmarz is able to converge to 
this solution by projecting orthogonally from one hyperplane to the next. Performing such 
projects ensures that every update gets closer to the solution as can be seen in the 
following figure.

```@raw html
<img src="../images/kaczmarz.png" width =400 height = 300/> 
```

Despite the simplicity of this procedure, it can be shown that if the ordering of these 
projections is randomized, where each row is sampled from a distribution where row ``i`` is 
sampled with probability ``\| A[i,:]\|_2^2 / \|A\|_F^2``, then, in expectation, 
Kaczmarz will converge at geometric rate. 

In the following example, we run a Kaczmarz solver for 30 iterations. 
To do this appropriately, we set the `max_it` field in the `BasicLogger` structure to be the 
`log` field in the `Kaczmarz` solver.

```julia
using RLinearAlgebra
using LinearAlgebra

# Generate the linear system
A = rand(10, 10);
x_sol = rand(10);
b = A * x_sol;

# create initalization for solution
x = zeros(10);

# Create the Kaczmarz Solver
solver = Kaczmarz(
    log = BasicLogger(max_it = 30)
);

# solve the system
rsolve!(solver, x, A, b)

# Check error of solution 
norm(x - x_sol)
```

Often for Kaczmarz, we can improve the rate of convergence by projecting onto blocks of
rows. To do this for the Kaczmarz method we need to change the `compression_dim` of the 
compressor. In this example, we set the `compression_dim = 5`. We also may wish to 
stop when the residual falls below `1e-1`, which we do by specifying the `threshold` field
in the `Logger` structure. After setting up these systems, we run our solver and plot the
history of the `FullResidual` by plotting the `hist` field of `log` field of the returned 
`SolverRecipe`.
```julia
using RLinearAlgebra
using LinearAlgebra
using Plots

# Generate the linear system
A = rand(10, 10);
x_sol = rand(10);
b = A * x_sol;

# create initalization for solution
x = zeros(10);

# Create the Kaczmarz Solver
solver = Kaczmarz(
    log = BasicLogger(
        max_it = 100,
        threshold = 1e-1
         
    ),
    compressor = SparseSign(compression_dim = 5)
);

# solve the system
x, kaczmarz_recipe = rsolve!(solver, x, A, b)

# Check error of solution 
norm(x - x_sol)

# Plot the residual history
plot(
    kaczmarz_recipe.log.hist[kaczmarz_recipe.log.hist .> 0.0], 
    yscale = :log10, 
    xlab = "Iteration", 
    ylab = "Norm of Residual"
)
```

