# Linear Systems
The other task that one may wish to accomplish with randomized methods is solving linear
systems. Given a linear system consisting of a matrix ``A \in \mathbb{R}^{m \times n}`` and
vector ``b \in \mathbb{R}^m`` we aim to use randomized methods to find to find the 
``x^* \in \mathbb{R}^{n}`` satisfying one of two key forms:
(1) the consistent linear system
``
    Ax^* = b
``
or 
(2) the least squares solution
``
x^* = \min_x \|Ax - b\|.
``
Most of the methods in randomized linear algebra are designed for finding solutions to 
problems of the form of (2); however, there are a few key approaches that solve systems 
of the form of (1). 

RLinearAlgebra.jl allows you to solve systems of the form of (1) or (2) simply by calling
the function `rsolve!(solver, x, A, b)`. Here `A` is the matrix of the linear system, 
`b` is the constant vector, and `x` is the initialization of the solution vector that will
be updated with solution produced by the solver. The `solver` of type `Solver` allows you
to specify the parameters of the solver you wish to use on the linear system. For any 
specific solving method, the solver could consist of any of the following sub-structures:
1. `Compressor`, specified how we will compress the system.
2. `SubSolver`, many of these randomized approaches are iterative and require solving a 
    compressed linear system. The `SubSolver` allows the user to specify how they wish to 
    solve the compressed system.
3. `Logger`, because many of the approaches are iterative, it is typically desirable to keep 
    track of the progress of the solver. This is done through the logging routine. 
    Specifically, the `Logger` performs two key tasks: (1) it stores the information from 
    an error method in history vector and (2) it checks whether an inputted stopping 
    criterion is satisfied.
4. `ErrorMethod`, the iterative nature also makes it desirable to check the progress of a 
    solver. The `ErrorMethod` allows you to specify the technique that you wish to use to 
    determine a solver's progress.
Any particular solver in RLinearAlgebra.jl could have some to all of these quantities
check the [Solvers API](@ref) for more details.

# A projection solver example
Kaczmarz is one randomized approach for solving consistent linear systems with a nice 
geometric intepretation.

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
    logger = BasicLogger(max_it = 30)
);

# solve the system
rsolve!(solver, x, A, b)

# Check error of solution 
norm(x - x_sol)
```

As the default number of iterations for Kaczmarz is only three times the number of rows, 
we may want to run for more iterations. We can do this by specifying the `max_it` parameter
in the logger to be 100. Additionally, we may wish to use a `compression_dim` of 5. We do 
this in the following example.
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
    logger = BasicLogger(
        max_it = 100,
        threshold_info = 1e-1
         
    ),
    compressor = SparseSign(compression_dim = 5)
);

# solve the system
rsolve!(Kaczmarz(), x, A, b)

# Check error of solution 
norm(x - x_sol)
```

