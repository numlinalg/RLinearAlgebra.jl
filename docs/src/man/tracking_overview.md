# Tracking an Iterative Sketching Solver

When using an iterative sketching method like Randomized Block Kaczmarz
known the quality of the solution is integral for being able to make
stopping decisions. 
it is important to track these techniques in a manner that does 
not undermine the stated benefits of the method, namely being able 
update a solution using only a row or block of rows of the matrix.
Na\"ive methods of tracking like computing the norm of the residual,
$\|Ax-b\|_2^2$, directly undermine these benefits, by requiring access
to the full matrix to compute. In fact, when the row dimension of the 
underlying linear system is far greater than block size, computing 
the progress estimator can be substantially more expensive than computing
the update itself. These cost could be avoided by computing the block
residual i.e. $\|S A x - Sb\|_2^2$, but at the cost exact knowledge of 
progress. To reduce this randomness Pritchard and Patel propose using 
a moving average of the sketched residuals in ``Solving, Tracking 
and Stopping Streaming Linear Inverse Problems." Specifically they define
the progress estimator 
$$
    \hat \rho_k^\lambda = \sum_{i-\lambda +1}^k \frac{\|S_i A x - S_ib\|_2^2}{\lambda}
$$  
to estimate progress where $\lambda$ is the width of the moving average window. 
This technique can be used in RLinearAlgebra.jl by using the log option
`LSLogFullMA()` when defining the solver, with a default moving average of 30.
```julia
using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(),              # Full Logger: maintains moving average residual history
    LSStopMaxIterations(200),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
```
## The moving average
The user is able to choose their own width of the moving average by inputting 
`lambda2=USER_WIDTH` as an option within `LSLogFullMA()`. Increasing the width
will decrease the variability of the progress estimate and therefore is a suggested
action when the matrix is poorly conditioned or has highly variable magnitude of row 
norms. In most cases, the default option of 30 should be more than sufficient for 
tracking. 

It is also important to note that because of the geometric convergence of 
many of these randomized methods early iterations have residual convergence that 
arises from the convergence of the algorithm itself. Meaning that it is often 
undesirable to smooth this initial variability. This lead to a two phase implementation
of the moving average scheme where during the first phase a moving average of $\lambda_1$, 
typically set to be one, is used to judge progress. Then in a second phase, which is judge
to be the point where there is no longer monotonic decreases in the norm of the residual,
the moving average is expanded to a value of $\lambda_2$.

If you want to compare the performance of the moving average of the sketched residuals to 
that of the true residuals then it is possible to input the option `true_res=true`. This will 
perform the same tracking procedure, but use the true residual rather than the sketched one.

Finally, if one wanted to get uncertainty sets for the sketched residual tracking they can use
the function `get_uncertainty()`, where they input the history accessed from the solver 
and can specify `alpha`, which indicates the probability that the moving average of the 
true residuals falls within that interval. As an example if one wanted to track a standard row solver
 with a moving average of the sketched residuals of width 100 and get a 99% uncertainty sets
they could run the following code.  
```julia

using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(lambda_2 = 100),# Full Logger: maintains moving average residual history
    LSStopMaxIterations(200),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
bounds = get_uncertainty(sol.log, alpha = .99)
```
!!! Note
    If one is using a subset of identity type sampling these uncertainty sets will perform conservatively
    because of a poor variance estimate. You can reduce this conservativeness using the option $\eta$ in 
    the `LSLogFullMA()` settings for the solver.

## Stopping
In addition to being able to form the uncertainty sets, Pritchard and Patel also proposed a criterion for
stopping when using the sketched moving average estimator. RLinearAlgebra.jl allows for the specification 
these methods this can be done using `LSStopMA().` Tho understand the stopping criterion, it is valuable
to define some notation. If we allow $\rho_k^\lambda$ be the moving average of the true residuals, and 
$\hat \rho_k^\lambda$ be the moving average of the sketched residuals then two types of errors can occur.  
The first can be viewed as stopping too late, and it occurs when the tracking parameter value, 
$\rho_k^\lambda \leq \delta_I \upsilon$, while $\hat \rho_k^\lambda > \upsilon$, where $\delta_I$ 
is a user defined parameter that permits the specification of where the gap between $\hat \rho_k^\lambda$ 
and $\rho_k^\lambda$ is large enough to be considered problematic. By using the condition on 
$\hat{\iota}^\lambda_k$ (defined on line 13), we approximately control the probability of this error at 
$\xi_{I}$.

The second error type can be viewed as stopping too early, and it occurs when the tracking parameter value, 
$\rho_k^\lambda \geq \delta_{II} \upsilon$, while $\hat \rho_k^\lambda < \upsilon$.
Where $\delta_{II}$ is a user defined parameter that permits the specification of where the gap between 
$\hat \rho_k^\lambda$ and $\rho_k^\lambda$ is great enough to be considered problematic. By choosing the 
right stopping criterion we can then control the probability of this error at  $\xi_{II}$.

The options for these stopping criterion parameters are represented by the options $\delta_I =$ `delta1`, $\delta_{II} = $ `delta2`,
$\xi_I =$ `chi1`, $\xi_{II}=$ `chi2`, and $\upsilon=$ `upsilon`. By default `upsilon=1e-10`, `delta1=.9`, `delta2=1.1`, `chi_I = .01`,
`chi_{II} = .01`. To use the stopping criterion, the user must input a max number of iterations, and specify
changes to the default settings as options. So for instance if one wanted to use the stopping criterion with a 
maximum iteration of 1000 and `upsilon = 1e-3`, the following code could be used. 

```julia

using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(lambda_2 = 100),# Full Logger: maintains moving average residual history
    LSStopMA(1000, upsilon=1e-3),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
bounds = get_uncertainty(sol.log, alpha = .99)
```
