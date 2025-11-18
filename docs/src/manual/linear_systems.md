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
1. `Compressor`,
2. `SubSolver`,
3. `Logger`,
4. `ErrorMethod`,
