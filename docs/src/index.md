# RLinearAlgebra

[RLinearAlgebra](https://github.com/numlinalg/RLinearAlgebra.jl) is a Julia library for the development and application of randomized algorithms to the solution of linear systems of equations. Because of the large diversity of randomized techniques, rather than offering isolated routine implementations of algorithms, this library implements a series of extendable data structures and methods which allow code reuse.

Currently, we implement data structures which allow the implementation of the following solver categories:

* **Row Projection Methods (RPM)** including randomized Kazmarcz, Gauss-sketch, and sampled Motzkin.
* **Column Projection Methods (CPM)** including randomized Coordinate Descent and randomized Gauss Seidel.

Users can use the available algorithms or create new ones by adding new methods. For example, the RPM methods are composed by a sampling and a projection methods. Users can easily create new algorithms by implementing a new row sampling technique and leverage the existing projection methods.

To facilitate benchmarking, we provide an abstract class for solver solution logging and stopping criteria.

## Documentation structure

This documentation serves both as a manual to the library and as an introduction to randomized linear algebra solvers. We divide it in three parts:

* **Tutorial**: here we offer examples of how to solve linear systems with RLinearAlgebra and how to extend the library.
* **Manual**: here we offer an introduction to solving linear systems with randomized linear algebra techniques. We introduce theoretical foundations and we provide code examples with RLinearAlgebra.
* **API**: a detailed description of all the classes and methods of the library.

## Acknowledgements
This work is supported by the National Science Foundation Office of Advanced Cyberinfrastructure under awards [2309445](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2309445) and [2309446](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2309446).
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
