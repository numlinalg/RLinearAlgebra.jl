using Documenter
using RLinearAlgebra

makedocs(
    sitename = "RLinearAlgebra",
    format = Documenter.HTML(),
    modules = [RLinearAlgebra],
    pages = [
        "Home" => "index.md",
        hide("API Reference" => "api/contents.md", [
            "Randomized Linear Solvers" => "api/linear_rsolve.md",
            "Linear Samplers" => "api/linear_samplers.md",
            "Linear Subsolvers" => "api/linear_solver_routines.md",
            "Linear Solver Logs" => "api/linear_solver_logs.md",
            "Linear Solver Stop Criteria" => "api/linear_solver_stops.md",
        ]),
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RLinearAlgebra.jl"
)
