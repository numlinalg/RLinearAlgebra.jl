using Documenter
using RLinearAlgebra

makedocs(
    sitename = "RLinearAlgebra",
    format = Documenter.HTML(
    collapselevel=1,
    ),
    modules = [RLinearAlgebra],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Consistent Linear Systems" => "man/cls_overview.md",
            "Tracking" => "man/tracking_overview.md"
        ],
        "API Reference" => [
            "Compressors" => "api/Compressors.md",
            "Solvers" => "api/solvers.md",
            "Solver SubRoutines" => [
                "SubSolvers" => "api/sub_solvers.md",
                "SolverErrors" => "api/solver_error.md",
                "Loggers" => "api/loggers.md"
            ],
            "Approximators" => "api/linear_solver_stops.md",
        ],
        "Development" => [
            "Design" => "dev/design.md"
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RLinearAlgebra.jl"
)
