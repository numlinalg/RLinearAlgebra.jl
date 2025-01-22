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
#        "Manual" => [
#            "Consistent Linear Systems" => "man/cls_overview.md",
#            "Tracking" => "man/tracking_overview.md"
#        ],
        "API Reference" => [
            "Compressors" => "api/compressors.md",
            "Solvers" => "api/solvers.md",
            "Solver Sub-routines" => [
                "SubSolvers" => "api/sub_solvers.md",
                "SolverErrors" => "api/solver_error.md",
                "Loggers" => "api/loggers.md"
            ],
            "Approximators" => "api/approximators.md",
            "Approximator Sub-routines" => [
                "ApproximatorErrors" => "api/approximator_error.md"
                                           ],
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
