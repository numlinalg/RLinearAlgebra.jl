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
            "Randomized Linear Solvers" => "api/linear_rsolve.md",
            "Linear Samplers" => "api/linear_samplers.md",
            "Linear Subsolvers" => [
                "Main Solvers" => "api/linear_solver_routines.md",
                "Solver Helpers" => "api/linear_solver_helpers.md",
            ],
            "Linear Solver Logs" => "api/linear_solver_logs.md",
            "Linear Solver Stop Criteria" => "api/linear_solver_stops.md",
            "Low Rank Approximations" => "api/low_rank_approximations.md",
        ],
        "Developers" => [
            "Contributing" => "dev/contributing.md"
            "Style Guide" => "dev/styleguide.md"
            "Checklists" => "dev/checklists.md"
        ],
        "Developers" => [
            "Contributing" => "dev/contributing.md"
            "Style Guide" => "dev/styleguide.md"
            "Checklists" => "dev/checklists.md"
        ],
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RLinearAlgebra.jl"
)
