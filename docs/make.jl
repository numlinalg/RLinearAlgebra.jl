using Documenter
using DocumenterCitations
using RLinearAlgebra

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)
makedocs(
    sitename = "RLinearAlgebra",
    format = Documenter.HTML(
    collapselevel=1,
    assets = String["custom_html.css"],
    ),
    plugins=[bib],
    modules = [RLinearAlgebra],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Introduction" => "tutorials/introduction.md",
            "Consistent Linear System" => [
                "tutorials/consistent_system/consistent_system.md",
                "tutorials/consistent_system/consistent_system_compressor.md",
            ],
        ],
        "API Reference" => [
            "Compressors" => [
                "Compressors Overview" => "api/compressors.md",
                "Compressor Sub-routines" => [
                    "Distributions" => "api/distributions.md",
                ],
            ],
            "Solvers" => [
                "Solvers Overview" => "api/solvers.md",
                "Solver Sub-routines" => [
                    "SubSolvers" => "api/sub_solvers.md",
                    "SolverErrors" => "api/solver_errors.md",
                    "Loggers" => "api/loggers.md"
                ],
            ],
            "Approximators" => [
                "Approximators Overview" => "api/approximators.md",
                "Approximator Sub-routines" => [
                    "ApproximatorErrors" => "api/approximator_errors.md"
                                           ],
            ],
        ],
        "References" => "references.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RLinearAlgebra.jl",
    devbranch = "main", # master's newest commit will become dev
    push_preview = true # pull requests to the master will become available
)
