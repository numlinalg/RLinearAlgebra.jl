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
            "Getting started" => "tutorials/getting_started.md"
        ],
        "Manual" => [
            "Introduction" => "manual/introduction.md", 
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
        "Contributing" => [
            "Contributing Overview" => "dev/contributing.md",
            "Design of Library" => "dev/design.md",
            "Checklists" => [
                "dev/checklists.md", 
                "Compressors" => "dev/checklists/compressors.md",
                "Loggers" => "dev/checklists/loggers.md"
            ],
            "Style Guide" => "dev/style_guide.md",
        ],
        "References" => "references.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/numlinalg/RLinearAlgebra.jl",
    devbranch = "master", # master's newest commit will become dev
    push_preview = true # pull requests to the master will become available
)
