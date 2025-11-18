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
    ),
    plugins=[bib],
    modules = [RLinearAlgebra],
    pages = [
        "Home" => "index.md",
#        "Tutorials" => [
#            "Introduction" => "tutorials/introduction.md",
#            "Getting started" => "tutorials/getting_started.md"
#        ],
        "Manual" => [
            "Introduction" => "manual/introduction.md", 
            "Compression" => "manual/compression.md",
            "Low-Rank Approximation" => "manual/low_rank_approximators.md",
            "Linear Systems" => "manual/linear_systems.md"
        ],
        "API Reference" => [
            "Compressors" => [
                "Compressors API" => "api/compressors.md",
                "Distributions API" => "api/distributions.md",
            ],
            "Solvers" => [
                "Solvers API" => "api/solvers.md",
                "SubSolvers API" => "api/sub_solvers.md",
                "SolverErrors API" => "api/solver_errors.md",
                "Loggers API" => "api/loggers.md",
            ],
            "Approximators" => [
                "Approximators API" => "api/approximators.md",
                "ApproximatorErrors API" => "api/approximator_errors.md",
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
    devbranch = "master", # master's newest commit will become dev
    push_preview = true # pull requests to the master will become available
)
