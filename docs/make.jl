using Documenter
using RLinearAlgebra

makedocs(
    sitename = "RLinearAlgebra",
    format = Documenter.HTML(),
    modules = [RLinearAlgebra]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
