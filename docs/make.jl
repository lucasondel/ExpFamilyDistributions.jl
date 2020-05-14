using Documenter

using Pkg
Pkg.activate("../")
using ExpFamilyDistributions


makedocs(
    sitename="ExpFamilyDistributions",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Installation" => "install.md",
            "Usage" => "usage.md"
        ],
    ]
)
