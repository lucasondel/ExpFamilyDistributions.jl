push!(LOAD_PATH, "../src/")

using Documenter
using ExpFamilyDistributions

DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

makedocs(
    sitename="ExpFamilyDistributions.jl",
    modules = [ExpFamilyDistributions],
    pages = [
        "Home" => "index.md",
        "Exponential Family Distributions" => "expfamily.md",
        "Distributions" => "dists.md",
    ],
)
deploydocs(
    repo = "github.com/lucasondel/ExpFamilyDistributions.jl.git",
)

