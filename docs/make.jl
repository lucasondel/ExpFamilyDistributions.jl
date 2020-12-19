push!(LOAD_PATH, "../src/")

using Documenter
using ExpFamilyDistributions

DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

makedocs(
    sitename="ExpFamilyDistributions",
    modules = [ExpFamilyDistributions],
    pages = [
        "Home" => "index.md",
        "Exponential Family Distributions" => "expfamily.md",
        "δ-Distributions" => "delta.md",
    ],
)
deploydocs(
    repo = "github.com/lucasondel/ExpFamilyDistributions.git",
)

