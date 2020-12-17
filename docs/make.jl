push!(LOAD_PATH, "../src/")

using Documenter
using ExpFamilyDistributions

DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

makedocs(sitename="ExpFamilyDistributions", modules = [ExpFamilyDistributions])
deploydocs(repo = "github.com/BUTSpeechFIT/ExpFamilyDistributions.git")

