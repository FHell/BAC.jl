using Documenter
using Literate

using BAC
using DiffEqFlux

# generate examples
examples = [
    joinpath(@__DIR__, "..", "examples", "simple_graph_example.jl")
    #joinpath(@__DIR__, "..", "examples", "swing_equation.jl")
]
OUTPUT = joinpath(@__DIR__, "src/generated")
isdir(OUTPUT) && rm(OUTPUT, recursive=true)
mkpath(OUTPUT)

for ex in examples
    Literate.markdown(ex, OUTPUT)
    Literate.script(ex, OUTPUT)
end

makedocs(;
    modules=[BAC],
    authors="Frank Hellmann, Ekaterina Zolotarevskaia",
    repo="https://github.com/FHell/BAC.jl",
    sitename="BAC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fhell.github.io/BAC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => ["Simple graph example" => "generated/simple_graph_example.md"]
    ],
)

deploydocs(;
    repo="github.com/FHell/BAC.jl.git",
    devbranch="master",
    # push_preview=true,
)
