using SubspaceClustering
using Documenter

DocMeta.setdocmeta!(
    SubspaceClustering,
    :DocTestSetup,
    :(using SubspaceClustering);
    recursive = true,
)

makedocs(;
    modules = [SubspaceClustering],
    authors = "David Hong <hong@udel.edu> and contributors",
    sitename = "SubspaceClustering.jl",
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/SubspaceClustering.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "Developer Docs" => "devdocs.md"],
)

deploydocs(; repo = "github.com/dahong67/SubspaceClustering.jl", devbranch = "main")
