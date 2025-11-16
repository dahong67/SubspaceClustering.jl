using SubspaceClustering
using Documenter

DocMeta.setdocmeta!(
    SubspaceClustering,
    :DocTestSetup,
    :(using SubspaceClustering);
    recursive = true,
)

ENV["LINES"] = 9

makedocs(;
    modules = [SubspaceClustering],
    authors = "David Hong <hong@udel.edu> and contributors",
    sitename = "SubspaceClustering.jl",
    format = Documenter.HTML(;
        canonical = "https://dahong67.github.io/SubspaceClustering.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Quick start guide" => "quickstart.md",
        "Algorithms" => [
            "Overview" => "algs/main.md",
            "K-subspaces" => "algs/kss.md",
            "Thresholding-based subspace clustering" => "algs/tsc.md",
        ],
        "API" => "api.md",
        "Developer Docs" => "devdocs.md",
    ],
)

deploydocs(; repo = "github.com/dahong67/SubspaceClustering.jl", devbranch = "main")
