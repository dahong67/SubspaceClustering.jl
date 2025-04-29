# SubspaceClustering: Cluster data points by subspace

[![version](https://juliahub.com/docs/General/SubspaceClustering/stable/version.svg)](https://juliahub.com/ui/Packages/General/SubspaceClustering)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dahong67.github.io/SubspaceClustering.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dahong67.github.io/SubspaceClustering.jl/dev/)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![PkgEval](https://juliahub.com/docs/General/SubspaceClustering/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/SubspaceClustering)
[![Build Status](https://github.com/dahong67/SubspaceClustering.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dahong67/SubspaceClustering.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dahong67/SubspaceClustering.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dahong67/SubspaceClustering.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

> 👋 *This package provides research code and work is ongoing.
> If you are interested in using it in your own research
> or in contributing to it,
> **I'd love to hear from you and collaborate!**
> Feel free to write: [hong@udel.edu](mailto:hong@udel.edu)*

![](https://dahong67.github.io/SubspaceClustering.jl/dev/index-31d11d05.png)

Data points in many modern datasets lie not along a single low-dimensional subspace,
but rather **cluster** around multiple low-dimensional **subspaces**
(as illustrated above).
This is sometimes called "union-of-subspace" structure
since the data points lie near the union of these low-dimensional subspaces.

Subspace clustering algorithms seek to cluster the data points
by their corresponding subspace - without knowing the subspaces a priori!

Ready to start? Check out the [quick start guide](https://dahong67.github.io/SubspaceClustering.jl/dev/quickstart/)!
