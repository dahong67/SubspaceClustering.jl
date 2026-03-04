## Script to create the logo file `docs/src/assets/logo.svg`

# Activate and instantiate the `docs` environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load Luxor
using Luxor

# Setup path
ASSET_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSET_DIR)

# Setup drawing
Drawing(500, 500, joinpath(ASSET_DIR, "logo.svg"))
setline(16)
origin()

# Draw plane
setcolor(Luxor.julia_purple)
plane = poly(
    [Point(-250, 50), Point(-100, -50), Point(250, -50), Point(100, 50)],
    :fill;
    close = true,
)

# Draw first line
setcolor(Luxor.julia_red)
line(Point(240, -120), Point(-240, 120); action = :stroke)

# Draw second line
setcolor(Luxor.julia_green)
line(Point(-140, -240), Point(140, 240); action = :stroke)

# Redraw lower half of plane (to handle overlap)
setcolor(Luxor.julia_purple)
poly(polyclip(plane, box([Point(-250, 0), Point(250, 250)])), :fill; close = true)

# Finish drawing
finish()
preview()
