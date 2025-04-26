# Developer Docs

## Live preview server for the documentation

Having a live preview can make it much easier to work on the documentation. We do this by running a local server that watches the files in `docs/` and rebuilds the documentation whenever there are edits. The script `docs/serve.jl` takes care of setting that all up. Run it from the shell as follows:
```bash
> julia docs/serve.jl   # from the package root folder
```
You should see a bunch of lines print out about the documentation build, concluding with a couple lines that look something like:
```bash
[... lines about the documentation build ...]
✓ LiveServer listening on http://localhost:8000/ ...
  (use CTRL+C to shut down)
```
Opening this link in a web browser will show a live preview of this documentation. Edit one of the documentation pages in `docs/src` and you should see your edits appear automatically!

**Minor Caveats:**
- Updates to docstrings are currently not automatically caught. To solve, simply shut down the server and relaunch.
- Adding a file to the documentation does not always register correctly. To solve, simply shut down the server and relaunch.

!!! tip "How does this all work?"

    The `docs/serve.jl` script does the following:
    1. Make sure the documentation packages (defined in `docs/Project.toml`) are all installed.
    2. Build and serve the documentation using the `servedocs` function from [LiveServer.jl](https://github.com/JuliaDocs/LiveServer.jl).
