# General checks

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(SubspaceClustering)
end

@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(SubspaceClustering; target_defined_modules = true)
end
