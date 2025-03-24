using TestItemRunner

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    using SubspaceClustering
    Aqua.test_all(SubspaceClustering)
end

@testitem "Code linting (JET.jl)" begin
    using JET
    using SubspaceClustering
    JET.test_package(SubspaceClustering; target_defined_modules = true)
end

@run_package_tests
