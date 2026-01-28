# JET

@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(SubspaceClustering)
end
