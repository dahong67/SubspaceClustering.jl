using SubspaceClustering
using Test
using Aqua
using JET

@testset "SubspaceClustering.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SubspaceClustering)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SubspaceClustering; target_defined_modules = true)
    end
    # Write your tests here.
end
