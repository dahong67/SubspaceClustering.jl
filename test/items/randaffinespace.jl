# randaffinespace function

@testitem "Small 2x2 matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    U, b = SubspaceClustering.randaffinespace(rng, 2, 2)
    @test size(U) == (2, 2)
    @test length(b) == 2
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "Rectangular (tall) matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    U, b = SubspaceClustering.randaffinespace(rng, 6, 4)
    @test size(U) == (6, 4)
    @test length(b) == 6
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "Square matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    U, b = SubspaceClustering.randaffinespace(rng, 4, 4)
    @test size(U) == (4, 4)
    @test length(b) == 4
    @test isapprox(U' * U, I, atol=1e-10)
    
end
