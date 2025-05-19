# randsubspace function

@testitem "Small 2x2 matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    Q = SubspaceClustering.randsubspace(rng, 2, 2)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Rectangular (tall) matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    Q = SubspaceClustering.randsubspace(rng, 6, 4)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Square matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    Q = SubspaceClustering.randsubspace(rng, 4, 4)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Small complex matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    Q = SubspaceClustering.randsubspace(rng, ComplexF64, 2, 2)
    @test typeof(U) == Matrix{ComplexF64}
    @test isapprox(Q' * Q, I, atol = 1e-10)

end