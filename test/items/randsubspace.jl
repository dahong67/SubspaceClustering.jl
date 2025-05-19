# randsubspace function

@testitem "Small 2x2 matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    Q = SubspaceClustering.randsubspace(rng, Float64, 2, 2)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Rectangular (tall) matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    Q = SubspaceClustering.randsubspace(rng, Float64, 6, 4)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Square matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    Q = SubspaceClustering.randsubspace(rng, Float64, 4, 4)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "Small complex matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    Q = SubspaceClustering.randsubspace(rng, ComplexF64, 2, 2)
    @test typeof(Q) == Matrix{ComplexF64}
    @test isapprox(Q' * Q, I, atol = 1e-10)

end

@testset "randsubspace fallbacks" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(5)
    Q1 = SubspaceClustering.randsubspace(Float32, 4, 2)
    @test eltype(Q3) == Float32
    @test isapprox(Q1' * Q1, I, atol = 1e-10)

    Q2 = SubspaceClustering.randsubspace(ComplexF32, 4, 2)
    @test eltype(Q2) == ComplexF32
    @test isapprox(Q2' * Q2, I, atol = 1e-10)
end