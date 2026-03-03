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

@testitem "randsubspace fallbacks" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(5)
    Q1 = SubspaceClustering.randsubspace(Float64, 4, 2)
    @test eltype(Q1) == Float64
    @test isapprox(Q1' * Q1, I, atol = 1e-10)

    Q2 = SubspaceClustering.randsubspace(ComplexF64, 4, 2)
    @test eltype(Q2) == ComplexF64
    @test isapprox(Q2' * Q2, I, atol = 1e-10)

    U = randn(rng, Float64, 4, 2)
    Q3 = SubspaceClustering.randsubspace!(U)
    @test eltype(Q3) == Float64
    @test size(Q3) == (4, 2)
    @test isapprox(Q3' * Q3, I, atol = 1e-10)
end

@testitem "randsubspace with type parameter" begin
    using LinearAlgebra, Random

    # Test with type parameter (no rng specified, uses default_rng)
    Random.seed!(42)
    Q = SubspaceClustering.randsubspace(Float32, 5, 3)
    @test eltype(Q) == Float32
    @test size(Q) == (5, 3)
    @test isapprox(Q' * Q, I)
end

@testitem "randsubspace with default parameters" begin
    using LinearAlgebra, Random

    # Test without rng or type (uses default_rng and Float64)
    Random.seed!(43)
    Q = SubspaceClustering.randsubspace(4, 2)
    @test eltype(Q) == Float64
    @test size(Q) == (4, 2)
    @test isapprox(Q' * Q, I)
end

@testitem "randsubspace! with default rng" begin
    using LinearAlgebra, Random

    # Test in-place version without rng
    Random.seed!(44)
    U = Matrix{Float64}(undef, 5, 3)
    result = SubspaceClustering.randsubspace!(U)
    @test result === U  # Should return the same matrix
    @test size(U) == (5, 3)
    @test isapprox(U' * U, I)
end
