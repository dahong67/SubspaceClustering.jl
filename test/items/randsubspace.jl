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

@testitem "randsubspace with type parameter" begin
    using LinearAlgebra, Random

    # Test with type parameter (no rng specified, uses default_rng)
    Random.seed!(42)
    Q = SubspaceClustering.randsubspace(Float32, 5, 3)
    @test eltype(Q) == Float32
    @test size(Q) == (5, 3)
    @test isapprox(Q' * Q, I, atol = 1e-6)
end

@testitem "randsubspace with default parameters" begin
    using LinearAlgebra, Random

    # Test without rng or type (uses default_rng and Float64)
    Random.seed!(43)
    Q = SubspaceClustering.randsubspace(4, 2)
    @test eltype(Q) == Float64
    @test size(Q) == (4, 2)
    @test isapprox(Q' * Q, I, atol = 1e-10)
end

@testitem "randsubspace! with default rng" begin
    using LinearAlgebra, Random

    # Test in-place version without rng
    Random.seed!(44)
    U = Matrix{Float64}(undef, 5, 3)
    result = SubspaceClustering.randsubspace!(U)
    @test result === U  # Should return the same matrix
    @test size(U) == (5, 3)
    @test isapprox(U' * U, I, atol = 1e-10)
end
