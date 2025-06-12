# randaffinespace function

@testitem "Small 2x2 matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    U, b = SubspaceClustering.randaffinespace(rng, Float64, 2, 2)
    @test size(U) == (2, 2)
    @test length(b) == 2
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "Rectangular (tall) matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    U, b = SubspaceClustering.randaffinespace(rng, Float64, 6, 2)
    @test size(U) == (6, 2)
    @test length(b) == 6
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "Square matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    U, b = SubspaceClustering.randaffinespace(rng, Float64, 4, 4)
    @test size(U) == (4, 4)
    @test length(b) == 4
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "Complex matrix" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    U, b = SubspaceClustering.randaffinespace(rng, ComplexF64, 6, 4)
    @test size(U) == (6, 4)
    @test length(b) == 6
    @test typeof(U) == Matrix{ComplexF64}
    @test typeof(b) == Vector{ComplexF64}
    @test isapprox(U' * U, I, atol=1e-10)
    
end

@testitem "randaffinespace fallbacks" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(5)
    Q1, b1 = SubspaceClustering.randaffinespace(rng, 6, 4)
    @test typeof(Q1)  == Matrix{Float64}
    @test typeof(b1) == Vector{Float64}
    @test isapprox(Q1' * Q1, I, atol=1e-10)

    Q2, b2 = SubspaceClustering.randaffinespace(6, 4)
    @test typeof(Q2)  == Matrix{Float64}
    @test typeof(b2) == Vector{Float64}
    @test isapprox(Q2' * Q2, I, atol=1e-10)

    U = randn(rng, Float64,  6, 4)
    b = randn(rng, Float64, 6)
    Q3, b3 = SubspaceClustering.randaffinespace!(U, b)
    @test size(Q3) == (6, 4)
    @test isapprox(Q3' * Q3, I, atol=1e-10)

    Q4, b4 = SubspaceClustering.randaffinespace(Float64, 6, 4)
    @test typeof(Q4) == Matrix{Float64}
    @test typeoof(b4) == Vector{Float64}
    @test isapprox(Q4' * Q4, I, atol=1e-10)
end