# tsc function

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "clusters > data points"  begin
        rng = StableRNG(1)
        X = randn(rng, 5, 30)

        @test_throws ArgumentError tsc(X, 40)
    end

    @testset "invalid number of iterations" begin
        rng = StableRNG(2)
        X = randn(rng, 5, 20)

        @test_throws ArgumentError tsc(X, 3; maxiters = -1)
    end

    @testset "invalid number of clusters" begin
        rng = StableRNG(3)
        X = randn(rng, 5, 40)

        @test_throws ArgumentError tsc(X, 0)
        @test_throws ArgumentError tsc(X, -1)
        @test_throws ArgumentError tsc(X, 1)
    end

end

@testitem "TSC: Basic functionality" begin
    using LinearAlgebra, StableRNGs
    
    rng = StableRNG(4)
    D, N = 5, 40
    k = 4
    X = randn(rng, D, N)
    result = tsc(X, k; maxiters = 80)

    @test length(result.c) == N
    @test length(result.counts) == k
    @test sum(result.counts) == N
    @test result.iterations <= 80
    @test size(result.U) == (k, k)

end