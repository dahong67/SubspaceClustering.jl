# tsc function

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "invalid number of clusters" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X, 0)
        @test_throws ArgumentError tsc(X, -1)
        @test_throws ArgumentError tsc(X, size(X, 2) + 1)
    end

    @testset "invalid maximum number of neighbors" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X, 2; max_nz = 0)
    end

    @testset "invalid maximum chunk size" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X, 2; max_chunksize = 0)
    end

    @testset "invalid number of K-means runs" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X, 2; kmeans_nruns = 0)
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
