# tsc function

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "invalid number of clusters" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K = 0)
        @test_throws ArgumentError tsc(X; K = -1)
        @test_throws ArgumentError tsc(X; K = size(X, 2) + 1)
    end

    @testset "invalid maximum number of neighbors" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K=2, max_nz = 0)
    end

    @testset "invalid maximum chunk size" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K=2, max_chunksize = 0)
    end

    @testset "invalid number of K-means runs" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K=2, kmeans_nruns = 0)
    end
end

@testitem "Basic noiseless case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    X = reduce(hcat, [svd(randn(rng, 100, 2)).U * randn(rng, 2, 4) for _ in 1:3])
    result = tsc(X; K=3, rng)

    @test Set([findall(==(k), result.assignments) for k in 1:3]) == Set([1:4, 5:8, 9:12])
end

@testitem "Automatic K Estimation in noiseless case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    X = reduce(hcat, [svd(randn(rng, 100, 4)).U * randn(rng, 4, 300) for _ in 1:3])
    result = tsc(X; rng=rng)

    @test size(result.embedding, 1) == 3
    @test Set([findall(==(k), result.assignments) for k in 1:3]) == Set([1:300, 301:600, 601:900])
end