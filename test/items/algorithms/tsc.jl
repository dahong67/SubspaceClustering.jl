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
        @test_throws ArgumentError tsc(X; K = 2, max_nz = 0)
    end

    @testset "invalid maximum chunk size" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K = 2, max_chunksize = 0)
    end

    @testset "invalid number of K-means runs" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        @test_throws ArgumentError tsc(X; K = 2, kmeans_nruns = 0)
    end
end

@testitem "Basic noiseless case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    X = reduce(hcat, [svd(randn(rng, 100, 2)).U * randn(rng, 2, 4) for _ in 1:3])
    result = tsc(X; K = 3, rng)

    @test Set([findall(==(k), result.assignments) for k in 1:3]) == Set([1:4, 5:8, 9:12])
end

@testitem "showprogress flag" begin
    using LinearAlgebra, Logging, ProgressLogging, StableRNGs, Test

    # Generate data
    rng = StableRNG(4)
    X = reduce(hcat, [svd(randn(rng, 100, 2)).U * randn(rng, 2, 10) for _ in 1:3])

    @testset "showprogress = true" begin
        logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
        with_logger(logger) do
            return tsc(
                X,
                3;
                showprogress = true,
                rng = StableRNG(4),
                kmeans_nruns = 5,
                max_chunksize = 3,
            )
        end
        progress_logs = filter(l -> l.level == ProgressLogging.ProgressLevel, logger.logs)
        logged_progress = [log.kwargs[:progress] for log in progress_logs]
        @test logged_progress ==
              [nothing; 0.1:0.1:1.0; "done"; nothing; 0.2:0.2:1.0; "done"]
    end

    @testset "showprogress = false" begin
        logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
        with_logger(logger) do
            return tsc(
                X,
                3;
                showprogress = false,
                rng = StableRNG(4),
                kmeans_nruns = 5,
                max_chunksize = 3,
            )
        end
        @test isempty(filter(l -> l.level == ProgressLogging.ProgressLevel, logger.logs))
    end
end
