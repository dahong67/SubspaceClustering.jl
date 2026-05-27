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

@testitem "Basic noiseless case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    X = reduce(hcat, [svd(randn(rng, 100, 2)).U * randn(rng, 2, 4) for _ in 1:3])
    result = tsc(X, 3; rng)

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

@testitem "TSCResult show method" begin
    using StableRNGs

    X = randn(StableRNG(5), 5, 40)
    result = tsc(X, 3; rng = StableRNG(5))

    output = sprint((io, x) -> show(io, "text/plain", x), result)

    assignments_preview =
        length(result.assignments) > 10 ?
        string("[", join(result.assignments[1:10], ","), ", ...]") :
        string(result.assignments)

    expected_string = string(
        " TSCResult ($(size(result.embedding, 1)) clusters, $(length(result.assignments)) cluster assignments)\n\n",
        " assignments       :   $(assignments_preview)\n\n",
        " Additional Fields: \n\n",
        " affinity          :   $(size(result.affinity, 1))x$(size(result.affinity, 2)) matrix\n",
        " embedding         :   $(size(result.embedding, 1))x$(size(result.embedding, 2)) matrix\n",
        " kmeans_runs       :   $(length(result.kmeans_runs))\n",
    )

    @test output == expected_string
end
