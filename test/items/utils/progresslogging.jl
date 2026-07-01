# @withprogressif and @logprogressif macros

@testitem "@withprogressif true / @logprogressif true" begin
    using Logging, ProgressLogging, Test
    using SubspaceClustering: @withprogressif, @logprogressif

    logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
    with_logger(logger) do
        @withprogressif true for i in 1:3
            @logprogressif true i / 3
        end
    end
    logged_progress = [log.kwargs[:progress] for log in logger.logs]
    @test logged_progress == [nothing; (1/3):(1/3):1; "done"]
end

@testitem "@withprogressif false / @logprogressif false" begin
    using Logging, ProgressLogging, Test
    using SubspaceClustering: @withprogressif, @logprogressif

    results = Int[]
    logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
    with_logger(logger) do
        @withprogressif false for i in 1:3
            @logprogressif false i / 3
            push!(results, i)
        end
    end
    @test isempty(logger.logs)
    @test results == [1, 2, 3]  # body still executed
end

@testitem "@withprogressif true / @logprogressif false" begin
    using Logging, ProgressLogging, Test
    using SubspaceClustering: @withprogressif, @logprogressif

    logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
    with_logger(logger) do
        @withprogressif true for i in 1:3
            @logprogressif false i / 3
        end
    end
    logged_progress = [log.kwargs[:progress] for log in logger.logs]
    @test logged_progress == [nothing, "done"]
end

@testitem "@withprogressif with name keyword" begin
    using Logging, ProgressLogging, Test
    using SubspaceClustering: @withprogressif, @logprogressif

    logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
    with_logger(logger) do
        @withprogressif true name = "my progress" for i in 1:2
            @logprogressif true i / 2
        end
    end
    logged_messages = [string(log.message) for log in logger.logs]
    @test all(==("my progress"), logged_messages)
end

@testitem "@withprogressif nested: outer false, inner true" begin
    using Logging, ProgressLogging, Test
    using SubspaceClustering: @withprogressif, @logprogressif

    logger = TestLogger(; min_level = ProgressLogging.ProgressLevel)
    with_logger(logger) do
        @withprogressif false name = "outer" for i in 1:2
            @logprogressif false i / 2
            @withprogressif true name = "inner" for j in 1:2
                @logprogressif true j / 2
            end
        end
    end
    logged_progress = [log.kwargs[:progress] for log in logger.logs]
    @test logged_progress == repeat([nothing; (1/2):(1/2):1; "done"], 2)
    logged_messages = [string(log.message) for log in logger.logs]
    @test all(==("inner"), logged_messages)
end
