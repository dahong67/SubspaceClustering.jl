# ekss function

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(4)
    X = randn(rng, 5, 20)

    @testset "invalid subspace dimension" begin
        @test_throws ArgumentError ekss(X, 0, 2)
        @test_throws ArgumentError ekss(X, -1, 2)
    end

    @testset "invalid number of clusters" begin
        @test_throws ArgumentError ekss(X, 2, 0)
        @test_throws ArgumentError ekss(X, 2, -1)
    end

    @testset "invalid number of candidate subspaces" begin
        @test_throws ArgumentError ekss(X, 2, 2; Kbar = 0)
        @test_throws ArgumentError ekss(X, 2, 2; Kbar = -1)
    end

    @testset "invalid number of iterations" begin
        @test_throws ArgumentError ekss(X, 2, 2; maxiters = 0)
        @test_throws ArgumentError ekss(X, 2, 2; maxiters = -1)
    end

    @testset "invalid number of ensemble runs" begin
        @test_throws ArgumentError ekss(X, 2, 2; nruns = 0)
        @test_throws ArgumentError ekss(X, 2, 2; nruns = -1)
    end

    @testset "invalid number of kmeans runs" begin
        @test_throws ArgumentError ekss(X, 2, 2; kmeans_nruns = 0)
        @test_throws ArgumentError ekss(X, 2, 2; kmeans_nruns = -1)
    end

    @testset "invalid threshold parameter q" begin
        N = size(X, 2)
        @test_throws ArgumentError ekss(X, 2, 2; q = 0)
        @test_throws ArgumentError ekss(X, 2, 2; q = N)
        @test_throws ArgumentError ekss(X, 2, 2; q = N + 1)
    end
end

@testitem "Parallel and serial modes return valid outputs" begin
    using LinearAlgebra, StableRNGs, SparseArrays

    rng = StableRNG(7)

    # Construct data with actual union-of-subspaces structure:
    # 3 subspaces, each of dimension 2, embedded in R^8,
    # with 6 points per subspace => total 18 points.
    D = 8
    d = 2
    K = 3
    n_per_cluster = 6

    subspaces = [qr(randn(rng, D, d)).Q[:, 1:d] for _ in 1:K]
    X = reduce(hcat, [U * randn(rng, d, n_per_cluster) for U in subspaces])

    result_serial = ekss(
        X, d, K;
        rng = StableRNG(11),
        nruns = 12,
        kmeans_nruns = 8,
        q = 4,
        parallel = false,
    )

    result_parallel = ekss(
        X, d, K;
        rng = StableRNG(11),
        nruns = 12,
        kmeans_nruns = 8,
        q = 4,
        parallel = true,
    )

    @testset "co-association matrix" begin
        A1 = result_serial.coassoc
        A2 = result_parallel.coassoc

        @test size(A1) == (K * n_per_cluster, K * n_per_cluster)
        @test size(A2) == (K * n_per_cluster, K * n_per_cluster)

        @test issymmetric(Matrix(A1))
        @test issymmetric(Matrix(A2))

        @test all(diag(Matrix(A1)) .== 0)
        @test all(diag(Matrix(A2)) .== 0)

        @test nnz(A1) > 0
        @test nnz(A2) > 0
    end

    @testset "embedding" begin
        E1 = result_serial.embedding
        E2 = result_parallel.embedding

        @test size(E1) == (K, K * n_per_cluster)
        @test size(E2) == (K, K * n_per_cluster)

        @test all(isfinite, E1)
        @test all(isfinite, E2)
    end

    @testset "kmeans outputs" begin
        @test length(result_serial.kmeans_runs) == 8
        @test length(result_parallel.kmeans_runs) == 8

        @test length(result_serial.assignments) == K * n_per_cluster
        @test length(result_parallel.assignments) == K * n_per_cluster

        @test all(1 .<= result_serial.assignments .<= K)
        @test all(1 .<= result_parallel.assignments .<= K)
    end
end

@testitem "Basic noiseless case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(5)

    # Three 2D subspaces in ambient dimension 20, 4 points each
    X = reduce(hcat, [svd(randn(rng, 20, 2)).U * randn(rng, 2, 4) for _ in 1:3])

    result = ekss(X, 2, 3; rng, nruns = 10, kmeans_nruns = 20, q = 3)

    @test Set([findall(==(k), result.assignments) for k in 1:3]) == Set([1:4, 5:8, 9:12])
end