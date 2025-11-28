# kas function

@testitem "Random data with 2 clusters with same affine space dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    D, N = 5, 20
    for T in (Float64, ComplexF64)
        X = randn(rng, T, D, N)
        d = [2, 2]
        result = kas(X, d)
        U, b, c = result.U, result.b, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])
        @test length(b) == length(d)
        @test typeof(U) == Vector{Matrix{T}}
        @test typeof(b) == Vector{Vector{T}}

        for subspace in U
            @test isapprox(subspace' * subspace, I, atol = 1e-10)
        end
        for bias in b
            @test length(bias) == D
        end
    end
end

@testitem "Random data with 3 clusters with different affine space dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    D, N = 5, 20
    for T in (Float64, ComplexF64)
        X = randn(rng, T, D, N)
        d = [2, 3, 4]
        result = kas(X, d)
        U, b, c = result.U, result.b, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])
        @test size(U[3]) == (D, d[3])
        @test length(b) == length(d)
        @test typeof(U) == Vector{Matrix{T}}
        @test typeof(b) == Vector{Vector{T}}

        for subspace in U
            @test isapprox(subspace' * subspace, I, atol = 1e-10)
        end
        for bias in b
            @test length(bias) == D
        end
    end
end

@testitem "Empty Cluster case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(2)
    D, N = 7, 30
    d = [2, 3]
    for T in (Float64, ComplexF64)
        U1, b1 = SubspaceClustering.randaffinespace(rng, T, D, d[1])
        X = U1 * randn(rng, d[1], N) .+ b1
        U2, b2 = SubspaceClustering.randaffinespace(rng, T, D, d[2])
        result = kas(X, d; init=[(U1, b1), (U2, b2)])
        U, b, c = result.U, result.b, result.c

        @test isempty(findall(==(2), c))
    end
end

@testitem "Nontrivial cluster case with noise"  begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    D, N = 8, 40
    d = [3, 4]
    for T in (Float64, ComplexF64)
        U1, b1 = SubspaceClustering.randaffinespace(rng, T, D, d[1])
        U2, b2 = SubspaceClustering.randaffinespace(rng, T, D, d[2])
        X = hcat(
            U1 * randn(rng, d[1], N) .+ b1,
            U2 * randn(rng, d[2], N) .+ b2,
        ) .+ 0.01 * randn(rng, D, 2N)
        result = kas(X, d; init=[(U1, b1), (U2, b2)])
        U, b, c = result.U, result.b, result.c

        # Checking all the points in X1 are assigned to cluster 1 and all the points in X2 are assigned to cluster 2
        @test all(c[1:N] .== 1)
        @test all(c[N+1:end] .== 2)

        # Confirming the clusters are not empty
        for k in 1:length(d)
            @test !isempty(findall(==(k), c))
        end
    end
end

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "affine space dimension > ambient dimension" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 60)
        d = [6, 7]
        @test_throws ArgumentError kas(X, d)
    end
    
    @testset "invalid affine space dimension" begin
        rng = StableRNG(5)
        X = randn(rng, 5, 80)
        d = [0, -1]
        @test_throws ArgumentError kas(X, d)
    end

    @testset "invalid number of iterations" begin
        rng = StableRNG(6)
        X = randn(rng, 5, 100)
        d = [2, 3]
        @test_throws ArgumentError kas(X, d; maxiters = -1)
    end
end