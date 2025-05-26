# kss function

@testitem "Random data with 2 clusters with same subspace dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    D, N = 5, 20
    for T in (Float64, ComplexF64)
        X = randn(rng, T, D, N)
        d = [2, 2]
        result = kss(X, d)
        U, c = result.U, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])

        for subspace in U
            @test isapprox(subspace' * subspace, I, atol = 1e-10)
        end
    end
end

@testitem "Random data with 2 clusters with same subspace dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    D, N = 5, 20
    for T in (Float64, ComplexF64)
        X = randn(rng, T, D, N)
        d = [2, 2]
        result = kss(X, d)
        U, c = result.U, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])

        for subspace in U
            @test isapprox(subspace' * subspace, I, atol = 1e-10)
        end
    end
end

@testitem "Random data with 3 clusters with different subspace dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    D, N = 5, 20
    for T in (Float64, ComplexF64)
        X = randn(rng, T, D, N)
        d = [2, 3, 4]
        result = kss(X, d)
        U, c = result.U, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])
        @test size(U[3]) == (D, d[3])

        for subspace in U
            @test isapprox(subspace' * subspace, I, atol = 1e-10)
        end
    end
end

@testitem "Empty cluster case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(2)
    D, N = 5, 20
    d = [2, 3]
    for T in (Float64, ComplexF64)
        U1 = SubspaceClustering.randsubspace(rng, T, D, d[1])
        X = U1 * randn(rng, d[1], N)
        U2 = SubspaceClustering.randsubspace(rng, T, D, d[2])
        result = kss(X, d; Uinit = [U1, U2])
        U, c = result.U, result.c

        @test isempty(findall(==(2), c))
    end
end

@testitem "Nontrivial cluster case with noise" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    D, N = 7, 20
    d = [2, 3]

    for T in (Float64, ComplexF64)
        U1 = SubspaceClustering.randsubspace(rng, T, D, d[1])
        U2 = SubspaceClustering.randsubspace(rng, T, D, d[2])
        X = hcat(U1 * randn(rng, d[1], N), U2 * randn(rng, d[2], N)) + 0.01 * randn(rng, D, 2N)
        result = kss(X, d; Uinit = [U1, U2])
        U, c = result.U, result.c

        # Checking all the points in X1 are assigned to cluster 1 and all the points in X2 are assigned to cluster 2
        @test all(c[1:N] .== 1)
        @test all(c[(N+1):end] .== 2)

        # Confirming clusters aren't empty
        for k in 1:length(d)
            @test !isempty(findall(==(k), c))
        end
    end
end

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "subspace dimension > ambient dimension" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        d = [6, 7]

        @test_throws ArgumentError kss(X, d)
    end

    @testset "invalid subspace dimension" begin
        rng = StableRNG(5)
        X = randn(rng, 5, 40)
        d = [0, -1]
        @test_throws ArgumentError kss(X, d)
    end

    @testset "invalid number of iterations" begin
        rng = StableRNG(6)
        X = randn(rng, 5, 40)
        d = [2, 3]
        @test_throws ArgumentError kss(X, d; maxiters = -1)
    end
end
