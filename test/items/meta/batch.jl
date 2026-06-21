
@testitem "Argument Validation" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)

    U1 = qr(randn(rng, 5, 2)).Q[:, 1:2]
    U2 = qr(randn(rng, 5, 2)).Q[:, 1:2]

    X = [U1*randn(rng, 2, 20) U2*randn(rng, 2, 20)]
    d = [1, 2]

    @testset "invalid nruns" begin
        @test_throws ArgumentError batch(kss, X, d; nruns = 0)
        @test_throws ArgumentError batch(kas, X, d; nruns = -1)
    end

    @testset "invalid algorithm" begin
        @test_throws ArgumentError batch(tsc, X, d; nruns = 5)
    end

    @testset "invalid dimensions" begin
        @test_throws MethodError batch(kss, X, 2)
        @test_throws MethodError batch(kas, X, 3)
    end
end

@testitem "batch KSS Results" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(2)

    D, N = 10, 50

    U1 = qr(randn(rng, D, N)).Q[:, 1:3]
    U2 = qr(randn(rng, D, N)).Q[:, 1:3]

    X = [U1*randn(rng, 3, N) U2*randn(rng, 3, N)]

    result = batch(kss, X, [1, 2]; nruns = 5)
    U = result.U

    @test result isa KSSResult
    @test length(result.c) == size(X, 2)

    for subspace in U
        @test isapprox(subspace' * subspace, I, atol = 1e-10)
    end
end

@testitem "batch KAS Results" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)

    D, N = 10, 50

    U1 = qr(randn(rng, D, N)).Q[:, 1:3]
    U2 = qr(randn(rng, D, N)).Q[:, 1:3]

    X = [U1*randn(rng, 3, N) U2*randn(rng, 3, N)]

    result = batch(kas, X, [1, 2]; nruns = 5)
    U, b = result.U, result.b

    @test result isa KASResult
    @test length(result.c) == size(X, 2)

    for subspace in U
        @test isapprox(subspace' * subspace, I, atol = 1e-10)
    end
    for bias in b
        @test length(bias) == D
        @test all(!isnan, bias)
    end
end
