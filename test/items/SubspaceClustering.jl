## SubspaceClustering

@testitem "polar function" begin
    using Random
    using LinearAlgebra

    @testset "Small 2x2 matrix" begin
        Random.seed!(0)
        A = randn(2, 2)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end


    @testset "Rectangular (Tall) matrix" begin
        Random.seed!(1)
        A = randn(6, 4)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end

    @testset "Rectangular (Wide) matrix" begin
        Random.seed!(2)
        A = randn(4, 6)
        Q = polar(A)
        @test isapprox(Q* Q', I, atol=1e-10)
    end

    @testset "Square matrix" begin
        Random.seed!(3)
        A = randn(4, 4)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end

end

@testitem "KSS function" begin
    using Random
    using LinearAlgebra

    @testset "Random Data with 2 Clusters with same subspace dimensions" begin
        Random.seed!(0)
        D, N = 5, 20
        X = randn(D, N)
        d = [2, 2]
        U, c = KSS(X, d; niters=100)

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])

        for subspace in U
            @test isapprox(subspace'* subspace, I, atol=1e-10)
        end
    end

    @testset "Random Data with 3 Clusters with different subspace dimensions" begin
        Random.seed!(1)
        D, N = 5, 20
        X = randn(D, N)
        d = [2, 3, 4]
        U, c = KSS(X, d; niters=100)

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])
        @test size(U[3]) == (D, d[3])

        for subspace in U
            @test isapprox(subspace'* subspace, I, atol=1e-10)
        end
    end

end