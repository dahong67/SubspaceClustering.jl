# This file contains the tests for the SubspaceClustering.jl package.

@testitem "randsubspace function" begin
    
    using TestItemRunner
    using StableRNGs
    using LinearAlgebra
    using SubspaceClustering
    using Test

    @testset "Small 2x2 matrix" begin
        rng = StableRNG(0)
        Q = randsubspace(rng, 2, 2)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end


    @testset "Rectangular (Tall) matrix" begin
        rng = StableRNG(1)  
        Q = randsubspace(rng, 6, 4)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end

    @testset "Rectangular (Wide) matrix" begin
        rng = StableRNG(2)  
        Q = randsubspace(rng, 4, 6)
        @test isapprox(Q* Q', I, atol=1e-10)
    end

    @testset "Square matrix" begin
        rng = StableRNG(3) 
        Q = randsubspace(rng, 4, 4)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end


end

@testitem "KSS function" begin
    
    using TestItemRunner
    using StableRNGs
    using LinearAlgebra
    using SubspaceClustering
    using Test

    @testset "Random Data with 2 Clusters with same subspace dimensions" begin
        rng = StableRNG(0)
        D, N = 5, 20
        X = randn(rng, D, N)
        d = [2, 2]
        result = KSS(X, d)
        U, c = result.U, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])

        for subspace in U
            @test isapprox(subspace'* subspace, I, atol=1e-10)
        end
    end

    @testset "Random Data with 3 Clusters with different subspace dimensions" begin
        rng = StableRNG(1)
        D, N = 5, 20
        X = randn(rng, D, N)
        d = [2, 3, 4]
        result = KSS(X, d)
        U, c = result.U, result.c

        @test length(U) == length(d)
        @test length(c) == N
        @test size(U[1]) == (D, d[1])
        @test size(U[2]) == (D, d[2])
        @test size(U[3]) == (D, d[3])

        for subspace in U
            @test isapprox(subspace'* subspace, I, atol=1e-10)
        end
    end

    @testset "Empty Cluster case" begin
        rng = StableRNG(2)
        D, N = 5, 20
        d = [2, 3]
        U1 = randsubspace(rng, D, d[1])
        X = U1 * randn(rng, d[1], N)
        U2 = randsubspace(rng, D, d[2])
        result = KSS(X, d; Uinit=[U1, U2])
        U, c = result.U, result.c

        @test isempty(findall(==(2), c))
    end

    @testset "Nontrivial Cluster case with Noise" begin
        rng = StableRNG(3)
        D, N = 7, 20
        d = [2, 3]

        U1 = randsubspace(rng, D, d[1])
        U2 = randsubspace(rng, D, d[2])
        X = hcat(U1 * randn(rng, d[1], N), U2 * randn(rng, d[2], N)) + 0.01 * randn(rng, D, 2N)
        result = KSS(X, d; Uinit=[U1, U2])
        U, c = result.U, result.c

        #Checking all the points in X1 are assigned to cluster 1 and all the points in X2 are assigned to cluster 2
        @test all(c[1:N] .== 1)
        @test all(c[N+1:end] .== 2)


        #Confirming clusters aren't empty
        for k in 1:length(d)
            @test !isempty(findall(==(k), c))
        end
        
    end

    @testset "Argument Checks" begin

        @testset "Subspace Dimensions > Feature space Dimensions" begin
            rng = StableRNG(4)
            X = randn(rng, 5, 20)
            d = [6, 7]
    
            @test_throws DimensionMismatch KSS(X, d)
        end
        
        @testset "Invalid Subspace Dimensions" begin
            rng = StableRNG(5)
            X = randn(rng, 5, 40)
            d = [0, -1]
            @test_throws ArgumentError KSS(X, d)
        end

        @testset "Invalid Number of Iterations" begin
            rng = StableRNG(6)
            X = randn(rng, 5, 40)
            d = [2, 3]
            @test_throws ArgumentError KSS(X, d; niters=0)
        end

    end


end


