# This file contains the tests for the SubspaceClustering.jl package.

@testitem "polar function" begin
    
    using TestItemRunner
    using StableRNGs
    using LinearAlgebra
    using SubspaceClustering
    using Test

    @testset "Small 2x2 matrix" begin
        rng = StableRNG(0)
        A = randn(rng, 2, 2)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end


    @testset "Rectangular (Tall) matrix" begin
        rng = StableRNG(1)  
        A = randn(rng, 6, 4)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end

    @testset "Rectangular (Wide) matrix" begin
        rng = StableRNG(2)  
        A = randn(rng, 4, 6)
        Q = polar(A)
        @test isapprox(Q* Q', I, atol=1e-10)
    end

    @testset "Square matrix" begin
        rng = StableRNG(3)  
        A = randn(rng, 4, 4)
        Q = polar(A)
        @test isapprox(Q'* Q, I, atol=1e-10)
    end

    @testset "Scaled Rotation" begin
        theta = π/4
        R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
        s = 3
        A = s * R

        Q = polar(A)
        @test isapprox(Q, R, atol=1e-5)
    end

    @testset "Diagonal Matrix" begin
        A = [ 3.0 0.0; 0.0 2.0]
        Q = polar(A)

        @test isapprox(Q, I(2), atol=1e-5)
    end

    @testset "Rectangular Polar Decomposition" begin
        A = [2 0; 0 1; 0 0]
        Q_exp = [1 0; 0 1; 0 0]
        Q_comp = polar(A)

        @test isapprox(Q_comp, Q_exp, atol=1e-5)
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
        result = KSS(X, d; niters=100, randng=rng)
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
        result = KSS(X, d; niters=100, randng=rng)
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
        U1 = polar(randn(rng, D, d[1]))
        X = U1 * randn(rng, 2, N)
        U2 = polar(randn(rng, D, d[2]))
        Uinit = [U1, U2]
        result = KSS(X, d; niters=100, randng=rng, Uinit=Uinit)
        U, c = result.U, result.c

        @test isempty(findall(==(2), c))
    end

    @testset "Nontrivial Cluster case" begin
        rng = StableRNG(3)
        D, N = 7, 20
        d = [2, 3]
        U1 = polar(randn(rng, D, d[1]))
        X1 = U1 * randn(rng, d[1], N)
        U2 = polar(randn(rng, D, d[2]))
        X2 = U2 * randn(rng, d[2], N)
        X = hcat(X1, X2)
        result = KSS(X, d; niters=100, randng=rng, Uinit=[U1, U2])
        U, c = result.U, result.c

        #Checking all the points in X1 are assigned to cluster 1 and all the points in X2 are assigned to cluster 2
        @test all(c[1:size(X1, 2)] .== 1)
        @test all(c[size(X1, 2)+1:end] .== 2)


        #Confirming clusters aren't empty
        for k in 1:length(d)
            @test !isempty(findall(==(k), c))
        end
        
    end

    @testset "Subspace Dimensions > Feature space Dimensions" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 20)
        d = [6, 7]

        @test_throws DimensionMismatch KSS(X, d; niters=100, randng=rng)
    end
end


