# kas function

@testitem "Random data with 2 clusters with same affine space dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(0)
    D, N = 5, 20
    X = randn(rng, D, N)
    d = [2, 2]
    result = kas(X, d)
    U, b, c = result.U, result.b, result.c

    @test length(U) == length(d)
    @test length(c) == N
    @test size(U[1]) == (D, d[1])
    @test size(U[2]) == (D, d[2])
    @test length(b) == length(d)

    for subspace in U
        @test isapprox(subspace' * subspace, I, atol = 1e-10)
    end
    for bias in b
        @test length(bias) == D
    end
end

@testitem "Random data with 3 clusters with different affine space dimensions" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(1)
    D, N = 5, 20
    X = randn(rng, D, N)
    d = [2, 3, 4]
    result = kas(X, d)
    U, b, c = result.U, result.b, result.c

    @test length(U) == length(d)
    @test length(c) == N
    @test size(U[1]) == (D, d[1])
    @test size(U[2]) == (D, d[2])
    @test size(U[3]) == (D, d[3])
    @test length(b) == length(d)

    for subspace in U
        @test isapprox(subspace' * subspace, I, atol = 1e-10)
    end
    for bias in b
        @test length(bias) == D
    end
end

@testitem "Empty Cluster case" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(2)
    D, N = 7, 30
    d = [2, 3]
    U1 = SubspaceClustering.randsubspace(rng, D, d[1])
    b1 = zeros(D)
    X = U1 * randn(rng, d[1], N) .+ b1
    U2 = SubspaceClustering.randsubspace(rng, D, d[2])
    b2 = zeros(D)
    result = kas(X, d; init=[(U1, b1), (U2, b2)])
    U, b, c = result.U, result.b, result.c

    @test isempty(findall(==(2), c))

end

@testitem "Nontrivial cluster case with noise"  begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    D, N = 8, 40
    d = [3, 4]
    U1 = SubspaceClustering.randsubspace(rng, D, d[1])
    b1 = zeros(D)
    U2 = SubspaceClustering.randsubspace(rng, D, d[2])
    b2 = zeros(D)
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

@testitem "Argument validation" begin
    using LinearAlgebra, StableRNGs

    @testset "affine space dimension > ambient dimension" begin
        rng = StableRNG(4)
        X = randn(rng, 5, 60)
        d = [6, 7]
        # Create init matrices that have the right number of columns (6 and 7)
        # but wrong ambient dimension
        U1 = randn(rng, 5, 6)  # 5×6 matrix - has d[1]=6 cols
        b1 = zeros(5)
        U2 = randn(rng, 5, 7)  # 5×7 matrix - has d[2]=7 cols  
        b2 = zeros(5)
        @test_throws DimensionMismatch kas(X, d; init=[(U1, b1), (U2, b2)])
    end
    
    @testset "invalid affine space dimension" begin
        rng = StableRNG(5)
        X = randn(rng, 5, 80)
        d = [2, -1]  # Use valid first dimension, invalid second
        U1 = SubspaceClustering.randsubspace(rng, 5, 2)
        b1 = zeros(5)
        U2 = randn(rng, 5, 0)  # 5×0 matrix - has d[2]=0 cols, but d[2]=-1 is invalid
        b2 = zeros(5)
        @test_throws ArgumentError kas(X, d; init=[(U1, b1), (U2, b2)])  # Will fail on column check since -1 != 0
    end

    @testset "invalid number of iterations" begin
        rng = StableRNG(6)
        X = randn(rng, 5, 100)
        d = [2, 3]
        @test_throws ArgumentError kas(X, d; maxiters = -1)
    end
end

@testitem "Bias vector validation" begin
    using LinearAlgebra, StableRNGs

    rng = StableRNG(7)
    D, N = 5, 20
    X = randn(rng, D, N)
    d = [2, 2]
    result = kas(X, d)
    b = result.b

    # Test that bias vectors are not NaN
    @test all(!isnan(val) for bias in b for val in bias)
    
    # Test that bias vectors are finite
    @test all(isfinite(val) for bias in b for val in bias)
end