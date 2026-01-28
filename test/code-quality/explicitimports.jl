# ExplicitImports

@testitem "Implicit imports" begin
    using ExplicitImports
    @test check_no_implicit_imports(SubspaceClustering) === nothing
end

@testitem "Stale imports" begin
    using ExplicitImports
    @test check_no_stale_explicit_imports(SubspaceClustering) === nothing
end

@testitem "Non-owner imports" begin
    using ExplicitImports
    @test check_all_explicit_imports_via_owners(SubspaceClustering) === nothing
end

@testitem "Non-public imports" begin
    using ExplicitImports
    if VERSION >= v"1.11-"  # public only declared on Julia 1.11+
        @test check_all_explicit_imports_are_public(SubspaceClustering) === nothing
    end
end

@testitem "Non-owner qualified accesses" begin
    using ExplicitImports
    @test check_all_qualified_accesses_via_owners(SubspaceClustering) === nothing
end

@testitem "Non-public qualified accesses" begin
    using ExplicitImports
    if VERSION >= v"1.11-"  # public only declared on Julia 1.11+
        ignore = ()
        @test check_all_qualified_accesses_are_public(SubspaceClustering; ignore) ===
              nothing
    end
end

@testitem "Self-qualified accesses" begin
    using ExplicitImports
    @test check_no_self_qualified_accesses(SubspaceClustering) === nothing
end
