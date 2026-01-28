# Aqua

@testitem "Method ambiguities" begin
    using Aqua: test_ambiguities
    using SubspaceClustering
    test_ambiguities(SubspaceClustering)
end

@testitem "Unbound type parameters" begin
    using Aqua: test_unbound_args
    using SubspaceClustering
    test_unbound_args(SubspaceClustering)
end

@testitem "Undefined exports" begin
    using Aqua: test_undefined_exports
    using SubspaceClustering
    test_undefined_exports(SubspaceClustering)
end

@testitem "Stale dependencies" begin
    using Aqua: test_stale_deps
    using SubspaceClustering
    test_stale_deps(SubspaceClustering)
end

@testitem "Compat entries" begin
    using Aqua: test_deps_compat
    using SubspaceClustering
    test_deps_compat(SubspaceClustering)
end

@testitem "Type piracy" begin
    using Aqua: test_piracies
    using SubspaceClustering
    test_piracies(SubspaceClustering)
end

@testitem "Persistent tasks" begin
    using Aqua: test_persistent_tasks
    using SubspaceClustering
    test_persistent_tasks(SubspaceClustering)
end

@testitem "Undocumented names" begin
    using Aqua: test_undocumented_names
    using SubspaceClustering
    test_undocumented_names(SubspaceClustering)
end
