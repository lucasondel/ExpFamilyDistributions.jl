
using Documenter
using ExpFamilyDistributions
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
using Test


DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

doctest(ExpFamilyDistributions)

for T in [Float32, Float64]
    @testset "Normal ($T)" begin
        n = Normal(T[1, 2], Symmetric(T[2 0; 0 2]))

        η = T[.5, 1, -.25, 0, 0, -.25]
        @test all(naturalparam(n) .≈ η)

        x = T[2, 3]
        Tx = T[2, 3, 4, 6, 6, 9]
        @test all(stats(n, x) .≈ Tx)

        A = .5 * log(det(n.Σ)) + .5 * dot(n.μ, inv(n.Σ), n.μ)
        @test lognorm(n) ≈ T(A)

        @test basemeasure(n, x) ≈ T(-log(2π))

        ETx = T[1, 2, 3, 2, 2, 6]
        @test all(gradlognorm(n) .≈ ETx)

        @test all(mean(n) .≈ n.μ)

        n2 = Normal{T, 2}()
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        update!(n, naturalparam(n2))
        @test all(naturalparam(n) .≈ naturalparam(n2))
    end
end

for T in [Float32, Float64]
    @testset "NormalDiag ($T)" begin
        n = NormalDiag(T[1, 2], T[2, 2])

        η = T[.5, 1, -.25, -.25]
        @test all(naturalparam(n) .≈ η)

        x = T[2, 3]
        Tx = T[2, 3, 4, 9]
        @test all(stats(n, x) .≈ Tx)

        A = .5 * log(det(n.Σ)) + .5 * dot(n.μ, inv(n.Σ), n.μ)
        @test lognorm(n) ≈ T(A)

        @test basemeasure(n, x) ≈ T(-log(2π))

        ETx = T[1, 2, 3, 6]
        @test all(gradlognorm(n) .≈ ETx)

        @test all(mean(n) .≈ n.μ)

        n2 = NormalDiag{T, 2}()
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        update!(n, naturalparam(n2))
        @test all(naturalparam(n) .≈ naturalparam(n2))
    end
end

for T in [Float32, Float64]
    @testset "Gamma ($T)" begin
        g = Gamma{T}(1, 2)

        η = T[-2, 1]
        @test all(naturalparam(g) .≈ η)

        x = T(2)
        Tx = T[x, log(x)]
        @test all(stats(g, x) .≈ Tx)

        A = loggamma(1) - log(2)
        @test lognorm(g) ≈ T(A)

        @test basemeasure(g, x) ≈ -log(x)

        ETx = T[1/2, digamma(1) - log(2)]
        @test all(gradlognorm(g) .≈ ETx)

        @test all(mean(g) .≈ g.α / g.β)

        g2 = Gamma{T}()
        @test kldiv(g, g) ≈  0
        @test kldiv(g, g2) >= 0
        @test kldiv(g2, g) >= 0

        update!(g, naturalparam(g2))
        @test all(naturalparam(g) .≈ naturalparam(g2))
    end
end

for T in [Float32, Float64]
    @testset "Dirichlet ($T)" begin
        d = Dirichlet(T[1, 2, 3])

        η = T[1, 2, 3]
        @test all(naturalparam(d) .≈ η)

        x = T[1, 2, 3]
        Tx = T[log(1), log(2), log(3)]
        @test all(stats(d, x) .≈ Tx)

        A = sum(loggamma.(d.α)) - loggamma(sum(d.α))
        @test lognorm(d) ≈ T(A)

        @test basemeasure(d, x) ≈ -log.(x)

        ETx = digamma.(d.α) .- digamma(sum(d.α))
        @test all(gradlognorm(d) .≈ ETx)

        @test all(mean(d) .≈ d.α / sum(d.α))

        d2 = Dirichlet{T,3}()
        @test kldiv(d, d) ≈  0
        @test kldiv(d, d2) >= 0
        @test kldiv(d2, d) >= 0

        update!(d, naturalparam(d2))
        @test all(naturalparam(d) .≈ naturalparam(d2))
    end
end
