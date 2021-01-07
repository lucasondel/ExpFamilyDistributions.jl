
using Documenter
using ExpFamilyDistributions
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
using Test


DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

doctest(ExpFamilyDistributions)

#######################################################################
# Normal

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

        s1, s2 = gradlognorm(n, vectorize = false)
        @test size(s1) == (2,)
        @test size(s2) == (2,2)

        @test all(mean(n) .≈ n.μ)

        n2 = Normal{T, 2}()
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        n2 = Normal(stdparam(n, naturalparam(n))...)
        @test kldiv(n, n2) ≈  0

        n2 = Normal{T, 2}()
        update!(n, naturalparam(n2))
        @test all(naturalparam(n) .≈ naturalparam(n2))
    end
end

#######################################################################
# δ-Normal

for T in [Float32, Float64]
    @testset "δNormal ($T)" begin
        n = δNormal(T[1, 2])

        ETx = T[1, 2, 1, 2, 2, 4]
        @test all(gradlognorm(n) .≈ ETx)

        s1, s2 = gradlognorm(n, vectorize = false)
        @test size(s1) == (2,)
        @test size(s2) == (2,2)

        @test all(mean(n) .≈ n.μ)

        prior = Normal{T,2}()
        n2 = δNormal(stdparam(n, naturalparam(prior)))
        @test all(n2.μ .≈ prior.μ)

        prior = Normal{T,2}()
        update!(n, naturalparam(prior))
        @test all(n.μ .≈ prior.μ)
    end
end

#######################################################################
# NormalDiag

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

        s1, s2 = gradlognorm(n, vectorize = false)
        @test size(s1) == (2,)
        @test size(s2) == (2,)

        @test all(mean(n) .≈ n.μ)

        n2 = NormalDiag{T, 2}()
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        n2 = NormalDiag(stdparam(n, naturalparam(n))...)
        @test kldiv(n, n2) ≈  0

        update!(n, naturalparam(n2))
        @test all(naturalparam(n) .≈ naturalparam(n2))
    end
end

#######################################################################
# δ-NormalDiag

for T in [Float32, Float64]
    @testset "δNormalDiag ($T)" begin
        n = δNormalDiag(T[1, 2])

        ETx = T[1, 2, 1, 4]
        @test all(gradlognorm(n) .≈ ETx)

        s1, s2 = gradlognorm(n, vectorize = false)
        @test size(s1) == (2,)
        @test size(s2) == (2,)

        @test all(mean(n) .≈ n.μ)

        prior = NormalDiag{T,2}()
        n2 = δNormalDiag(stdparam(n, naturalparam(prior)))
        @test all(n2.μ .≈ prior.μ)

        prior = NormalDiag{T,2}()
        update!(n, naturalparam(prior))
        @test all(n.μ .≈ prior.μ)
    end
end

#######################################################################
# Gamma

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

        s1, s2 = gradlognorm(g, vectorize = false)
        @test size(s1) == ()
        @test size(s2) == ()

        @test all(mean(g) .≈ g.α / g.β)

        g2 = Gamma{T}()
        @test kldiv(g, g) ≈  0
        @test kldiv(g, g2) >= 0
        @test kldiv(g2, g) >= 0

        g2 = Gamma{T}(stdparam(g, naturalparam(g))...)
        @test kldiv(g, g2) ≈  0

        update!(g, naturalparam(g2))
        @test all(naturalparam(g) .≈ naturalparam(g2))
    end
end

#######################################################################
# δ-Gamma

for T in [Float32, Float64]
    @testset "δGamma ($T)" begin
        @test_throws ArgumentError δGamma{T}(-1)

        g = δGamma{T}(1/2)

        ETx = T[1/2, log(1/2)]
        @test all(gradlognorm(g) .≈ ETx)

        s1, s2 = gradlognorm(g, vectorize = false)
        @test size(s1) == ()
        @test size(s2) == ()

        @test all(mean(g) .≈ 1/2)

        g2 = Gamma{T}(0.9, 0.9)
        @test_throws ArgumentError update!(g, naturalparam(g2))

        prior = Gamma{T}()
        g2 = δGamma{T}(stdparam(g, naturalparam(prior)))
        @test all(g2.μ .≈ (prior.α-1)/prior.β)

        g2 = Gamma{T}()
        update!(g, naturalparam(g2))
        @test all(g.μ .≈ (g2.α-1)/g2.β)
    end
end

#######################################################################
# Dirichlet

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

        s = gradlognorm(d, vectorize = false)
        @test size(s) == (3,)

        @test all(mean(d) .≈ d.α / sum(d.α))

        d2 = Dirichlet{T,3}()
        @test kldiv(d, d) ≈  0
        @test kldiv(d, d2) >= 0
        @test kldiv(d2, d) >= 0

        d2 = Dirichlet(stdparam(d, naturalparam(d)))
        @test kldiv(d, d2) ≈  0

        update!(d, naturalparam(d2))
        @test all(naturalparam(d) .≈ naturalparam(d2))
    end
end

#######################################################################
# δ-Dirichlet

for T in [Float32, Float64]
    @testset "δDirichlet ($T)" begin
        @test_throws ArgumentError δDirichlet(T[1, 2, 3])

        d = δDirichlet(T[1/3, 1/3, 1/3])

        ETx = log.(d.μ)
        @test all(gradlognorm(d) .≈ ETx)

        s = gradlognorm(d, vectorize = false)
        @test size(s) == (3,)

        @test all(mean(d) .≈ d.μ)

        d2 = Dirichlet(T[0.8, 0.1, 0.1])
        @test_throws ArgumentError update!(d, naturalparam(d2))

        d2 = Dirichlet(T[3, 2, 2])
        update!(d, naturalparam(d2))
        @test all(d.μ .≈ (d2.α .- 1) / sum(d2.α .- 1))
    end
end

#######################################################################
# Wishart

for T in [Float32, Float64]
    @testset "Wishart($T)" begin
        X = Symmetric(T[1 0.5; 0.5 2])

        D = 2
        W = Symmetric(Matrix{T}(I, 2, 2))
        v = 2
        w = Wishart(W, 2)

        η = vcat(vec(-.5*inv(W)), v/2)
        @test eltype(naturalparam(w)) == T
        @test all(naturalparam(w) .≈ η)

        TX = T[X..., logdet(X)]
        @test all(stats(w, X) .≈ TX)

        A = .5*( v*logdet(W) + v*D*log(2) ) + sum([loggamma((v+1-i)/2) for i in 1:D])
        @test lognorm(w) ≈ T(A)

        B = -.5*( (D-1)*logdet(X) + .5*D*(D-1)*log(π) )
        @test basemeasure(w, X) ≈ B

        ETX = vcat(vec(v*W), sum([digamma((v+1-i)/2) for i in 1:D]) + D*log(2) + logdet(W))
        @test eltype(gradlognorm(w)) == T
        @test all(gradlognorm(w) .≈ ETX)

        s = gradlognorm(w, vectorize = false)
        @test length(s) == 2
        @test all(s[1] .≈ v*W)
        @test s[2] .≈ sum([digamma((v+1-i)/2) for i in 1:D]) + D*log(2) + logdet(W)

        @test all(mean(w) .≈ v*W)

        w2 = Wishart{T,2}()
        @test kldiv(w, w) ≈  0
        @test kldiv(w, w2) >= 0
        @test kldiv(w2, w) >= 0

        w2 = Wishart(stdparam(w, naturalparam(w))...)
        @test kldiv(w, w2) ≈  0

        update!(w, naturalparam(w2))
        @test all(naturalparam(w) .≈ naturalparam(w2))
    end
end

#######################################################################
# δ-Wishart

for T in [Float32, Float64]
    @testset "δWishart ($T)" begin
        W = Symmetric(T[1 0.5; 0.5 2])
        D = 2

        w = δWishart{T,D}()
        @test all(w.μ .≈ Matrix{T}(I,D,D))

        w = δWishart(W)

        ETX = vcat(vec(w.μ), logdet(w.μ))
        @test eltype(gradlognorm(w)) == T
        @test all(gradlognorm(w) .≈ ETX)

        s = gradlognorm(w, vectorize = false)
        @test length(s) == 2
        @test all(s[1] .≈ w.μ)
        @test s[2] .≈ logdet(w.μ)

        @test all(mean(w) .≈ w.μ)

        w2 = Wishart{T,2}()
        update!(w, naturalparam(w2))
        @test all(w.μ .≈ w2.v*w2.W)
    end
end
