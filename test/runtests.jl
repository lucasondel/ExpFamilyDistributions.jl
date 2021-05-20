# SPDX-License-Identifier: MIT

using Documenter
using ExpFamilyDistributions
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
using Test


DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)

doctest(ExpFamilyDistributions)

#######################################################################
# DefaultParameter

for T in [Float32, Float64]
    @testset "DefaultParameter ($T)" begin
        ξ = T[1.0, 2.0, 3.0]
        p = DefaultParameter(ξ)
        @test typeof(p).mutable

        @test all(naturalform(p) .≈ ξ)
        @test all(realform(p) .≈ ξ)
        @test all(jacobian(p) .≈ Matrix{T}(I, length(ξ), length(ξ)))
    end
end

#######################################################################
# Normal

for T in [Float32, Float64]
    @testset "Normal ($T)" begin
        n = Normal(T[1, 2], Symmetric(T[2 0; 0 3]))

        η = T[1/2, 2/3, -1/4, -1/6, 0]
        @test all(naturalform(n.param) .≈ η)

        x = T[2, 3]
        Tx = T[2, 3, 4, 9, 6]
        @test all(stats(n, x) .≈ Tx)

        A = .5 * log(det(n.Σ)) + .5 * dot(n.μ, inv(n.Σ), n.μ)
        @test typeof(lognorm(n)) == T
        @test lognorm(n) ≈ T(A)

        @test basemeasure(n, x) ≈ T(-log(2π))

        ETx = T[1, 2, 3, 7, 2]
        @test all(gradlognorm(n) .≈ ETx)

        s1, s2, s3 = splitgrad(n, gradlognorm(n))
        @test size(s1) == (2,)
        @test size(s2) == (2,)
        @test size(s3) == (1,)

        n2 = Normal(zeros(T, 2), Matrix{T}(I, 2, 2))
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        n2 = Normal(stdparam(n, naturalform(n.param))...)
        @test kldiv(n, n2) ≈  0

        @test length(sample(n2, 1)) == 1
        @test length(sample(n2, 10)) == 10
        @test eltype(sample(n2, 2)[1]) == T
    end
end

#######################################################################
# NormalDiag

for T in [Float32, Float64]
    @testset "NormalDiag ($T)" begin
        n = NormalDiag(T[1, 2], T[2, 2])

        η = T[.5, 1, -.25, -.25]
        @test all(naturalform(n.param) .≈ η)

        x = T[2, 3]
        Tx = T[2, 3, 4, 9]
        @test all(stats(n, x) .≈ Tx)

        A = .5 * log(det(n.Σ)) + .5 * dot(n.μ, inv(n.Σ), n.μ)
        @test typeof(lognorm(n)) == T
        @test lognorm(n) ≈ T(A)

        @test basemeasure(n, x) ≈ T(-log(2π))

        ETx = T[1, 2, 3, 6]
        @test all(gradlognorm(n) .≈ ETx)

        s1, s2 = splitgrad(n, gradlognorm(n))
        @test size(s1) == (2,)
        @test size(s2) == (2,)

        n2 = NormalDiag(zeros(T, 2), ones(T, 2))
        @test kldiv(n, n) ≈  0
        @test kldiv(n, n2) >= 0
        @test kldiv(n2, n) >= 0

        n2 = NormalDiag(stdparam(n, naturalform(n.param))...)
        @test kldiv(n, n2) ≈  0

        @test length(sample(n, 1)) == 1
        @test length(sample(n, 10)) == 10
        @test eltype(sample(n, 2)[1]) == T
    end
end

#######################################################################
# Gamma

for T in [Float32, Float64]
    @testset "Gamma ($T)" begin
        g = Gamma(T(1), T(2))

        η = T[-2, 1]
        @test all(naturalform(g.param) .≈ η)

        x = T(2)
        Tx = T[x, log(x)]
        @test all(stats(g, x) .≈ Tx)

        A = loggamma(1) - log(2)
        @test lognorm(g) ≈ T(A)

        @test basemeasure(g, x) ≈ -log(x)

        ETx = T[1/2, digamma(1) - log(2)]
        @test all(gradlognorm(g) .≈ ETx)

        s1, s2 = splitgrad(g, gradlognorm(g))
        @test size(s1) == ()
        @test size(s2) == ()

        g2 = Gamma(T(1), T(1))
        @test kldiv(g, g) ≈  0
        @test kldiv(g, g2) >= 0
        @test kldiv(g2, g) >= 0

        g2 = Gamma(stdparam(g, naturalform(g.param))...)
        @test kldiv(g, g2) ≈  0

        @test length(sample(g, 1)) == 1
        @test length(sample(g, 10)) == 10
    end
end

#######################################################################
# Dirichlet

for T in [Float32, Float64]
    @testset "Dirichlet ($T)" begin
        d = Dirichlet(T[1, 2, 3])

        η = T[1, 2, 3]
        @test all(naturalform(d.param) .≈ η)

        x = T[1, 2, 3]
        Tx = T[log(1), log(2), log(3)]
        @test all(stats(d, x) .≈ Tx)

        A = sum(loggamma.(d.α)) - loggamma(sum(d.α))
        @test lognorm(d) ≈ T(A)

        @test basemeasure(d, x) ≈ -log.(x)

        ETx = digamma.(d.α) .- digamma(sum(d.α))
        @test all(gradlognorm(d) .≈ ETx)

        s = splitgrad(d, gradlognorm(d))
        @test size(s) == (3,)

        d2 = Dirichlet(ones(T, 3))
        @test kldiv(d, d) ≈  0
        @test kldiv(d, d2) >= 0
        @test kldiv(d2, d) >= 0

        d2 = Dirichlet(stdparam(d, naturalform(d.param))...)
        @test kldiv(d, d2) ≈  0

        @test length(sample(d, 1)) == 1
        @test length(sample(d, 10)) == 10
    end
end

#######################################################################
# Wishart

for T in [Float32, Float64]
    @testset "Wishart($T)" begin
        X = Symmetric(T[1 0.5; 0.5 2])

        D = T(2)
        W = Symmetric(Matrix{T}(I, 2, 2))
        v = T(2)
        w = Wishart(W, v)

        invW = inv(W)
        η = vcat(-.5 * diag(invW), vec_tril(invW), v/2)
        @test eltype(naturalform(w.param)) == T
        @test all(naturalform(w.param) .≈ η)

        TX = T[diag(X)..., vec_tril(X)..., logdet(X)]
        @test all(stats(w, X) .≈ TX)

        A = .5*( v*logdet(W) + v*D*log(2) ) + sum([loggamma((v+1-i)/2) for i in 1:D])
        @test lognorm(w) ≈ T(A)

        B = -.5*( (D-1)*logdet(X) + .5*D*(D-1)*log(π) )
        @test basemeasure(w, X) ≈ B

        vW = v*W
        ETX = vcat(diag(vW), vec_tril(vW), sum([digamma(T(0.5)*(v+1-i)) for i in 1:D]) + D*T(log(2)) + logdet(W))
        @test eltype(gradlognorm(w)) == T
        @test all(gradlognorm(w) .≈ ETX)

        s = splitgrad(w, gradlognorm(w))
        @test length(s) == 3
        @test all(s[1] .≈ diag(vW))
        @test all(s[2] .≈ vec_tril(vW))
        @test s[3] .≈ sum([digamma((v+1-i)/2) for i in 1:D]) + D*log(2) + logdet(W)

        w2 = Wishart(Matrix{T}(I, 2, 2), 2)
        @test kldiv(w, w) ≈  0
        @test kldiv(w, w2) >= 0
        @test kldiv(w2, w) >= 0

        w2 = Wishart(stdparam(w, naturalform(w.param))...)
        @test kldiv(w, w2) ≈  0

        @test length(sample(w, 1)) == 1
        @test length(sample(w, 10)) == 10
    end
end

