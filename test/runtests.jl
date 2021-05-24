# SPDX-License-Identifier: MIT

using CUDA
using Documenter
using ExpFamilyDistributions
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
using Test

CUDA.allowscalar(false)

const types = [Float32, Float64]
const arraytypes = [Array]

if CUDA.functional()
    @info "gpu available testing on CuArray"
    CUDA.versioninfo()
    push!(arraytypes, CuArray)
end

for T in types

    @testset "PD matrix operations ($T)" begin
        M = T[1 0.5; 0.5 1]
        @test all(inv(M) .≈ pdmat_inverse(M))
        @test all(logdet(M) .≈ pdmat_logdet(M))
    end

    for AT in arraytypes

        @testset "DefaultParameter ($AT | $T)" begin
            ξ = AT(T[1.0, 2.0, 3.0])
            p = DefaultParameter(ξ)

            @test all(naturalform(p) .≈ ξ)
            @test all(realform(p) .≈ ξ)
            @test all(Array(jacobian(p)) .≈ Matrix{T}(I, length(ξ), length(ξ)))
            p̄ = reallocate(p, Tuple)
            @test typeof(realform(p̄)) == Tuple{T, T, T}
        end

        @testset "Normal ($AT | $T)" begin
            n = Normal(AT(T[1, 2]), AT(T[2 0; 0 3]))
            @test typeof(n).mutable

            η = AT(T[1/2, 2/3, -1/4, -1/6, 0])
            @test all(naturalform(n.param) .≈ η)

            x = AT(T[2, 3])
            Tx = AT(T[2, 3, 4, 9, 6])
            @test all(stats(n, x) .≈ Tx)

            A = .5 * pdmat_logdet(n.Σ) + .5 * dot(n.μ, pdmat_inverse(n.Σ) * n.μ)
            @test typeof(lognorm(n)) == T
            @test lognorm(n) ≈ T(A)

            @test basemeasure(n, x) ≈ T(-log(2π))

            ETx = AT(T[1, 2, 3, 7, 2])
            @test all(gradlognorm(n) .≈ ETx)

            s1, s2, s3 = splitgrad(n, gradlognorm(n))
            @test size(s1) == (2,)
            @test size(s2) == (2,)
            @test size(s3) == (1,)

            n2 = Normal(AT(zeros(T, 2)), AT(Matrix{T}(I, 2, 2)))
            @test isapprox(kldiv(n, n), 0; atol = (T == Float32 ? 1e-6 : 1e-16))
            @test kldiv(n, n2) >= 0
            @test kldiv(n2, n) >= 0

            n2 = Normal(stdparam(n, naturalform(n.param))...)
            @test isapprox(kldiv(n, n2),  0; atol = (T == Float32 ? 1e-6 : 1e-16))

            @test size(sample(n2, 1)) == (2, 1)
            @test size(sample(n2, 10)) == (2, 10)
            @test eltype(sample(n2, 2)) == T
        end

        @testset "NormalDiag ($AT | $T)" begin
            n = NormalDiag(AT(T[1, 2]), AT(T[2, 2]))
            @test typeof(n).mutable

            η = AT(T[.5, 1, -.25, -.25])
            @test all(naturalform(n.param) .≈ η)

            x = AT(T[2, 3])
            Tx = AT(T[2, 3, 4, 9])
            @test all(stats(n, x) .≈ Tx)

            A = .5 * pdmat_logdet(n.Σ) + .5 * dot(n.μ, pdmat_inverse(n.Σ) * n.μ)
            @test typeof(lognorm(n)) == T
            @test lognorm(n) ≈ T(A)

            @test basemeasure(n, x) ≈ T(-log(2π))

            ETx = AT(T[1, 2, 3, 6])
            @test all(gradlognorm(n) .≈ ETx)

            s1, s2 = splitgrad(n, gradlognorm(n))
            @test size(s1) == (2,)
            @test size(s2) == (2,)

            n2 = NormalDiag(AT(zeros(T, 2)), AT(ones(T, 2)))
            @test kldiv(n, n) ≈  0
            @test kldiv(n, n2) >= 0
            @test kldiv(n2, n) >= 0

            n2 = NormalDiag(stdparam(n, naturalform(n.param))...)
            @test kldiv(n, n2) ≈  0

            @test size(sample(n, 10)) == (2, 10)
            @test eltype(sample(n, 2)) == T
       end

        @testset "Gamma ($AT | $T)" begin
            g = Gamma(AT(T[1, 2]), AT(T[2, 3]))
            @test typeof(g).mutable

            η = AT(T[-2, -3, 1, 2])
            @test all(naturalform(g.param) .≈ η)

            x = AT(T[1, 2])
            Tx = vcat(x, log.(x))
            @test all(stats(g, x) .≈ Tx)

            A = loggamma(1) + loggamma(2) - 1*log(2) - 2*log(3)
            @test lognorm(g) ≈ T(A)

            @test basemeasure(g, x) ≈ sum(-log.(x))

            ETx = AT(T[1/2, 2/3, digamma(1) - log(2), digamma(2) - log(3)])
            @test all(gradlognorm(g) .≈ ETx)

            s1, s2 = splitgrad(g, gradlognorm(g))
            @test length(s1) == 2
            @test length(s2) == 2

            g2 = Gamma(AT(T[1, 1]), AT(T[1, 1]))
            @test kldiv(g, g) ≈  0
            @test kldiv(g, g2) >= 0
            @test kldiv(g2, g) >= 0

            g2 = Gamma(stdparam(g, naturalform(g.param))...)
            @test kldiv(g, g2) ≈  0

            @test size(sample(g, 10)) == (2,10)
        end

        @testset "Dirichlet ($AT | $T)" begin
            d = Dirichlet(AT(T[1, 2, 3]))
            @test typeof(d).mutable

            η = AT(T[1, 2, 3])
            @test all(naturalform(d.param) .≈ η)

            x = AT(T[1, 2, 3])
            Tx = AT(T[log(1), log(2), log(3)])
            @test all(stats(d, x) .≈ Tx)

            A = sum(loggamma_dot(d.α)) - loggamma(sum(d.α))
            @test lognorm(d) ≈ T(A)

            @test basemeasure(d, x) ≈ -log.(x)

            ETx = digamma.(d.α) .- digamma(sum(d.α))
            @test all(gradlognorm(d) .≈ ETx)

            s = splitgrad(d, gradlognorm(d))
            @test size(s) == (3,)

            d2 = Dirichlet(AT(ones(T, 3)))
            @test kldiv(d, d) ≈  0
            @test kldiv(d, d2) >= 0
            @test kldiv(d2, d) >= 0

            d2 = Dirichlet(stdparam(d, naturalform(d.param))...)
            @test kldiv(d, d2) ≈  0

            @test size(sample(d, 10)) == (3, 10)
        end

        @testset "Wishart($AT | $T)" begin
            X = AT(AT(T[1 0.5; 0.5 2]))

            D = T(2)
            W = AT(Matrix{T}(I, 2, 2))
            v = T(2)
            w = Wishart(W, v)
            @test typeof(w).mutable

            invW = pdmat_inverse(W)
            η = vcat(-.5 * diag(invW), vec_tril(invW), v/2)
            @test eltype(naturalform(w.param)) == T
            @test all(naturalform(w.param) .≈ η)

            TX = vcat(diag(X), vec_tril(X), pdmat_logdet(X))
            @test all(stats(w, X) .≈ TX)

            A = .5*( v*pdmat_logdet(W) + v*D*log(2) ) + sum([loggamma((v+1-i)/2) for i in 1:D])
            @test lognorm(w) ≈ T(A)

            B = -.5*( (D-1)*pdmat_logdet(X) + .5*D*(D-1)*log(π) )
            @test basemeasure(w, X) ≈ B

            vW = v*W
            ETX = vcat(diag(vW), vec_tril(vW), sum([digamma(T(0.5)*(v+1-i)) for i in 1:D]) + D*T(log(2)) + pdmat_logdet(W))
            @test eltype(gradlognorm(w)) == T
            @test all(gradlognorm(w) .≈ ETX)

            s = splitgrad(w, gradlognorm(w))
            @test length(s) == 3
            @test all(s[1] .≈ diag(vW))
            @test all(s[2] .≈ vec_tril(vW))
            @test s[3] .≈ sum([digamma((v+1-i)/2) for i in 1:D]) + D*log(2) + pdmat_logdet(W)

            w2 = Wishart(AT(Matrix{T}(I, 2, 2)), 2)
            @test kldiv(w, w) ≈  0
            @test kldiv(w, w2) >= 0
            @test kldiv(w2, w) >= 0

            w2 = Wishart(stdparam(w, naturalform(w.param))...)
            @test kldiv(w, w2) ≈  0

            @test size(sample(w, 10)) == (2, 2, 10)
        end
    end
end

# Finally, test the documentation
DocMeta.setdocmeta!(ExpFamilyDistributions, :DocTestSetup,
                    :(using ExpFamilyDistributions), recursive = true)
doctest(ExpFamilyDistributions)

