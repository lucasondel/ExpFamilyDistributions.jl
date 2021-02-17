# δ-Distributions

Maximum likelihood (ML) and Maximum A Posteriori inference are special
cases of Bayesian inference where the posterior is assumed to be a
Dirac δ distribution:
```math
\begin{aligned}
\delta_{\mu}(x) &= \delta(x - \mu) = \begin{cases}
    \infty,& \text{if } x - \mu = 0 \\
    0 & \text{otherwise}
\end{cases} \\
\int_x \delta_{\mu}(x) \text{d}x &= 1
\end{aligned}
```

To easily switch between Bayesian inference ML/MAP, the package
provides the *δ-distributions*, i.e. Dirac δ dsistribution wrap around
an "equivalent" distribution member of the exponential family.
```@docs
δDistribution
```
Each subtype of `δDistribution` implements partially the Exponential
Family interface:
* [`gradlognorm`](@ref)
* [`mean`](@ref)
* [`sample`](@ref)
* [`splitgrad`](@ref)
* [`update!`](@ref).

## δ-Normal distribution

```@docs
δNormal
δNormalDiag
```

## δ-Gamma distribution

```@docs
δGamma
```

## δ-Dirichlet distribution

```@docs
δDirichlet
```

## δ-Wishart distribution

```@docs
δWishart
```

