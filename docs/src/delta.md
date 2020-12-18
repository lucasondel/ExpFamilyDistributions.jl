# δ-Distributions

Maximum likelihood (ML) or Maximum A Posteriori inference is a special
case of Bayesian inference where the posterior is assumed to be a
Dirac delta distribution:
```math
\begin{aligned}
\delta_{\mu}(x) &= \delta(x - \mu) = \begin{cases}
    \infty,& \text{if } x - \mu = 0 \\
    0 & \text{otherwise}
\end{cases} \\
\int_x \delta_{\mu}(x) \text{d}x &= 1
\end{aligned}
```

To easily switch between Bayesian inference ML/MAP, the package provide
the *δ-distributions*
```@docs
δDistribution
```
Each subtype `δDistribution` implements partially the Exponential
Family interace:
* [`gradlognorm`](@ref)
* [`mean`](@ref)
* [`update!`](@ref)
where [`gradlognorm`](@ref) returns the expectation of the equivalent
distribution's sufficient statistics, [`mean`](@ref) returns the the
Dirac δ pulse location ``\mu`` and [`update!`](@ref) sets the pulse
location to model of the equivalent distribution.

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

