# Exponential Family Distributions

All the distributions provided by this package are members of the
exponential family of distribution, i.e. they have the follotwing
canonical form:
```math
p(x) = \exp \{ \eta^\top T(x) - A(\eta) + B(x) \}
```
where:
* ``\eta`` is the natural parameter (scalar or vector)
* ``T(x)`` is the sufficient statistic (scalar or vector)
* ``A(\eta)`` is the log-normalizer (scalar)
* ``B(x)`` is the base measure (scalar)

Practically, the package provide the following abstract type
```@docs
ExpFamilyDistribution
```
Each subtype of `ExpFamilyDistribution` implements the following interface:
* [`basemeasure`](@ref)
* [`gradlognorm`](@ref)
* [`kldiv`](@ref)
* [`loglikelihood`](@ref)
* [`lognorm`](@ref)
* [`mean`](@ref)
* [`naturalparam`](@ref)
* [`stats`](@ref)
* [`stdparam`](@ref)
* [`update!`](@ref)

```@docs
basemeasure
gradlognorm
kldiv
lognorm
loglikelihood
mean
naturalparam
stats
stdparam
update!
```

## Multivariate Normal distribution

Likelihood:
```math
\mathcal{N}(\mu, \Sigma) = \frac{1}{(2\pi)^{\frac{D}{2}} | \Sigma
|^{\frac{1}{2}}} \exp \big\{ -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x -
\mu) \big\}
```

Terms of the canonical form:
```math
\begin{aligned}
    \eta &= \begin{bmatrix}
        \Sigma^{-1} \mu \\
        -\frac{1}{2}\text{vec}(\Sigma^{-1})
    \end{bmatrix} \\

    T(x) &= \begin{bmatrix}
        x \\
        \text{vec}(xx^\top)
    \end{bmatrix} \\

    A(\eta) &= \frac{1}{2} \ln |\Sigma| + \frac{1}{2} \mu^\top
        \Sigma^{-1}\mu \\

    B(x) &= -\frac{D}{2} \ln 2\pi \\

    \nabla_{\eta} A(\eta) &= \begin{bmatrix}
        \mu \\
        \Sigma + \mu \mu^\top
    \end{bmatrix}
\end{aligned}
```

```@docs
Normal
NormalDiag
```

## Gamma distribution

Likelihood:
```math
\mathcal{G}(x | \alpha, \beta) = \frac{1}{\Gamma (\alpha)}\beta^{\alpha} x^{\alpha - 1} \exp \{ -\beta x \}
```

Terms of the canonical form:
```math
\begin{aligned}
    \eta &= \begin{bmatrix}
        -\beta  \\
        \alpha
    \end{bmatrix} \\

    T(x) &= \begin{bmatrix}
        x \\
        \ln x
    \end{bmatrix} \\

    A(\eta) &= \ln \Gamma(\alpha) - \alpha \ln \beta \\

    B(x) &= -\ln x \\

    \nabla_{\eta} A(\eta) &= \begin{bmatrix}
        \frac{\alpha}{\beta} \\
        \psi(\alpha) - \ln\beta
    \end{bmatrix}
\end{aligned}
```

```@docs
Gamma
```

## Dirichlet distribution

Likelihood:
```math
\mathcal{D}(x | \alpha) = \frac{\Gamma(\sum_{i=1}^D \alpha_i)}{\prod_{i=1}^{D}\Gamma (\alpha_i)}
    \prod_{i=1}^D x_i^{\alpha - 1}
```

Terms of the canonical form:
```math
\begin{aligned}
    \eta &= \alpha \\

    T(x) &= \ln x \\

    A(\eta) &= \sum_{i=1}^D \ln \Gamma(\alpha_i) - \ln \Gamma(\sum_{i=1}^D \alpha_i) \\

    B(x) &= -\ln x \\

    \nabla_{\eta} A(\eta) &= \begin{bmatrix}
        \psi(\alpha_1) - \psi(\sum_{i=1}^D \alpha_i) \\
        \vdots \\
        \psi(\alpha_D) - \psi(\sum_{i=1}^D \alpha_i) \\
    \end{bmatrix}
\end{aligned}
```

```@docs
Dirichlet
```
