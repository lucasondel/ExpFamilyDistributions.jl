# Distributions

We proved here the list of distribution implemented by the package.

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
        -\frac{1}{2}\text{vec}(\text{diag}(\Sigma^{-1})) \\
        -\text{vec}(\text{tril}(\Sigma^{-1}))
    \end{bmatrix} \\

    T(x) &= \begin{bmatrix}
        x \\
        \text{vec}(\text{diag}(xx^\top)) \\
        \text{vec}(\text{tril}(xx^\top))
    \end{bmatrix} \\

    A(\eta) &= \frac{1}{2} \ln |\Sigma| + \frac{1}{2} \mu^\top
        \Sigma^{-1}\mu \\

    B(x) &= -\frac{D}{2} \ln 2\pi \\

    \nabla_{\eta} A(\eta) &= \begin{bmatrix}
        \mu \\
        \text{vec}(\Sigma + \mu \mu^\top)
    \end{bmatrix}
\end{aligned}
```
where $\text{tril}$ is a function that returns the lower-triangular
part of a matrix (diagonal not included).

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

!!! note
    In practice, the Gamma structure in the package represents the
    distribution of `D` independent Gamma distributed variables.

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

## Wishart distribution

Likelihood:
```math
\mathcal{W}(X | W, v) = B(W, v)|X|^{\frac{(v-D-1)}{2}} \exp \bigg\{
    -\frac{1}{2} \text{tr}(W^{-1}X) \bigg\} \\
B(W,v) = |W|^{-\frac{v}{2}}\bigg( 2^{\frac{vD}{2}} \pi^{\frac{D(D-1)}{4}}
    \prod_{i=1}^D \Gamma \big( \frac{v+1-i}{2} \big) \bigg)^{-1}
```
where $X$ and $W$ are $D \times D$ symmetric positive definite matrices.

Terms of the canonical form:
```math
\begin{aligned}
    \eta &= \begin{bmatrix}
        \text{vec}(-\frac{1}{2} W^{-1}) \\
        \frac{v}{2}
    \end{bmatrix}\\

    T(x) &= \begin{bmatrix}
        \text{vec}(\text{diag}(X)) \\
        \text{vec}(\text{tril}(X)) \\
        \ln |X|
    \end{bmatrix} \\

    A(\eta) &= \frac{v}{2} \ln |W| + \frac{vD}{2} \ln 2
        + \sum_{i=1}^D \ln \Gamma \big( \frac{v+1-i}{2} \big) \\

    B(x) &= -\frac{(D-1)}{2} \ln |X| - \frac{D(D-1)}{4} \ln \pi     \\

    \nabla_{\eta} A(\eta) &= \begin{bmatrix}
        \text{vec}(vW) \\
        \ln |W| + D \ln 2 + \sum_{i=1}^D \psi \big( \frac{v+1-i}{2} \big)
    \end{bmatrix}
\end{aligned}
```

```@docs
Wishart
```
