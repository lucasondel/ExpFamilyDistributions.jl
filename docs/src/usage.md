# Usage

## ExpFamilyDistribution interface

Every distribution provided by the ExpFamilyDistributions package
inherits from the supertype `ExpFamilyDistibution` and implements the
following interface:

| Methods                 | Brief description  |
|:----------------------- |:------------------- |
| `basemeasure(p, X)`     | Returns the base measure of `p` for each column of `X` |
| `gradlognorm(p)`        | Returns the gradient of the log-normalizer of `p`      |
| `kldiv(q, p)`           | Compute the Kullback-Leibler divergence between `q` and `p` |
| `lognorm(p)`            | Returns the value of the log-normalizer of `p`         |
| `mean(p)`               | Returns the mean of the distribution `p`               |
| `naturalparam(p)`       | Returns the vector of natural parameters of `p`        |
| `stats(p, X)`           | Returns the matrix of sufficient statistics            |
| `update!(p, natparam)`  | Update the parameters of `p` given a vector of natural parameters |

In the following, we show a basic demonstration of the interface using
a (multivariate) Normal distribution.
```juliashowcase
julia> using ExpFamilyDistributions

julia> p = Normal{Float64, 2}()
Normal{Float64,2}
  μ = [0.0, 0.0]
  Σ = [1.0 0.0; 0.0 1.0]

julia> q = Normal([1., -1.], [2 0.5; 0.5 1])
Normal{Float64,2}
  μ = [1.0, -1.0]
  Σ = [2.0 0.5; 0.5 1.0]

julia> basemeasure(p, ones(2, 5))
5-element Array{Float64,1}:
 -1.8378770664093453
 -1.8378770664093453
 -1.8378770664093453
 -1.8378770664093453
 -1.8378770664093453

julia> gradlognorm(p)
6-element Array{Float64,1}:
 0.0
 0.0
 1.0
 0.0
 0.0
 1.0

julia> kldiv(q, p)
1.2201921060322882

julia> lognorm(p), lognorm(q)
(-0.0, 1.4226650368248541)

julia> mean(p)
2-element Array{Float64,1}:
 0.0
 0.0

julia> naturalparam(p)
6-element Array{Float64,1}:
  0.0
  0.0
 -0.5
 -0.0
 -0.0
 -0.5

julia> stats(p, ones(2, 5))
6×5 Array{Float64,2}:
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0

julia> update!(q, naturalparam(p))
julia> q
Normal{Float64,2}
  μ = [0.0, 0.0]
  Σ = [1.0 0.0; 0.0 1.0]
```

## Supported distributions

| Julia type            | Description                              |
|:--------------------- |:---------------------------------------- |
| `Dirichlet{T, D}`     | `D`-dimensional Dirichlet distribution   |
| `Gamma{T, D}`         | `D` independent Gamma distributions      |
| `Normal{T, D}`        | `D`-multivariate normal distribution     |
| `NormalDiag{T, D}`    | `D`-multivariate normal distribution with diagonal covariance matrix |

For all the distributions, the `T<:AbstractFloat` is the type of the
distribution parameters. Note that you cannot compute the KL divergence
(`kldiv`) with distributions having different `T`.

