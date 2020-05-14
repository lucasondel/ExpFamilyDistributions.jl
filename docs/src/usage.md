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
| `stdparam(p)`           | Returns the standard parameters of `p`                 |
| `update!(p, natparam)`  | Update the parameters of `p` given a vector of natural parameters |

In the following, we show a basic demonstration of the interface using
a (multivariate) Normal distribution.
```juliashowcase
julia> using ExpFamilyDistributions

julia> p = Normal{2}()
Normal{2}
  μ = [0.0, 0.0]
  Σ = [1.0 0.0; 0.0 1.0]

julia> q = Normal([1., -1.], [2 0.5; 0.5 1])
Normal{2}
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
1.5699519734919276

julia> lognorm(p), lognorm(q)
(-0.0, 1.0729051693652147)

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

julia> stdparam(p)
(μ = [0.0, 0.0], Σ = [1.0 0.0; 0.0 1.0])

julia> update!(q, naturalparam(p))
julia> q
Normal{2}
  μ = [0.0, 0.0]
  Σ = [1.0 0.0; 0.0 1.0]
```

## Supported distributions

| Julia type            | Description                              |
|:--------------------- |:---------------------------------------- |
| `Normal{D}`           | `D`-multivariate normal distribution     |
| `Gamma{D}`            | `D` independent Gamma distributions      |
| `PolyaGamma{D}`       | `D` independent PolyaGamma distributions |

!!! warning
    The PolyaGamma distribution does not implement the basemeasure as
    it involves an infinite sum. Therefore, calling the `basemeasure`
    method with an instance of a `PolyaGamma{D}` will raise an error.

