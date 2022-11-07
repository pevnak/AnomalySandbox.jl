module AnomalySandbox

using Flux
using StatsBase
using GaussianMixtures

include("data.jl")
include("vae.jl")
include("gan.jl")
include("gmm.jl")

end
