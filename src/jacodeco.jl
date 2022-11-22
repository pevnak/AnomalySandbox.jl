using FluxExtensions, LinearAlgebra
FluxExtensions.log_normal(x, μ, σ2::T, d::Int) where {T<:Real}  = (d > 0) ? - sum((@. ((x - μ)^2) ./ σ2), dims=1)/2 .- d * (log.(σ2) + log(2π) )/2 : 0

"""
	LinearAlgebra.logabsdet(f, x::Vector)	

	determinant of a jacobian of `f` at point `x`
"""
function LinearAlgebra.logabsdet(f, x::Vector)	
	Σ = Flux.data(Flux.Tracker.jacobian(f, x))
	S = svd(Σ)
	mask = [S.S .!= 0]
	2*sum(log.(abs.(S.S[mask])))
end

jacodeco(model, x::Vector, z::Vector) = sum(log_normal(z)) + sum(logpx(model, x, z)) - logabsdet(model.f, z)
function jacodeco(m, x, z)
	@assert nobs(x) == nobs(z)
	[jacodeco(m, getobs(x, i), getobs(z, i)) for i in 1:nobs(x)]
end
jacodeco(m, x) = jacodeco(m, x, m.g(x))

function correctedjacodeco(model, x::Vector, z::Vector)
	Σ = Flux.data(Flux.Tracker.jacobian(model.f, z))
	S = svd(Σ)
	J = inv(S)
	logd = 2*sum(log.(S.S .+ 1f-6))
	logpz = log_normal(z, 0, 1 .+ J * transpose(J))[1]
	logpx = log_normal(x, model.f(z), 1, length(x) - length(z))[1]
	logpz + logpx - logd
end

function correctedjacodeco(m, x, z)
	@assert nobs(x) == nobs(z)
	[correctedjacodeco(m, getobs(x, i), getobs(z, i)) for i in 1:nobs(x)]
end

correctedjacodeco(m, x) = correctedjacodeco(m, x, manifoldz(model, x))

"""
	manifoldz(model, x, steps = 100, rand_init = false)

	find the point in the latent space `z` such that it minimizes the 
	reconstruction error. This means that it finds the point on manifold 
	defined by `model.g(z)` and x
"""
function manifoldz(model, x, steps = 100, z = Flux.data(model.g(x)))
	z = param(z)
	ps = Flux.Tracker.Params([z])
	li = Flux.data(mean(logpx(model, x, z)))
	Flux.train!((i) -> -mean(logpx(model, x, z)), ps, 1:steps, ADAM())
	le = Flux.data(mean(logpx(model, x, z)))
	@info "initial = $(li) final = $(le)"
	Flux.data(z)
end
