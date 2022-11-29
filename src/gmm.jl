fit_gmm(n::Int, x::AbstractMatrix, args...; kwargs...) = GaussianMixtures.GMMk(n, Array(x'))
function StatsBase.predict(m::GMM, x::AbstractMatrix; score="weighted") 
	if score == "weighted"
		return -llpg(m, Array(x')) * m.w
	elseif score == "min"
		return minimum(-llpg(m, Array(x')), dims=2)
	end
end

struct Mixture{T<:Real}
	α::Vector{T}
	μ::Matrix{T}
	σ::Vector{T}
end

Mixture(x::Matrix{T}, k::Int, σ::Real) where {T} = (α = T.(rand(k)); Mixture(α./sum(α), x[:, sample(1:size(x,2), k, replace = false)], fill(T(σ), k)))

function (m::Mixture)(x)
	d = pairwise(SqEuclidean(), m.μ, x)
	logZ = size(x,1) .* log.(2π .*m.σ .^ 2) ./ 2
	logp = - d ./ (2 .* m.σ .^ 2) .- logZ
	-sum(m.α .* exp.(logp), dims = 1)
end

function update!(m::Mixture, x)	
	d = pairwise(SqEuclidean(), m.μ, x)
	lkl = 0.0
	for i in 1:length(m.α)
		logZ = size(x,1) .* log.(2π .* m.σ .^ 2) ./ 2
		logp = - d ./ (2 .* m.σ .^ 2) .- logZ
		lkl = sum(m.α .* exp.(logp), dims = 1)
		k = mapslices(xx -> argmax(xx), logp, dims = 1)[:]
		mask = findall(k .== i)
		m.α[i] = length(mask) / size(x,2)
		isempty(mask) && continue
		xx = x[:, mask]
		m.μ[:, i] = mean(xx, dims = 2)
		m.σ[i] = 1 / sqrt(mean(diag(xx * xx')))
	end
	mean(lkl)
end

function StatsBase.fit!(m, x ; steps = 10)
	for i in 1:steps
		lkl = update!(m, x)
	end
	update!(m, x)
end