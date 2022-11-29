struct PCADetector{T}
	mn::Vector{T}
	P::Matrix{T}
end

function PCADetector(x::Matrix, k::Int)
	mn = mean(x, dims = 2)
	x = x .- mn
	e, v = eigen(cov(x'))
	I = sortperm(e, rev = true)
	P = v'[:,I[1:k]]
	PCADetector(vec(mn), P)
end

function StatsBase.predict(m::PCADetector, x)
	mn, P = m.mn, m.P
	x = x .- mn
	vec(sum(((P * P' * x) .- x).^2, dims = 1))
end	