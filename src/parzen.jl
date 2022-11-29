struct Parzen{T}
	x::Matrix{T}
	σ::T
end

function StatsBase.predict(m::Parzen, x)
	d = -pairwise(Euclidean(), m.x, x)
	-log.(mean(pdf.(Normal(0, m.σ), d), dims = 1)[:])
end
