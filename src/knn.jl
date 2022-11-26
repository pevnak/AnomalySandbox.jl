struct KNN{T}
	x::Matrix{T}
	k::Integer
end

function StatsBase.predict(m::KNN, x)
	d = pairwise(Euclidean(), m.x, x)
	vec(mapslices(s -> sort(s)[m.k], d, dims = 1))
end
