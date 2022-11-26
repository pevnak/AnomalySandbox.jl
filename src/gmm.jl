fit_gmm(n::Int, x::AbstractMatrix, args...; kwargs...) = GaussianMixtures.GMMk(n, Array(x'))
function StatsBase.predict(m::GMM, x::AbstractMatrix; score="weighted") 
	if score == "weighted"
		return -llpg(m, Array(x')) * m.w
	elseif score == "min"
		return minimum(-llpg(m, Array(x')), dims=2)
	end
end
