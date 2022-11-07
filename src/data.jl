function one_gaussian()
	normal = randn(2,500)
	anomalous = randn(2,100) .* 10
	normal, anomalous[:,vec(sum(anomalous .^ 2, dims=1) .> 15)]
end

function two_gaussians()
	m1 = [5,5]
	m2 = [-5,-5]
	normal = hcat(randn(2,500) .+ m1, randn(2,500) .+ m2)
	anomalous = randn(2,100) .* 10
	ainds1 = vec(sum((anomalous .+ m1) .^ 2, dims=1) .> 15)
	ainds2 = vec(sum((anomalous .+ m2) .^ 2, dims=1) .> 15)
	normal, anomalous[:, ainds1 .& ainds2]
end

polar(R::Real, θ::Real) = [R*cos(θ), R*sin(θ)]
circular_section(μ::Vector, R::Real, N::Int, θ1::Real, θ2::Real) = 
	μ .+ hcat(map(x->polar(R, x),range(θ1,stop=θ2,length=N))...)
circle(μ::Vector, R::Real, N::Int) = circular_section(μ, R, N, 0, 2*pi)
rand_ring(μ::Vector, R::Real, N::Int, sd) = circle(μ, R, N) .+ randn(2, N).*sd
rand_rings(μs, Rs::Vector, Ns::Vector, sds) = map(rand_ring, μs, Rs, Ns, sds)
rand_rings(Rs::Vector, Ns::Vector, sds) = rand_rings([zeros(2) for i in 1:length(Rs)], Rs, Ns, sds)

two_rings() = rand_rings([5,10], [500,500], [0.7,1])

function three_rings() 
	r1, r2, r3 = rand_rings([5,15,25], [500,500,500], [1,1.5,2])
	normal = r2
	anomalous = hcat(r1, r3)
	normal, anomalous
end

moon(μ::Vector, R::Real, N::Int, θ1::Real, θ2::Real, maxsd::Real) = 
	circular_section(μ, R, N, θ1, θ2) .+ randn(2, N).*(vcat(range(0,stop=maxsd,length=Int(floor(N/2))), 
		range(maxsd,stop=0,length=Int(ceil(N/2)))))'

function two_moons()
	normal = moon([-1,-0.4], 2, 500, 0, pi, 0.2)
	anomalous = moon([1,0.4], 2, 500, pi, 2*pi, 0.2)
	return normal, anomalous
end

banana(μ::Vector, R::Real, N::Int, θ1::Real, θ2::Real, maxsd::Real) = 
	circular_section(μ, R, N, θ1, θ2) .+ randn(2, N).*(vcat(range(maxsd/2,stop=maxsd,length=Int(floor(N/2))), 
		range(maxsd,stop=maxsd/2,length=Int(ceil(N/2)))))'

function two_bananas()
	# generate the bananas
	normal = banana([-1,-0.4], 2, 500, 0, pi, 0.2)
	anomalous = banana([1,0.4], 2, 500, pi, 2*pi, 0.2)
	return normal, anomalous
end

