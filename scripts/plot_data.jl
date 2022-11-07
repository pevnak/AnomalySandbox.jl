using Plots
using AnomalySandbox: one_gaussian, two_gaussians, two_rings, three_rings, two_moons, two_bananas

mkpath("plots")

function makefig(Xn, Xa, name)
	scatter(Xn[1,:], Xn[2,:], label="normal")
	scatter!(Xa[1,:], Xa[2,:], label="anomalous")
	savefig("plots/$(name).png")	
end

makefig(one_gaussian()..., "one_gaussian")
makefig(two_gaussians()..., "two_gaussians")
makefig(two_rings()..., "two_rings")
makefig(three_rings()..., "three_rings")
makefig(two_moons()..., "two_moons")
makefig(two_bananas()..., "two_bananas")
