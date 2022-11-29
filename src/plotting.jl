"""
	heatcontplot(f, xs, ys, xtrn, xtst, ytst; quantiles=[0.99])

	Plot the heatmap of a `f` on a grid `xs` and `ys`.
	If `xn` and `xa` are provided, they  will be plot
	assuming `xn` to be normal `xa` to be anomalous.
"""	
function heatcontplot(f, xs, ys, xtrn, xtst, ytst; quantiles=[0.99])
	xn = xtst[:,ytst .== 0]
	xa = xtst[:,ytst .== 1]
	xx = reduce(hcat, [[x,y] for x in xs, y in ys])
	scores = f(xx)
	zs = reshape(scores, length(xs), length(ys))
	fig, ax, hm = heatmap(xs, ys, zs, colormap = :gray1)
	levels = quantile(f(xtrn), quantiles)
	contour!(ax, xs, ys, zs; color = :red, levels, linewidth = 3)
	scatter!(ax, xn[1,:], xn[2,:], label="normal")
	scatter!(ax, xa[1,:], xa[2,:], label="anomalous")
	Colorbar(fig[:, end+1], hm)
	fig
end

function heatcontplot!(ax, f, xs, ys, xtrn, xtst, ytst; quantiles=[0.99])
	xn = xtst[:,ytst .== 0]
	xa = xtst[:,ytst .== 1]
	xx = reduce(hcat, [[x,y] for x in xs, y in ys])
	scores = f(xx)
	zs = reshape(scores, length(xs), length(ys))
	heatmap!(ax, xs, ys, zs, colormap = :gray1)
	levels = quantile(f(xtrn), quantiles)
	contour!(ax, xs, ys, zs; color = :red, levels, linewidth = 3)
	scatter!(ax, xn[1,:], xn[2,:], label="normal")
	scatter!(ax, xa[1,:], xa[2,:], label="anomalous")
end

function heatcontplot!(ax, f, dataset; quantiles=[0.99])
	(xtrn, (xtst, ytst), (xs, ys)) = dataset
	xn = xtst[:,ytst .== 0]
	xa = xtst[:,ytst .== 1]
	xx = reduce(hcat, [[x,y] for x in xs, y in ys])
	scores = f(xx)
	zs = reshape(scores, length(xs), length(ys))
	heatmap!(ax, xs, ys, zs, colormap = :gray1)
	levels = quantile(f(xtrn), quantiles)
	contour!(ax, xs, ys, zs; color = :red, levels, linewidth = 3)
	scatter!(ax, xn[1,:], xn[2,:], label="normal")
	scatter!(ax, xa[1,:], xa[2,:], label="anomalous")
end
