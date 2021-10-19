include("src/magpoly.jl")

using .magpoly
using Polynomials
sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
simul_lengths = [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 2_000_000, 2_000_000]
REE = Float64[]
@time begin
    for i in 1:length(sizes)
        println("\n\n", sizes[i], "  ", simul_lengths[i])
        push!(REE, magpoly.simulation(sizes[i],simul_lengths[i]))
    end
    println(fit(log.(sizes), log.(REE), 1))
end



