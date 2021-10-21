include("src/magpoly.jl")

using .magpoly
using Polynomials
sizes = [200, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000]
simul_lengths = [1_000_000 for i in 1:9]
REE = Float64[]
RG2 = Float64[]
@time begin
    #=for i in 1:length(sizes)
        println("\n\n", sizes[i], "  ", simul_lengths[i])
        res = magpoly.simulation(sizes[i],simul_lengths[i])
        push!(REE, res[1])
        push!(RG2, res[2])
        
    end
    println(fit(log.(sizes), log.(RG2), 1))
    println(fit(log.(sizes), log.(REE), 1))
    
    println(REE, "\n")
    println(RG2,"\n")
    println(REE./RG2)=#
    println(magpoly.simulation(100,1000000))
    
end



