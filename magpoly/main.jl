include("src/magpoly.jl")

using .magpoly
using Polynomials
@time begin
    #beta_temp = 0.5
    #spins_coupling = 1.0
    #alpha = 1.0
    #magpoly.simulation(200, 2, beta_temp, spins_coupling, alpha, "final_config.txt")
    #magpoly.simulation("input.txt")

    #magpoly.MMC_simulation(100, 2000, 500)
    #magpoly.MMC_simulation(150, 6000, 500)
    #magpoly.MMC_simulation(500, 200, 100)
    magpoly.simulation("input.txt")
end



