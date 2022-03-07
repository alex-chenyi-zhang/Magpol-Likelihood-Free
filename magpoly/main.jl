#include("src/magpoly.jl")

#using .magpoly
#using Polynomials
#@time begin
    #beta_temp = 0.5
    #spins_coupling = 1.0
    #alpha = 1.0
    #magpoly.simulation(200, 2, beta_temp, spins_coupling, alpha, "final_config.txt")
    #magpoly.simulation("input.txt")

    #magpoly.MMC_simulation(100, 2000, 500)
    #magpoly.MMC_simulation(150, 6000, 500)
    #magpoly.MMC_simulation(500, 200, 100)
    #magpoly.simulation("input.txt")
#end

include("src/sf_synthetic_likelihood.jl")
#synthetic_likelihood_polymer(300, 2, 100, 30000, 0.03, [-0.6,2.2,0.3], "features.txt", "data_file_uncoupled.txt")
Qamhi_polymer(200, 2, 100, 10000, 0.05, [-0.60148367,  0.94279556,  0.06326793], "features.txt", "saw_conf_data20.txt")




