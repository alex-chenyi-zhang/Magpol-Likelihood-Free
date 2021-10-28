include("src/magpoly.jl")

using .magpoly
using Polynomials
@time begin
    beta_temp = 0.5
    spins_coupling = 1.0
    alpha = 1.0
    magpoly.simulation(300, 100000, beta_temp, spins_coupling, alpha)
    
end



