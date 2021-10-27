module magpoly
include("structures_and_functions.jl")

function simulation(n_monomers::Int, n_steps::Int)
    beta_temp = 0.4
    spins_coupling = 1.0
    alpha = 1.0
    
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_monomers, beta_temp, spins_coupling, alpha)

    initialize_poly!(polymer)
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
    #display_polymer(polymer)

    #burn_in = floor(Int, 20*(n_monomers^(1.11)))
    #result = mean(simulation_data.rg2[burn_in:end])
    #return result
end

end # end of module
