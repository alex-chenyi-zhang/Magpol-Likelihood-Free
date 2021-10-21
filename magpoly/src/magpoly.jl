module magpoly
include("structures_and_functions.jl")

function simulation(n_monomers::Int, n_steps::Int)
    #pol = Magnetic_polymer(20, 1.2, 1.0, 0.5, [], [], Array{Int}(undef, 0, 0))
    #initialize_poly(pol)
    beta_temp = 1.0
    spins_coupling = 0.5
    alpha = 0.5
    
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_monomers, beta_temp, spins_coupling, alpha)

    initialize_poly!(polymer)
    MC_run!(polymer, simulation_data)
    #display_polymer(polymer)

    burn_in = floor(Int, 20*(n_monomers^(1.11)))
    result1 = mean(simulation_data.re2[burn_in:end])
    result2 = mean(simulation_data.rg2[burn_in:end])
    return (result1,result2)
end

end # module
