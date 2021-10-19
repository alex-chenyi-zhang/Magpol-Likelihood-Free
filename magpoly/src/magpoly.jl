module magpoly
include("structures_and_functions.jl")

function simulation(n::Int, n_st::Int)
    #pol = Magnetic_polymer(20, 1.2, 1.0, 0.5, [], [], Array{Int}(undef, 0, 0))
    #initialize_poly(pol)
    n_monomers = n
    beta_temp = 1.0
    spins_coupling = 0.5
    alpha = 0.5
    n_steps = n_st
    d = Dict{Int, Int}()
    simulation_data = MC_data(n_steps, zeros(n_steps), zeros(n_steps), zeros(Int, n_steps))
    polymer = Magnetic_polymer(n_monomers, beta_temp, spins_coupling, alpha, zeros(Int, n_monomers), zeros(n_monomers), 
                                zeros(Int, 3,n_monomers),zeros(Int, 3,n_monomers), d)
    initialize_poly!(polymer)
    MC_run!(polymer, simulation_data)
    #display_polymer(polymer)

    burn_in = floor(Int, 20*n_monomers^(1.11))
    result = mean(sqrt.(simulation_data.re2[burn_in:end]))
    return result
end

end # module
