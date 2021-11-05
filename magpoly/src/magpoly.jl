module magpoly
include("structures_and_functions.jl")


function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
                    n_strides::Int, stride::Int, inv_temps::Array{Float64})
    n_temps = length(inv_temps)
    accepted_swaps = 0
    temp_coord = zeros(Int,3)

    for i_strides in 1:n_strides
        for i_temp in 1:n_temps
            MC_run!(polymers[i_temp], trajs[i_temp],(i_strides-1)*stride+1,i_strides*stride)
        end

        swap = rand(1:n_temps-1)
        delta_ene = (inv_temps[swap] - inv_temps[swap+1]) * (trajs[swap+1].energies[i_strides*stride] - trajs[swap].energies[i_strides*stride])
        alpha_swap = 0.0
        delta_ene<=0 ? alpha_swap=1.0 : alpha_swap=exp(-delta_ene)
        if alpha_swap >= rand()
            for i_mono in 1:polymers[1].n_mono
                for j in 1:3
                   temp_coord[j] = polymers[swap].coord[j,i_mono] 
                   polymers[swap].coord[j,i_mono] = polymers[swap+1].coord[j,i_mono]
                   polymers[swap+1].coord[j,i_mono] =  temp_coord[j]
                end
                temp_spin = polymers[swap].spins[i_mono]
                polymers[swap].spins[i_mono] = polymers[swap+1].spins[i_mono]
                polymers[swap+1].spins[i_mono] = temp_spin
            end
            accepted_swaps += 1
        end
    end
    println("Accepted_swaps: ", accepted_swaps)
end

function MMC_simulation(n_mono::Int, n_strides::Int, stride::Int)
    spins_coupling = 1.0
    alpha = 1.0

    
    inv_temps = [1.1, 1.05, 1.0, 0.95, 0.9, 0.8, 0.74, 0.69, 0.67, 0.64, 0.63, 0.625, 0.617,
    0.606, 0.595, 0.588, 0.581, 0.571, 0.555, 0.541, 0.531, 0.518, 0.5, 0.4]
    n_temps = length(inv_temps)
    polymers = Array{Magnetic_polymer}(undef, n_temps)
    trajs = Array{MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = MC_data(n_strides*stride)
        initialize_poly!(polymers[i_temp])#,"simulation_data/final_config_$(n_mono)_$(inv_temps[i_temp]).txt")
    end 

    #=for i_strides in 1:n_strides
        println("Istride = ", i_strides)
        for i_temp in 1:n_temps
            MC_run!(polymers[i_temp], trajs[i_temp],(i_strides-1)*stride+1,i_strides*stride)
        end
    end=#

    MMC_run!(polymers,trajs,n_strides,stride,inv_temps)
    
    for i_temp in 1:n_temps
        write_results(polymers[i_temp],trajs[i_temp])
    end
end

end # end of module
