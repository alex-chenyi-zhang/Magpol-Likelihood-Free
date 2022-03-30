module magpoly
include("structures_and_functions.jl")

#############################################################################################################
#############################################################################################################

"""
    MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
Runs a MC simulation of the magnetic polymer by indexing the monte carlo 
steps from 'start' to 'finish'. Modifications are done on the pol::Magnetic_polymer
and the data is stored in the traj::MC_data at the positions from 'start' to 'finish'
"""
function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_base!(pol, traj, start, finish, inv_temp, J, fields)
end



"""
    MC_run!(pol::Magnetic_polymer, traj::MC_data, inv_temp::Float64, J::Float64, fields::Array{Float64})
Same thins as MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int)
but start and finish are defaulted to 1 and n_steps
"""
function MC_run!(pol::Magnetic_polymer, traj::MC_data, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_base!(pol, traj, 1, traj.n_steps, inv_temp, J, fields)
end



function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_base!(pol, traj, start, min(finish, sample_lag*n_samples), inv_temp, J, fields)
end


"""
    MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, sample_lag::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
Still performs the monte carlo simulation but one every 'sample_lag' steps the spins configuration is stored. 
The configurations are stored in spins_configs::Matrix{Int}
"""
function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_save!(pol, traj, start, finish, spins_configs, sample_lag, n_samples, inv_temp, J, fields)
end



"""
    MC_run!(pol::Magnetic_polymer, traj::MC_data, spins_configs::Matrix{Int}, sample_lag::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
Monte Carlo that saves configurations as above but defaults starting and ending steps to 1 and n_steps
"""
function MC_run!(pol::Magnetic_polymer, traj::MC_data, spins_configs::Matrix{Int}, sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_save!(pol, traj, 1, traj.n_steps, spins_configs, sample_lag, n_samples, inv_temp, J, fields)
end

"""
    MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, ising_energies::Array{Float64}, sample_lag::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
Not only the spin configurations are saved every 'sample_lag' steps but also the ising part of the energy. 
The configurations are stored in spins_configs::Matrix{Int} and the energies in ising_energies::Array{Float64}
"""
function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, ising_energies::Array{Float64}, sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64})
    MC_run_save!(pol, traj, start, finish, spins_configs, ising_energies, sample_lag, n_samples, inv_temp, J, fields)
end



#############################################################################################################
#############################################################################################################

"""
    MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, J::Float64, fields::Array{Float64})
Performs simulation with Multiple Markov Chains (MMC) attempting a swap between Chains
at two contiguous temperatures every 'stride' steps. The number of attempet
swaps is 'n_strides' and the values of the inverse temperatures of the 
different Markov chains are in the array 'inv_temps'
"""
function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, J::Float64, fields::Array{Float64})
    MMC_run_base!(polymers, trajs, n_strides ,stride, inv_temps, J, fields)
end



"""
    MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, sample_lag::Int, J::Float64, fields::Array{Float64})
Same thing as above but the spins configurations are saved once every 'sample_lag' steps. (only for the chain with lower temperature)
"""
function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, sample_lag::Int,n_samples::Int, J::Float64, fields::Array{Float64})
    MMC_run_save!(polymers, trajs, n_strides ,stride, inv_temps,spins_configs,sample_lag,n_samples, J, fields)
end

"""
    MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, ising_energies::Array{Float64}, sample_lag::Int, J::Float64, fields::Array{Float64})
Same thing as above but the spins configurations are saved once every 'sample_lag' steps and also the ising part of the energy. (only for the chain with lower
temperature)
"""
function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, ising_energies::Array{Float64}, sample_lag::Int,n_samples::Int, J::Float64, fields::Array{Float64})
    MMC_run_save!(polymers, trajs, n_strides ,stride, inv_temps,spins_configs, ising_energies, sample_lag,n_samples, J, fields)
end



#############################################################################################################
#############################################################################################################

function write_results(pol::Magnetic_polymer, traj::MC_data)
    !isdir("simulation_data") && mkdir("simulation_data")
    
    open("simulation_data/final_config_$(pol.n_mono).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
    open("simulation_data/MC_data_$(pol.n_mono).txt", "w") do io
        writedlm(io, [traj.energies traj.magnetization traj.rg2])
    end
end

function write_results(pol::Magnetic_polymer)
    !isdir("simulation_data") && mkdir("simulation_data")

    open("simulation_data/final_config_$(pol.n_mono).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
end 

end # end of module