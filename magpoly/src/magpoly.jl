module magpoly
include("structures_and_functions.jl")

function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int)
    MC_run_base!(pol, traj, start, finish)
end

function MC_run!(pol::Magnetic_polymer, traj::MC_data)
    MC_run_base!(pol, traj, 1, traj.n_steps)
end

function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, sample_lag::Int)
    MC_run_save!(pol, traj, start, finish, spins_configs, sample_lag)
end

function MC_run!(pol::Magnetic_polymer, traj::MC_data, spins_configs::Matrix{Int}, sample_lag::Int)
    MC_run_save!(pol, traj, 1, traj.n_steps, spins_configs, sample_lag)
end

function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64})
    MMC_run_base!(polymers, trajs, n_strides ,stride, inv_temps)
end

function MMC_run!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, sample_lag::Int)
    MMC_run_save!(polymers, trajs, n_strides ,stride, inv_temps,spins_configs,sample_lag)
end



function write_results(pol::Magnetic_polymer, traj::MC_data)
    !isdir("simulation_data") && mkdir("simulation_data")
    
    open("simulation_data/final_config_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
    open("simulation_data/MC_data_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [traj.energies traj.magnetization traj.rg2])
    end
end

function write_results(pol::Magnetic_polymer)
    !isdir("simulation_data") && mkdir("simulation_data")

    open("simulation_data/final_config_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
end

###################################################################################
###################################################################################

function simulation(n_mono::Int, n_steps::Int ,beta_temp::Float64, spins_coupling::Float64, alpha::Float64)
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    initialize_poly!(polymer)
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end

function simulation(n_mono::Int, n_steps::Int ,beta_temp::Float64, spins_coupling::Float64, alpha::Float64, conf_file::String)
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    initialize_poly!(polymer, conf_file)
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end



function simulation(input::String)
    # First read parameters from input
    io = open("input.txt", "r")
    n_mono = 1
    n_steps = 1
    beta_temp = 1.0
    spins_coupling = 0.0
    alpha = 1.0
    conf_file = ""
    from_file = false
    while ! eof(io)
        line = split(readline(io))
        if line[1] == "n_mono"
            n_mono = parse(Int, line[2])
        elseif line[1] == "n_steps"
            n_steps = parse(Int, line[2])
        elseif line[1] == "spins_coupling"
            spins_coupling = parse(Float64, line[2])
        elseif line[1] == "alpha"
            alpha = parse(Float64, line[2])
        elseif line[1] == "inverse_temperature"
            beta_temp = parse(Float64, line[2])
        elseif line[1] == "initialize_from_files"
            from_file = parse(Bool, line[2])
        elseif line[1] == "conf_file"
            conf_file = String(line[2])
        end
    end
    
    close(io)

    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    
    if from_file
        initialize_poly!(polymer, conf_file)
    else
        initialize_poly!(polymer)
    end
    
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end


function MMC_simulation(n_mono::Int, n_strides::Int, stride::Int)
    spins_coupling = 1.0
    alpha = 1.0
 
    #inv_temps = [1.1, 1.05, 1.0, 0.95, 0.9, 0.8, 0.74, 0.69, 0.67, 0.64, 0.63, 0.625, 0.617,
    #0.606, 0.595, 0.588, 0.581, 0.571, 0.555, 0.541, 0.531, 0.518, 0.5, 0.4]

    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.57, 0.55, 0.5, 0.4]
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
