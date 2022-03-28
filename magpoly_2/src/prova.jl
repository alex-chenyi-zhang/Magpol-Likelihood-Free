include("magpoly.jl")

using .magpoly
const mp = magpoly
using DelimitedFiles, Random, Distributions, LinearAlgebra


spins_coupling = 1.0
#spins_coupling = 0.0
alpha = 0.5
inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
n_temps = length(inv_temps)
n_mono = 300
n_strides = 200
stride = 200

polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
trajs = Array{mp.MC_data}(undef, n_temps)
for i_temp in 1:n_temps
    polymers[i_temp] = mp.Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
    trajs[i_temp] = mp.MC_data(n_strides*stride)
    if isfile("simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
        mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
    else
        mp.initialize_poly!(polymers[i_temp])
    end
end 

fields = zeros(n_mono)
mp.set_fields!(polymers, fields)
mp.set_coupling!(polymers, 0.7)
println(polymers[1].J)

println("ok")