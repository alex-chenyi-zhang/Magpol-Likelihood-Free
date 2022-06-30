include("src/magpoly.jl")

using .magpoly
const mp = magpoly
using DelimitedFiles, Random, Distributions, LinearAlgebra

n_mono = 1000
inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
n_temps = length(inv_temps)
stride = 200
n_strides = 50

polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
trajs = Array{mp.MC_data}(undef, n_temps)
Threads.@threads for i_temp in 1:n_temps
    polymers[i_temp] = mp.Magnetic_polymer(n_mono)
    trajs[i_temp] = mp.MC_data(n_strides*stride)
    mp.initialize_poly!(polymers[i_temp])
end 

fields = rand(n_mono) .- 0.5.*ones(n_mono)

@time mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, 0.5, fields)

