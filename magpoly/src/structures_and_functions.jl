using Random, Distributions
Random.seed!(2021)

#=struct poly_parameters{T1<:Int, T2<:AbstractFloat}  
    n_mono::T1
    inv_temp::T2
    J::T2
    alpha::T2
end=#


# This structure contains all I need to compute the Hamiltonian
struct Magnetic_polymer{T1<:Int, T2<:AbstractFloat}
    n_mono::T1
    inv_temp::T2
    J::T2
    alpha::T2
    spins::Array{T1}
    fields::Array{T2}
    coord::Array{T1,2}         # Coordinates of the monomers
end

function initialize_poly!(poly::Magnetic_polymer)
    d = Normal(0,2)
    poly.spins .= rand(0:1, poly.n_mono)
    poly.fields .= rand(d, poly.n_mono)
    for i_mono in 1:poly.n_mono
        poly.coord[1, i_mono] = i_mono
    end    
end

function display_polymer(polymer::Magnetic_polymer)
    println("n_mono: $(polymer.n_mono)")
    println("inv_temp: $(polymer.inv_temp)")
    println("coupling: $(polymer.J)")
    for i in 1:(polymer.n_mono)
        println(polymer.coord[:,i])
    end
    for i in 1:(polymer.n_mono)
        println("f$i: $(polymer.fields[i])")
    end
    for i in 1:(polymer.n_mono)
        println("s$i: $(polymer.spins[i])")
    end
    println(typeof(polymer.n_mono))
    println(typeof(polymer.inv_temp))
    println(typeof(polymer.J))
    println(typeof(polymer.alpha))
    polymer.spins
end

