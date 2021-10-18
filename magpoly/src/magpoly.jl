module magpoly
include("structures_and_functions.jl")

function prova()
    #pol = Magnetic_polymer(20, 1.2, 1.0, 0.5, [], [], Array{Int}(undef, 0, 0))
    #initialize_poly(pol)
    n_monomers = 20
    beta_temp = 1.0
    spins_coupling = 0.5
    alpha = 0.5

    polymer = Magnetic_polymer(n_monomers, beta_temp, spins_coupling, alpha, Array{Int}(zeros(n_monomers)), zeros(n_monomers), Array{Int, 2}(zeros(3,n_monomers)))
    initialize_poly!(polymer)

end

end # module
