include("src/magpoly.jl")

using .magpoly
using Polynomials
@time begin
    magpoly.simulation("input.txt")
end



