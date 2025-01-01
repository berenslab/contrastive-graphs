using .SparseArrays
using .NPZ
using .SGtSNEpi
using .ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_iter"
    help = "Parameter `max_iter` in sgtsnepi"
    arg_type = Int
    default = 10
end
args = parse_args(ARGS, s)
A = sparse([1, 2, 3], [2, 3, 1], UInt8.([1, 1, 1]), 3, 3)
Y0 = randn(3, 2)
kwargs = Dict(Symbol(k) => v for (k, v) in args)
Y = sgtsnepi(A; Y0=Y0, kwargs...)
npzwrite("Y.npy", Y)
npzread("Y.npy")
rm("Y.npy")
