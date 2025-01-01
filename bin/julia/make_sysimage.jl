import Pkg
pkgs = ["SGtSNEpi", "SparseArrays", "NPZ", "ArgParse", "PackageCompiler"]
Pkg.add(pkgs)

import PackageCompiler

function (@main)(ARGS)
    PackageCompiler.create_sysimage(
        ["SGtSNEpi", "SparseArrays", "NPZ", "ArgParse"],
        sysimage_path="nik_sgtsnepi.so",
        import_into_main = true,
    )
end
