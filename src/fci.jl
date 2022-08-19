#ENV["PYTHON"] = "/Users/smerlak/opt/miniconda3/envs/fci/bin/python"
using PyCall
using LinearMaps, SparseArrays, LinearAlgebra

py"""
from pyscf import gto, scf, fci, ao2mo
from functools import reduce
from numpy import dot, array

def hop(coordinates, basis):

    mol = gto.M(
        atom = coordinates,
        basis = basis, 
        symmetry = True)

    hf = scf.HF(mol)
    hf.kernel()
    nelec = mol.nelectron
    norb = hf.mo_coeff.shape[0]

    neleca, nelecb = fci.addons._unpack_nelec(nelec)

    na = fci.cistring.num_strings(norb, neleca)
    nb = fci.cistring.num_strings(norb, nelecb)


    h1e = reduce(dot, (hf.mo_coeff.T, hf.get_hcore(), hf.mo_coeff))
    eri = ao2mo.incore.general(hf._eri, (hf.mo_coeff,)*4, compact=False)
    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)

    fci_solver = fci.FCI(hf)

    return lambda c: fci.direct_spin1.contract_2e(h2e, c.reshape(na,nb), norb, nelec).ravel(), na, nb, fci.direct_spin1.make_hdiag(h1e, eri, norb, nelec), hf, fci_solver
"""

function molecular_hamiltonian(molecule::String, basis::String; return_as_array = false)
    H, na, nb, hdiag, scf, fci_solver = py"hop"(molecule, basis)
    E₀ = scf.energy_nuc()
    L = LinearMap(c -> H(Array(c)) + E₀ * Array(c), na * nb; issymmetric=true)
    if return_as_array
        S = convert(SparseMatrixCSC, L)
        s = sortperm(diag(S); by = real)
        return S[s, s]
    else
        return (map = L, diagonal = hdiag)
    end
end