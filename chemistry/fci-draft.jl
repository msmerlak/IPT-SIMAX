import SparseArrays: sparse
using LinearAlgebra

using IterativePerturbationTheory, KrylovKit, Arpack

using PyCall
@pyimport openfermion as of
@pyimport openfermionpyscf as ofpyscf
@pyimport openfermionpsi4 as ofpsi4
@pyimport scipy.sparse as sp
@pyimport pyscf
@pyimport primme

const scipy_sparse_find = pyimport("scipy.sparse")["find"]
function sparse(Apy::PyObject)
    IA, JA, SA = scipy_sparse_find(Apy)
    return sparse(Int[i + 1 for i in IA], Int[i + 1 for i in JA], SA)
end

#create Hamiltonian and compute HF energy
molecule_geometry = of.geometry_from_pubchem("LiH")
molecule = of.chem.MolecularData(molecule_geometry, basis="6-31g", multiplicity=1)

molecule = ofpyscf.run_pyscf(molecule, run_scf=true, run_fci=true)
molecule.hf_energy
molecule.fci_energy

#convert Hamiltonian to matrix form
h = of.get_sparse_operator(molecule.get_molecular_hamiltonian())

H = sparse(h);
s = sortperm(diag(H), by=real);
H = H[s, s];
hop = sp.linalg.LinearOperator(h.shape, matvec=x -> h * x)
e = Vector(SparseVector(size(H, 1), [s[1]], [1.0]))
@CPUtime pyscf.lib.davidson(hop, e, h.diagonal(), tol=1e-10)[1]
@time pyscf.lib.davidson(hop, e, h.diagonal(), tol=1e-10)[1]

@CPUtime E = ipt(H, 1; tol=1e-10, acceleration=:acx).values
@time E = ipt(H, 1; tol=1e-10, acceleration=:acx).values

@CPUtime eigsolve(H, e, 1, :SR)[1]

v0 = Complex.(e)
@CPUtime eigs(H; nev=1, v0=v0, which=:SR)[1]




