include("fci.jl")

H₂O = """
O 0 0 0; 
H 0.2774 0.8929 0.2544;
H 0.6068, -0.2383, -0.7169
"""

O₂ = """
H 0 0 0; 
O 0 0 1.2;
"""

water = molecular_hamiltonian(H₂O, "sto-6g"; return_as_array = true)
oxygen = molecular_hamiltonian(O₂, "sto-6g"; return_as_array = true)