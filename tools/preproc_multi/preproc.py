import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile

gas = ct.Solution("input_cantera.yaml")
#gas = ct.Solution("h2o2.yaml")

N_S = gas.n_species
N_R = gas.n_reactions

species_names = gas.species_names

mw = gas.molecular_weights
rgas = ct.gas_constant / mw

if N_R > 0:
    reactions = []
    A = np.zeros((N_R, 2))
    b = np.zeros((N_R, 2))
    Ea = np.zeros((N_R, 2))
    falloff_coeff = np.zeros((N_R, 5))
    thirdbody_eff = np.ones((N_R, N_S))
    reactants = np.zeros((N_R, N_S))
    products  = np.zeros((N_R, N_S))
    ReacType = np.zeros(N_R) 
    for nr,rxn in enumerate(gas.reactions()):
        rate = rxn.rate
        for sp, coeff in rxn.reactants.items():
            lsp = species_names.index(sp)
            reactants[nr, lsp] = coeff
        for sp, coeff in rxn.products.items():
            lsp = species_names.index(sp)
            products[nr, lsp] = coeff
        A[nr,:] = 0.0
        b[nr,:] = 0.0
        Ea[nr,:] = 0.0
        falloff_coeff[nr,:] = 0.0
    
        if isinstance(rate, ct.ArrheniusRate):
            if rxn.third_body is None: ReacType[nr] = 0 # Arrhenius
            if rxn.third_body is not None: ReacType[nr] = 1 #"three-body-Arrhenius"
            A[nr,0]  = rxn.rate.pre_exponential_factor
            b[nr,0]  = rxn.rate.temperature_exponent
            Ea[nr,0] = rxn.rate.activation_energy
        elif isinstance(rate, ct.LindemannRate):
            ReacType[nr] = 2 #"falloff-Lindemann"
            low = rate.low_rate
            high = rate.high_rate
            A[nr, 0] = low.pre_exponential_factor
            b[nr, 0] = low.temperature_exponent
            Ea[nr, 0] = low.activation_energy
            A[nr, 1] = high.pre_exponential_factor
            b[nr, 1] = high.temperature_exponent
            Ea[nr, 1] = high.activation_energy
        elif isinstance(rate, ct.TroeRate) or isinstance(rate, ct.SriRate):
            if isinstance(rate, ct.TroeRate): ReacType[nr] = 3 #"falloff-Troe"
            if isinstance(rate, ct.SriRate): ReacType[nr] = 4 #"falloff-SRI"
            low = rate.low_rate
            high = rate.high_rate
            A[nr, 0] = low.pre_exponential_factor
            b[nr, 0] = low.temperature_exponent
            Ea[nr, 0] = low.activation_energy
            A[nr, 1] = high.pre_exponential_factor
            b[nr, 1] = high.temperature_exponent
            Ea[nr, 1] = high.activation_energy
            c = rate.falloff_coeffs
            for j in range(len(c)):
                falloff_coeff[nr,j] = c[j]
            if len(c) < 5:
                falloff_coeff[nr,len(c):] = [0.0]*(5-len(c))
        else:
            raise TypeError(f"reaction {nr}: unsupported rate {type(rxn.rate)}")
    
        if rxn.third_body is not None:
            eff = rxn.third_body.efficiencies
            for lsp, sp in enumerate(species_names):
                if eff is not None and sp in eff:
                    thirdbody_eff[nr, lsp] = eff[sp]
                else:
                    thirdbody_eff[nr, lsp] = 1.0
        else:
            thirdbody_eff[nr, :] = 1.0

for lsp,sp in enumerate(gas.species()):
    nasa = sp.thermo
    if isinstance(nasa, ct.NasaPoly2):  # NASA7 with 2 temp ranges
        thermo_model = "NASA7"
        number_cp_coeff = 7
        nsetcv = 2
        indx_cp_l = 0
        indx_cp_r = 4
    elif isinstance(nasa, ct.Nasa9PolyMultiTempRegion):  # NASA9
        thermo_model = "NASA9"
        number_cp_coeff = 9
        nsetcv = int(nasa.coeffs[0])
        indx_cp_l = -2
        indx_cp_r = 4
    else:
        raise TypeError(f"{sp.name}: unsupported thermo model {type(nasa)}")
    lencv = len(nasa.coeffs)
    if lsp == 0:
        thermo_model_ref = thermo_model
        nsetcv_ref = nsetcv
        lencv_ref = lencv
        number_cp_coeff_ref = number_cp_coeff
    else:
        assert thermo_model == thermo_model_ref, "Thermo model not consistent across species"
        assert nsetcv == nsetcv_ref, f"{sp.name}: nsetcv mismatch ({nsetcv} vs {nsetcv_ref})"
        assert lencv == lencv_ref, f"{sp.name}: coeff vector length mismatch"
        assert number_cp_coeff == number_cp_coeff_ref, "Mismatch in cp coeffs per region"

cp_coeffs = np.zeros([indx_cp_r-indx_cp_l+1+2, N_S, nsetcv])
trange = np.zeros([N_S,nsetcv+1])
for lsp, sp in enumerate(gas.species()):
    nasa = sp.thermo
    coeffs = np.array(nasa.coeffs)
    if thermo_model == "NASA7":
        # CANTERA rer: coeffs	Vector of coefficients used to set the parameters for the standard state
        # [Tmid, 7 high-T coeffs, 7 low-T coeffs]. This is the coefficient order used in the standard NASA format.
        trange[lsp,:] = [nasa.min_temp, coeffs[0], nasa.max_temp]
        cp_coeffs[:, lsp, 0] = coeffs[8:15]
        cp_coeffs[:, lsp, 1] = coeffs[1:8]
    elif thermo_model == "NASA9":
        trange_list = []
        for nset in range(nsetcv):
            trange_list.extend(coeffs[1+11*nset:3+11*nset])
            cp_coeffs[:, lsp, nset] = coeffs[3+11*nset:12+11*nset]
        trange[lsp,:] = np.unique(trange_list)


#transport_data
sigma    = np.zeros(N_S)
epsK     = np.zeros(N_S)
dipole   = np.zeros(N_S)
polariz  = np.zeros(N_S)
geometry = ["" for _ in range(N_S)]

for lsp, sp in enumerate(gas.species()):
    trdata = sp.transport
    if not isinstance(trdata, ct.GasTransportData):
        raise TypeError(f"Species {sp.name} transport is not GasTransportData")

    sigma[lsp]    = trdata.diameter         # meters
    epsK[lsp]     = trdata.well_depth / ct.boltzmann  # K
    dipole[lsp]   = trdata.dipole           # C·m
    polariz[lsp]  = trdata.polarizability   # m^3
    geometry[lsp] = trdata.geometry.lower()
    if geometry[lsp] == 'atom': 
        geometry[lsp] = 0
    elif geometry[lsp] == 'linear': 
        geometry[lsp] = 1
    elif geometry[lsp] == 'nonlinear': 
        geometry[lsp] = 2
    else:
        raise TypeError(f"Species {sp.name} geometry non recognized") 

# H at Tref
Tref = 298.15
yy = np.zeros(N_S)
yy[0] = 1. #dummy composition
gas.TPY = Tref, ct.one_atm, yy
hTref = gas.partial_molar_enthalpies # J/Kmol

# Tabulation parameters
t_min_tab = 200.
t_max_tab = 5500. 
dt_tab = 1.
num_t_tab = int(t_max_tab-t_min_tab)/dt_tab
num_t_tab = int(num_t_tab)
dt_tab = (t_max_tab-t_min_tab)/num_t_tab

# viscosity, thermal conductivity, and binary diffusivities
visc_species = np.zeros([num_t_tab+1,N_S])
lambda_species = np.zeros([num_t_tab+1,N_S])
diffbin_species_1atm = np.zeros([num_t_tab+1,N_S,N_S])
tloc_vec = np.zeros(num_t_tab+1)
for it in range(num_t_tab+1): 
    tloc = t_min_tab + it*dt_tab
    tloc_vec[it] = tloc
    for lsp in range(N_S):
        yy = np.zeros(N_S)
        yy[lsp] = 1.
        gas.TPY = tloc, ct.one_atm, yy
        visc_species[it,lsp] = gas.viscosity
        lambda_species[it,lsp] = gas.thermal_conductivity
    gas.TP = tloc, 7173. #ct.one_atm
    diffbin_species_1atm[it,:,:] = gas.binary_diff_coeffs

# equilibrium constants
if N_R > 0:
    kc = np.zeros([num_t_tab+1,N_R])
    for it in range(num_t_tab+1): 
        tloc = t_min_tab + it*dt_tab
        gas.TP = tloc, ct.one_atm
        kc[it,:] = gas.equilibrium_constants

# ---------------- 1) Species ----------------
fixed_len_species = 30
species_fixed = np.array([s.ljust(fixed_len_species) for s in species_names], dtype='S30')

with FortranFile('fluid_prop.bin','w') as f:
    f.write_record(np.array([N_S], dtype=np.int32),np.array([N_R], dtype=np.int32))
    f.write_record(species_fixed)
    f.write_record(mw.astype(np.float64))
    f.write_record(rgas.astype(np.float64))
    f.write_record(hTref.astype(np.float64))
    f.write_record(sigma.astype(np.float64))
    f.write_record(epsK.astype(np.float64))
    f.write_record(dipole.astype(np.float64))
    f.write_record(polariz.astype(np.float64))
    f.write_record(np.array(geometry,dtype=np.int32))

# ---------------- 2) Cp coefficients ----------------
#with FortranFile('cp_coeffs.bin','w') as f:
    f.write_record(np.array([number_cp_coeff, nsetcv], dtype=np.int32))
    f.write_record(trange.ravel(order='F').astype(np.float64))
    f.write_record(cp_coeffs.ravel(order='F').astype(np.float64))

# ---------------- Viscosities ----------------
#with FortranFile('viscosities.bin','w') as f:
    f.write_record(np.array([t_min_tab, t_max_tab, dt_tab], dtype=np.float64))
    f.write_record(visc_species.ravel(order='F').astype(np.float64))

# ---------------- Thermal conductivities ----------------
#with FortranFile('thermal_conductivities.bin','w') as f:
#    f.write_record(np.array([t_min_tab, t_max_tab, dt_tab], dtype=np.float64))
    f.write_record(lambda_species.ravel(order='F').astype(np.float64))

# ---------------- Binary diffusivities ----------------
#with FortranFile('binary_diffusivities.bin','w') as f:
#    f.write_record(np.array([t_min_tab, t_max_tab, dt_tab], dtype=np.float64))
    f.write_record(diffbin_species_1atm.ravel(order='F').astype(np.float64))

# ---------------- Reactions ----------------
#with FortranFile('reactions.bin','w') as f:
    if N_R > 0:
        f.write_record(ReacType.astype(np.int32))
        f.write_record(A.ravel(order='F').astype(np.float64))
        f.write_record(b.ravel(order='F').astype(np.float64))
        f.write_record(Ea.ravel(order='F').astype(np.float64))
        f.write_record(falloff_coeff.ravel(order='F').astype(np.float64))
        f.write_record(thirdbody_eff.ravel(order='F').astype(np.float64))
        f.write_record(reactants.ravel(order='F').astype(np.float64))
        f.write_record(products.ravel(order='F').astype(np.float64))
        f.write_record(kc.ravel(order='F').astype(np.float64))


