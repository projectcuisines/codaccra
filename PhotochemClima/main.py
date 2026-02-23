from photochem.clima import AdiabatClimate
from photochem.extensions import gasgiants
from photochem.utils import settings_file_for_climate, species_file_for_climate

import numpy as np
from matplotlib import pyplot as plt
# from threadpoolctl import threadpool_limits
# _ = threadpool_limits(limits=4)

###
### Utils needed for all cases
###

def write_outputfile(c, filename, planet_diameter, planet_gravity, star_distance, star_type, 
                     star_temperature, atmosphere_description):

    OLR = (c.rad.wrk_ir.fup_n[-1] - c.rad.wrk_ir.fdn_n[-1])/1e3
    ASR = -(c.rad.wrk_sol.fup_n[-1] - c.rad.wrk_sol.fdn_n[-1])/1e3

    tmp = f"""# Model name = photochem.clima
# Diameter: {int(planet_diameter)}                # Diameter of the planet [km]
# Gravity: {float(planet_gravity):.2f}                 # Gravity at the surface of the atmosphere [m/s2]
# Star-distance: {float(star_distance):.3f}               # Planet-star distance [au]
# Star-type: {star_type}                 # Host-star type
# Star-temperature: {int(star_temperature)}         # Host-star temperature [K]
# Wavelength-min: N/A            # Minimum wavelength [um]
# Wavelength-max: N/A            # Maximum wavelength [um]
# Radiance-unit: W/m^2           # Radiation output unit
# Surface-temperature: {float(c.T_surf):.0f}       # Surface temperature [K]
# Surface-albedo: {float(c.rad.surface_albedo[0]):.2f}           # Surface albedo
# Surface-emissivity: {float(c.rad.surface_emissivity[0]):.2f}        # Surface emissivity
# Opacity: corrk + continuum     # corrk + continuum, other ? 
# Line opacities = H2O, CO2
# CIA opacities = H2O-H2O, H2O-N2, CO2-CO2, N2-N2
# Rayleigh opacities = N2, CO2, H2O
# Outgoing longwave radiation (W/m^2) = {OLR:.2f}
# Absorbed stellar radiation (W/m^2) = {ASR:.2f}
# Atmosphere-description: {atmosphere_description}
# Atmosphere-columns: P T MMW Alt N2 CO2 H2O OTR ASR
# Atmosphere-fit-T: N/A
# Atmosphere-fit-H2: N/A
# Atmosphere-fit-He: N/A
# Atmosphere-fit-H2O: N/A
# Atmosphere-fit-CH4: N/A
# Atmosphere-layers: {int(len(c.P))}
P (Pa)        T (K)         MMW (g/mol)   Alt (km)      N2            CO2           H2O           OTR           ASR       
"""

    assert c.species_names == ['H2O', 'CO2', 'N2']
    mmw = np.zeros_like(c.P)
    for i in range(len(c.P)):
        mmw[i] = np.sum(c.f_i[i,:]*np.array([18,44,28]))

    OLRs = -(c.rad.wrk_ir.fdn_n - c.rad.wrk_ir.fup_n)/1.0e3
    OLRs = OLRs[::2]

    ASRs = (c.rad.wrk_sol.fdn_n - c.rad.wrk_sol.fup_n)/1.0e3
    ASRs = ASRs[::2]

    fmt = '{:14}'
    tmp1 = fmt.format('%.3e'%(c.P_surf/10))
    tmp1 += fmt.format('%.4f'%(c.T_surf))
    tmp1 += fmt.format('%.4f'%(mmw[0]))
    tmp1 += fmt.format('%.4f'%(0.0))
    tmp1 += fmt.format('%.4e'%(c.f_i[0,2]))
    tmp1 += fmt.format('%.4e'%(c.f_i[0,1]))
    tmp1 += fmt.format('%.4e'%(c.f_i[0,0]))
    tmp1 += fmt.format('%.4e'%(OLRs[0]))
    tmp1 += fmt.format('%.4e'%(ASRs[0]))
    tmp1 += '\n'
    for i in range(len(c.P)):
        tmp1 += fmt.format('%.3e'%(c.P[i]/10))
        tmp1 += fmt.format('%.4f'%(c.T[i]))
        tmp1 += fmt.format('%.4f'%(mmw[i]))
        tmp1 += fmt.format('%.4f'%(c.z[i]/1e5))
        tmp1 += fmt.format('%.4e'%(c.f_i[i,2]))
        tmp1 += fmt.format('%.4e'%(c.f_i[i,1]))
        tmp1 += fmt.format('%.4e'%(c.f_i[i,0]))
        tmp1 += fmt.format('%.4e'%(OLRs[i+1]))
        tmp1 += fmt.format('%.4e'%(ASRs[i+1]))
        tmp1 += '\n'

    tmp += tmp1

    with open(filename,'w') as f:
        f.write(tmp)

def plot(c):

    plt.rcParams.update({'font.size': 15})
    fig,ax = plt.subplots(1,1,figsize=[5,4])

    for i in range(len(c.species_names)):
        ax.plot(c.f_i[:,i], c.P/1e6, lw=2, label=c.species_names[i])
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(c.P_surf/1e6,c.P_top/1e6)
    ax.legend(ncol=1,bbox_to_anchor=(1.1, 1.02), loc='upper left')
    ax.grid(alpha=0.4)
    ax.set_ylabel('Pressure (bar)')
    ax.set_xlabel('Mixing ratio')

    ax1 = ax.twiny()

    ax1.plot(np.append(c.T_surf,c.T), np.append(c.P_surf,c.P)/1e6, 'k--', lw=2, label='Temperature')
    ax1.set_xlabel('Temperature (K)')
    ax1.legend(ncol=1,bbox_to_anchor=(1.1, .2), loc='upper left')

    return fig, ax, ax1

###
### I-1: Modern Earth (Inverse)
###

def main_I_1(data_dir=None):

    # First make settings file.
    settings_file_for_climate(
        filename='inputs/settings_I1.yaml', 
        planet_mass=5.972e27,
        planet_radius=6.371e8,
        surface_albedo=0.32, 
        number_of_layers=200, 
        number_of_zenith_angles=4, 
        photon_scale_factor=0.98771561409974729
    )

    c = AdiabatClimate(
        'inputs/species.yaml',
        'inputs/settings_I1.yaml',
        'inputs/stellar_flux_Earth.txt',
    )

    _, _, z, P, T, N2, CO2, H2O = np.loadtxt('inputs/input_Modern_Earth.txt',skiprows=15).T
    P *= 10 # to dynes/cm^2
    f_i = np.empty((len(P),len(c.species_names)))
    f_i[:,c.species_names.index('N2')] = N2
    f_i[:,c.species_names.index('CO2')] = CO2
    f_i[:,c.species_names.index('H2O')] = H2O

    c.rad.surface_albedo = np.ones(len(c.rad.surface_albedo))*0.32
    c.rad.surface_emissivity = np.ones(len(c.rad.surface_emissivity))*0.9

    ASR, OLR = c.TOA_fluxes_dry(P.copy(), T.copy(), f_i.copy())

    write_outputfile(
        c, 
        filename='results/codaccra_photochemclima_I-1a.txt',
        planet_diameter=12742, 
        planet_gravity= 9.81, 
        star_distance=1, 
        star_type='Sun', 
        star_temperature=5772, 
        atmosphere_description='Modern Earth'
    )
    fig, ax, ax1 = plot(c)
    plt.savefig('results/codaccra_photochemclima_I-1a.pdf',bbox_inches='tight')

###
### II-1: Modern Earth (RCE)
###

def main_II_1():

    # First make settings file.
    settings_file_for_climate(
        filename='inputs/settings_II1.yaml', 
        planet_mass=5.972e27,
        planet_radius=6.371e8,
        surface_albedo=0.32, 
        number_of_layers=100, 
        number_of_zenith_angles=4, 
        photon_scale_factor=0.98771561409974729
    )

    c = AdiabatClimate(
        'inputs/species.yaml',
        'inputs/settings_II1.yaml',
        'inputs/stellar_flux_Earth.txt',
    )

    c.rad.surface_albedo = np.ones(len(c.rad.surface_albedo))*0.32
    c.rad.surface_emissivity = np.ones(len(c.rad.surface_emissivity))*0.9
    c.P_top = 1e2
    c.RH = np.ones(len(c.species_names))*1.0
    c.max_rc_iters = 30

    # Input atmospheric composition
    P_i = np.ones(len(c.species_names))*1e-10
    P_i[c.species_names.index('H2O')] = 270 # This is 1 ocean in bars
    P_i[c.species_names.index('N2')] = 0.983002E+00
    P_i[c.species_names.index('CO2')] = 0.393201E-03
    P_i *= 1e6 # convert to dynes/cm^2

    c.T_trop = 180
    c.surface_temperature(P_i, 280)

    assert c.RCE(P_i, c.T_surf, c.T)

    write_outputfile(
        c, 
        filename='results/codaccra_photochemclima_II-1.txt',
        planet_diameter=12742, 
        planet_gravity= 9.81, 
        star_distance=1, 
        star_type='Sun', 
        star_temperature=5772, 
        atmosphere_description='Modern Earth'
    )
    fig, ax, ax1 = plot(c)
    plt.savefig('results/codaccra_photochemclima_II-1.pdf',bbox_inches='tight')

###
### III-7: TRAPPIST-1 g
###

def write_trappist1g_outputfile(c, filename, atmosphere_description):

    planet_gravity = gasgiants.gravity(7.200875e+08, 7.889234e+27, 0.0)/1e2

    write_outputfile(
        c, 
        filename=filename,
        planet_diameter=(7.200875e+08*2)/1e5, 
        planet_gravity=planet_gravity, 
        star_distance=0.04683, 
        star_type='TRAPPIST-1', 
        star_temperature=2566, 
        atmosphere_description=atmosphere_description
    )

def run_trappist1g(P_CO2, T_trop_guess, T_surf_guess, max_rc_iters_convection=5):

    settings_file_for_climate(
        filename='inputs/settings_III7.yaml', 
        planet_mass=7.889234e+27, # 1.321 Earth masses, 
        planet_radius=7.200875e+08, # 1.129 Earth radii
        surface_albedo=0.2, 
        number_of_layers=50, 
        number_of_zenith_angles=4, 
        photon_scale_factor=1.0
    )

    c = AdiabatClimate(
        'inputs/species.yaml',
        'inputs/settings_III7.yaml',
        'inputs/stellar_flux_TRAPPIST1g.txt'
    )

    # Input atmospheric composition
    P_i = np.ones(len(c.species_names))*1e-10
    P_i[c.species_names.index('H2O')] = 270 # This is 1 ocean in bars
    P_i[c.species_names.index('N2')] = 1
    P_i[c.species_names.index('CO2')] = P_CO2
    P_i *= 1e6 # convert to dynes/cm^2

    # Various settings
    c.RH = np.ones(len(c.species_names))*1
    c.P_top = 10
    c.xtol_rc = 1e-8
    c.max_rc_iters = 30
    c.max_rc_iters_convection = max_rc_iters_convection

    # Initial guess
    c.T_trop = T_trop_guess
    c.make_profile(T_surf_guess, P_i)

    # Solve for RCE
    converged = c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below)
    assert converged

    return c

def main_III_7():
    c = run_trappist1g(
        P_CO2=5, 
        T_trop_guess=200, 
        T_surf_guess=300,
        max_rc_iters_convection=-1
    )
    write_trappist1g_outputfile(c, 'results/codaccra_photochemclima_III-7a.txt', 'TRAPPIST1g CO2=5bar')
    fig, ax, ax1 = plot(c)
    ax.set_ylim(12,2e-5)
    ax.set_xlim(1e-8,2)
    ax1.set_xlim(150,300)
    plt.savefig('results/codaccra_photochemclima_III-7a.pdf',bbox_inches='tight')

    c = run_trappist1g(
        P_CO2=10, 
        T_trop_guess=205, 
        T_surf_guess=300,
        max_rc_iters_convection=-1
    )
    write_trappist1g_outputfile(c, 'results/codaccra_photochemclima_III-7b.txt', 'TRAPPIST1g CO2=10bar')
    fig, ax, ax1 = plot(c)
    ax.set_ylim(12,2e-5)
    ax.set_xlim(1e-8,2)
    ax1.set_xlim(150,300)
    plt.savefig('results/codaccra_photochemclima_III-7b.pdf',bbox_inches='tight')

###
### Setup for all models
###

def setup():
    species_file_for_climate(
        filename='inputs/species.yaml', 
        species=['H2O','CO2','N2'], 
        condensates=['H2O','CO2']
    )

###
### main
###

def main():
    setup()
    main_I_1()
    main_II_1()
    main_III_7()

if __name__ == '__main__':
    main()
