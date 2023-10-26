import numpy as np
import math
import os
import sys
import time

from basic_model import *

### set up save location
savepath = '/home/lai/data/tool_and_cover_1D/model_comparison/HSR_length_Qw_V_response_time_20230921_only_full_cover'
time.sleep(np.random.default_rng().uniform(0, 5))  # avoid multiple tasks create the folder at the same time
if not os.path.isdir(savepath):
    try:  # avoid error when multiple tasks try to create the folder at the same time
        os.mkdir(savepath)
    except:
        pass

use_numba = True

water_discharge_list = 25*np.array([1e5, 2e5, 5e5, 1e6, 2e6])
length_list = [100, 200, 500, 1e3, 2e3, 5e3, 1e4]
V_list = [1, 10, 100, 1000, 10000]
#slope_list = np.linspace(0, 0.05, num=21)[1:]
#upstream_feed_rate_list = np.linspace(0, 1e6, num=21)[1:]

def slurm_id_to_values(idx, a_array, b_array, c_array):
    count = 0
    for a in a_array:
        for b in b_array:
            for c in c_array:
                if count == idx:
                    return a, b, c
                else:
                    count += 1
    
    return None, None, None

input_idx = int(sys.argv[1])

#tmp_slope, tmp_feed_rate = slurm_id_to_values(input_idx, slope_list, upstream_feed_rate_list)
tmp_length, tmp_water_discharge, tmp_V = slurm_id_to_values(input_idx, length_list, water_discharge_list, V_list)
#print(tmp_length, tmp_water_discharge, tmp_V)

if tmp_length is None:
    sys.exit(0)

#if tmp_slope == 0 or tmp_feed_rate == 0:
#    sys.exit(0)

outfile = 'HSR_length_{}_Qw_{}_V_{}.nc'.format(tmp_length, tmp_water_discharge, tmp_V)

ed_result_exists = False
exn_result_exists = False
if os.path.exists(os.path.join(savepath, 'ed_{}'.format(outfile))):
    ed_result_exists = True
    print('{} exists.'.format(os.path.join(savepath, 'ed_{}'.format(outfile))))
    #raise FileExistsError('{} exists.'.format(os.path.join(savepath, 'ed_{}'.format(outfile))))
if os.path.exists(os.path.join(savepath, 'exn_{}'.format(outfile))):
    exn_result_exists = True
    print('{} exists.'.format(os.path.join(savepath, 'exn_{}'.format(outfile))))
    #raise FileExistsError('{} exists.'.format(os.path.join(savepath, 'exn_{}'.format(outfile))))
###

### start model run

length = tmp_length
dx = length/100
nx = int(length/dx) + 1

uplift_rate = 0.00
bedrock_erodibility_coefficient = 1e-6
abrasion_coefficient = 0
upstream_feed_rate = 0.1*tmp_water_discharge # Ksc*Q*S
denudation_rate = 0.0

grain_size = 10e-3

## the following parameters will result in same steady state slope
## for the given water discharge and channel width
channel_width = 25
#water_discharge = 2.5e7
water_discharge = tmp_water_discharge
sediment_capacity_coefficient = 1
#sediment_erodibility_coefficient = 5e-6
#settling_velocity = 5
settling_velocity = tmp_V
sediment_erodibility_coefficient = sediment_capacity_coefficient*settling_velocity/(water_discharge/channel_width)

tau_ed = length/sediment_erodibility_coefficient/(water_discharge/channel_width)
tau_exn = length**2/sediment_capacity_coefficient/(water_discharge/channel_width)
if tau_ed < tau_exn:
        tau_ed = tau_exn

steady_slope = upstream_feed_rate/water_discharge/sediment_capacity_coefficient

slope = 0.01
initial_bed_elev = np.arange(nx-1, -1, -1)*dx*slope
#initial_bed_elev = 0
initial_bed_elev_ed = initial_bed_elev
initial_bed_elev_exn = initial_bed_elev

#initial_sediment_thk = np.arange(nx-1, -1, -1)*dx*0.0165-initial_bed_elev
initial_sediment_thk = 0.
initial_sediment_thk_ed = initial_sediment_thk
initial_sediment_thk_exn = initial_sediment_thk

ed = ErosionDeposition(nx=nx, dx=dx, uplift_rate=uplift_rate,
              bedrock_erosion_approach='saltation-abrasion', sediment_erodibility_coefficient=sediment_erodibility_coefficient, settling_velocity=settling_velocity, CFL_limit=0.001)
exn = ExnerType(nx=nx, dx=dx, uplift_rate=uplift_rate,
              bedrock_erosion_approach='saltation-abrasion', sediment_capacity_coefficient=sediment_capacity_coefficient, sediment_flux_derivative_scheme=1, CFL_limit=0.001)

#sediment_feed_rate = ed.drainage_area*denudation_rate
#sediment_feed_rate[1:] = sediment_feed_rate[1:] - sediment_feed_rate[:-1]
sediment_feed_rate = np.zeros(ed.nx)
sediment_feed_rate[0] += upstream_feed_rate
#sediment_feed_rate[:] = 0

ed.grain_size[:] = grain_size
ed.abrasion_coefficient[:] = abrasion_coefficient
ed.sediment_feed_rate = sediment_feed_rate.copy()
ed.channel_width[:] = channel_width
ed.water_discharge[:] = water_discharge
ed.bedrock_elevation[:] = initial_bed_elev_ed
ed.sediment_thickness[:] = initial_sediment_thk_ed
ed.elevation = ed.bedrock_elevation + ed.sediment_thickness
ed.cover_factor_lower = 1e-3

exn.grain_size[:] = grain_size
exn.abrasion_coefficient[:] = abrasion_coefficient
exn.sediment_feed_rate = sediment_feed_rate.copy()
exn.channel_width[:] = channel_width
exn.water_discharge[:] = water_discharge
exn.bedrock_elevation[:] = initial_bed_elev_exn
exn.sediment_thickness[:] = initial_sediment_thk_exn
exn.elevation = exn.bedrock_elevation + exn.sediment_thickness
exn.cover_factor_lower = 1e-3

var_list = ['sediment_thickness', 'cover_factor',
            'sediment_load_per_unit_width']


# Erosion-deposition
if not ed_result_exists:
    runtime = 0.5*tau_ed
    dt = np.power(10, np.floor(np.log10(tau_ed/1000)))
    nt = int(runtime/dt)
    save_dt = np.power(10, np.floor(np.log10(tau_ed/100)))
    save_time_limit = 50000

    # sedimentation
    ed.save_model_state(0, var_list=var_list)
    save_time = save_dt
    for i in range(nt):
        curr_t = (i+1)*dt
        ed.run_one_step(dt, use_numba=use_numba)

        if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
            save_time += save_dt
            if save_time <= save_time_limit:
                ed.save_model_state(curr_t, var_list=var_list)
    
    # evacuation
    sed_surf_elev = np.arange(nx-1, -1, -1)*dx*steady_slope
    ed.sediment_thickness[:] = sed_surf_elev - ed.bedrock_elevation
    ed.elevation = ed.bedrock_elevation + ed.sediment_thickness

    # spin-up for evacuation
    for i in range(int(0.2*tau_ed/dt)):
        ed.run_one_step(dt, use_numba=use_numba)

    # shut down input
    ed.sediment_feed_rate[:] = 0

    save_time = save_dt
    for i in range(nt):
        curr_t = (i+1)*dt
        ed.run_one_step(dt, use_numba=use_numba)

        if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
            save_time += save_dt
            if save_time <= save_time_limit:
                ed.save_model_state(curr_t+runtime, var_list=var_list)
    
    #print(os.path.join(savepath, 'ed_{}'.format(outfile)))
    ed.write_saved_results_to_file(os.path.join(savepath, 'ed_{}'.format(outfile)))

    print("Erosion-deposition done")

# Exner-type
if not exn_result_exists:
    runtime = 0.5*tau_exn
    #dt = 1e-7
    dt = np.power(10, np.floor(np.log10(tau_exn/1000)))
    nt = int(runtime/dt)
    save_dt = np.power(10, np.floor(np.log10(tau_exn/100)))
    save_time_limit = 50000

    # sedimentation
    exn.save_model_state(0, var_list=var_list)
    save_time = save_dt
    for i in range(nt):
        #print(i/nt*100.)
        curr_t = (i+1)*dt
        exn.run_one_step(dt, use_numba=use_numba)

        if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
            save_time += save_dt
            if save_time <= save_time_limit:
                exn.save_model_state(curr_t, var_list=var_list)
    
    # evacuation
    sed_surf_elev = np.arange(nx-1, -1, -1)*dx*steady_slope
    exn.sediment_thickness[:] = sed_surf_elev - exn.bedrock_elevation
    exn.elevation = exn.bedrock_elevation + exn.sediment_thickness

    # spin-up for evacuation
    for i in range(int(0.2*tau_exn/dt)):
        exn.run_one_step(dt, use_numba=use_numba)

    # shut down input
    exn.sediment_feed_rate[:] = 0

    save_time = save_dt
    for i in range(nt):
        curr_t = (i+1)*dt
        exn.run_one_step(dt, use_numba=use_numba)

        if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
            save_time += save_dt
            if save_time <= save_time_limit:
                exn.save_model_state(curr_t+runtime, var_list=var_list)
    
    #print(os.path.join(savepath, 'exn_{}'.format(outfile)))
    exn.write_saved_results_to_file(os.path.join(savepath, 'exn_{}'.format(outfile)))

    print("Exner-type done")

print("All done")