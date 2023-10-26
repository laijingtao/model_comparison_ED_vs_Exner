import numpy as np
import math
import os
import sys
import time

from basic_model import *

### set up save location
savepath = '/home/lai/data/tool_and_cover_1D/model_comparison/HSR_periodic_feed_20220826'
time.sleep(np.random.default_rng().uniform(0, 5))  # avoid multiple tasks create the folder at the same time
if not os.path.isdir(savepath):
    try:  # avoid error when multiple tasks try to create the folder at the same time
        os.mkdir(savepath)
    except:
        pass

duration_list = [1, 2, 5, 10, 25, 50, 100, 200]
upstream_feed_rate_list = np.linspace(0, 1e6, num=41)[1:]

def slurm_id_to_values(idx, a_array, b_array):
    count = 0
    for a in a_array:
        for b in b_array:
            if count == idx:
                return a, b
            else:
                count += 1
    
    return None, None

input_idx = int(sys.argv[1])

tmp_duration, tmp_feed_rate = slurm_id_to_values(input_idx, duration_list, upstream_feed_rate_list)

if tmp_duration is None:
    sys.exit(0)

if tmp_feed_rate == 0:
    sys.exit(0)

outfile = 'HSR_periodic_{}_feed_rate_{}.nc'.format(tmp_duration, tmp_feed_rate)

if os.path.exists(os.path.join(savepath, 'ed_{}'.format(outfile))):
    raise FileExistsError('{} exists.'.format(os.path.join(savepath, 'ed_{}'.format(outfile))))
if os.path.exists(os.path.join(savepath, 'exn_{}'.format(outfile))):
    raise FileExistsError('{} exists.'.format(os.path.join(savepath, 'exn_{}'.format(outfile))))
###

### start model run

length = 1e3
dx = length/100
nx = int(length/dx) + 1

uplift_rate = 0.00
bedrock_erodibility_coefficient = 1e-6
abrasion_coefficient = 0
upstream_feed_rate = tmp_feed_rate
denudation_rate = 0.0

grain_size = 10e-3

## the following parameters will result in same steady state slope
## for the given water discharge and channel width
channel_width = 25
water_discharge = 2.5e7
sediment_transport_coefficient = 1
sediment_erodibility_coefficient = 5e-6
settling_velocity = 5

chezy_resistance_coefficient = 4/1.65/sediment_transport_coefficient

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
              bedrock_erosion_approach='saltation-abrasion', sediment_erodibility_coefficient=sediment_erodibility_coefficient, settling_velocity=settling_velocity)
exn = ExnerType(nx=nx, dx=dx, uplift_rate=uplift_rate,
              bedrock_erosion_approach='saltation-abrasion', chezy_resistance_coefficient=chezy_resistance_coefficient, sediment_flux_derivative_scheme=1)

#sediment_feed_rate = ed.drainage_area*denudation_rate
#sediment_feed_rate[1:] = sediment_feed_rate[1:] - sediment_feed_rate[:-1]
sediment_feed_rate = np.zeros(ed.nx)
sediment_feed_rate[0] += upstream_feed_rate
#sediment_feed_rate[:] = 0

def is_feeding_phase(curr_t, duration):
    if int((curr_t//duration))%2 == 0:
        return True
    else:
        return False

ed.grain_size[:] = grain_size
ed.abrasion_coefficient[:] = abrasion_coefficient
ed.sediment_feed_rate = sediment_feed_rate.copy()
ed.channel_width[:] = channel_width
ed.water_discharge[:] = water_discharge
ed.bedrock_elevation[:] = initial_bed_elev_ed
ed.sediment_thickness[:] = initial_sediment_thk_ed
ed.elevation = ed.bedrock_elevation + ed.sediment_thickness

exn.grain_size[:] = grain_size
exn.abrasion_coefficient[:] = abrasion_coefficient
exn.sediment_feed_rate = sediment_feed_rate.copy()
exn.channel_width[:] = channel_width
exn.water_discharge[:] = water_discharge
exn.bedrock_elevation[:] = initial_bed_elev_exn
exn.sediment_thickness[:] = initial_sediment_thk_exn
exn.elevation = exn.bedrock_elevation + exn.sediment_thickness

var_list = ['bedrock_elevation', 'sediment_thickness', 'cover_factor',
            'sediment_load_per_unit_width']

spin_up = True

# Erosion-deposition
runtime = int(tmp_duration * 20)
if runtime < 500:
    runtime = 500
if runtime > 1000:
    runtime = 1000
dt = 0.01
nt = int(runtime/dt)
save_dt = 0.1

now_feeding = True

if spin_up:
    for i in range(int(tmp_duration * 100/dt)):
        curr_t = (i+1)*dt
        
        if is_feeding_phase(curr_t, tmp_duration):
            if not now_feeding:
                ed.sediment_feed_rate = sediment_feed_rate.copy()
                now_feeding = True
        else:
            if now_feeding:
                ed.sediment_feed_rate[:] = 0
                now_feeding = False

        ed.run_one_step(dt)

ed.save_model_state(0, var_list=var_list)
save_time = save_dt
for i in range(nt):
    curr_t = (i+1)*dt
    
    if is_feeding_phase(curr_t, tmp_duration):
        if not now_feeding:
            ed.sediment_feed_rate = sediment_feed_rate.copy()
            now_feeding = True
    else:
        if now_feeding:
            ed.sediment_feed_rate[:] = 0
            now_feeding = False

    ed.run_one_step(dt)

    if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
        save_time += save_dt
        ed.save_model_state(curr_t, var_list=var_list)

ed.write_saved_results_to_file(os.path.join(savepath, 'ed_{}'.format(outfile)))


#sys.exit(0)
# Exner-type
runtime = int(tmp_duration * 20)
#if runtime < 500:
#    runtime = 500
if runtime > 500:
    runtime = 500
dt = 0.000001
nt = int(runtime/dt)
save_dt = 0.1

exn.save_model_state(0, var_list=var_list)
save_time = save_dt
now_feeding = True
for i in range(nt):
    curr_t = (i+1)*dt

    if is_feeding_phase(curr_t, tmp_duration):
        if not now_feeding:
            exn.sediment_feed_rate = sediment_feed_rate.copy()
            now_feeding = True
    else:
        if now_feeding:
            exn.sediment_feed_rate[:] = 0
            now_feeding = False

    exn.run_one_step(dt)

    if math.isclose(curr_t, save_time, abs_tol=dt/1e3):
        save_time += save_dt
        exn.save_model_state(curr_t, var_list=var_list)

exn.write_saved_results_to_file(os.path.join(savepath, 'exn_{}'.format(outfile)))
