import numpy as np
import xarray as xr
import pickle

class BasicModel:
    def __init__(self, nx=None, dx=None, **kwargs):
        self.nx = nx
        self.dx = dx
        self.length = self.nx*self.dx
        self.distance_downstream = np.arange(self.nx)*self.dx + self.dx/2.
        self.distance_upstream = self.nx*self.dx - self.distance_downstream
        
        self.upstream_end = 0  # ghost point, only for upstream sediment feeding
        self.downstream_end = self.nx-1  # fixed elevation
        self.core_nodes = np.where(np.logical_and(np.arange(self.nx) > self.upstream_end, np.arange(self.nx) < self.downstream_end))
        self.core_nodes_and_downstream_end = np.where(np.arange(self.nx) > self.upstream_end)
        self.core_nodes_and_upstream_end = np.where(np.arange(self.nx) < self.downstream_end)
        
        self.elevation = np.zeros(self.nx)
        self.bedrock_elevation = np.zeros(self.nx)
        self.sediment_thickness = np.zeros(self.nx)
        
        self.slope = np.zeros(self.nx)
        self.slope_r = np.zeros(self.nx)
        self.slope_s = np.zeros(self.nx)
        
        self.uplift_rate = kwargs['uplift_rate']
        
        self.sec_per_year = 31556952.
        
        self.water_density = 1000
        self.sediment_density = 2650
        self.R = (self.sediment_density - self.water_density)/self.water_density
        self.g = 9.81
        
        self.grain_size = np.zeros(self.nx)
        try:
            self.grain_size = kwargs['grain_size']
        except:
            self.grain_size[:] = 20e-3
        
        self.drainage_area = np.zeros(self.nx)
        try:
            drainage_area_head = kwargs['drainage_area_head']
        except KeyError:
            drainage_area_head = 0.
        try:
            drainage_area_distance_coefficient = kwargs['drainage_area_distance_coefficient']
        except KeyError:
            drainage_area_distance_coefficient = 30
        try:
            drainage_area_distance_exponent = kwargs['drainage_area_distance_exponent']
        except KeyError:
            drainage_area_distance_exponent = 1.5
        self.drainage_area = drainage_area_distance_coefficient \
            * np.power(self.distance_downstream+np.power(drainage_area_head/drainage_area_distance_coefficient, 1./drainage_area_distance_exponent), drainage_area_distance_exponent)
        
        self.water_discharge = np.zeros(self.nx)
        try:
            precipitation_rate = kwargs['precipitation_rate']
        except KeyError:
            precipitation_rate = 1
        self.water_discharge = precipitation_rate * self.drainage_area
        
        self.channel_width = np.zeros(self.nx)
        try:
            width_discharge_coefficient = kwargs['width_discharge_coefficient']
        except KeyError:
            width_discharge_coefficient = 0.005
        try:
            width_discharge_exponent = kwargs['width_discharge_exponent']
        except KeyError:
            width_discharge_exponent = 0.5
        self.channel_width = width_discharge_coefficient * np.power(self.water_discharge, width_discharge_exponent)
        
        self.cover_factor = np.zeros(self.nx)
        self.cover_factor_lower = 0.05
        self.cover_factor_upper = 0.95
        
        self.bedrock_roughness_scale = 1
        
        try:
            self.cover_factor_approach = kwargs['cover_factor_approach']
            if self.cover_factor_approach not in ['SPACE', 'MRSAA']:
                raise ValueError('cover_factor_approach has to be SPACE or MRSAA')
        except KeyError:
            self.cover_factor_approach = 'MRSAA'
        
        if self.cover_factor_approach == 'MRSAA':
            self.downstream_end_limit = self.bedrock_roughness_scale * (1-self.cover_factor_lower) / (self.cover_factor_upper-self.cover_factor_lower)
        else:
            self.downstream_end_limit = self.bedrock_roughness_scale
        
        self.sediment_transport_capacity_per_unit_width = np.zeros(self.nx)
        self.sediment_flux_per_unit_width = np.zeros(self.nx)
        self.sediment_flux = np.zeros(self.nx)
        
        # sediment feed along the profile
        try:
            self.sediment_feed_rate = kwargs['sediment_feed_rate']
        except KeyError:
            self.sediment_feed_rate = np.zeros(self.nx)
        
        self.bedrock_erosion_rate = np.zeros(self.nx)
        
        self.sediment_porosity = 0.35
        
        try:
            self.bedrock_erodibility_coefficient = kwargs['bedrock_erodibility_coefficient']
        except KeyError:
            self.bedrock_erodibility_coefficient = 1e-4
        self.bedrock_erosion_threshold = 0
        
        self.abrasion_coefficient = np.zeros(self.nx)
        
        try:
            self.bedrock_erosion_approach = kwargs['bedrock_erosion_approach']
            if self.bedrock_erosion_approach not in ['stream-power', 'saltation-abrasion']:
                raise ValueError('bedrock_erosion_approach has to be stream-power or saltation-abrasion')
        except KeyError:
            self.bedrock_erosion_approach = 'stream-power'
    
        A_s = np.zeros((self.nx, self.nx))
        A_s[self.nx-1, self.nx-1] = 1
        A_s[self.nx-1, self.nx-2] = -1
        for i in range(1, self.nx-1): # first node is a ghost node for sediment feed, no slope
            A_s[i, i+1] = 1
            A_s[i, i] = -1
        self.slope_matrix = -A_s/self.dx
        
        A_f = np.zeros((self.nx, self.nx))
        try:
            au = kwargs['sediment_flux_derivative_scheme']
        except KeyError:
            au = 1  # 0 for downstream, 1 for upstream
        A_f[0, 0] = -1
        A_f[0, 1] = 1
        A_f[self.nx-1, self.nx-2] = -1
        A_f[self.nx-1, self.nx-1] = 1
        for i in range(1, self.nx-1):
            A_f[i, i-1]=-au
            A_f[i, i]=2*au-1
            A_f[i, i+1]=1-au
        self.flux_gradient_matrix = A_f/self.dx

        self.model_state = None
        self.update_model_state()
        
        self.saved_results = None
    
    def update_slope(self):
        self.slope_r = np.dot(self.slope_matrix, self.bedrock_elevation)
        self.slope_s = np.dot(self.slope_matrix, self.sediment_thickness)
        self.slope = self.slope_r + self.slope_s

    def update_cover_factor_space(self):
        self.cover_factor = 1 - np.exp(-self.sediment_thickness/self.bedrock_roughness_scale)
        
    def update_cover_factor_mrsaa(self):
        p0 = self.cover_factor_lower
        p1 = self.cover_factor_upper
        chi = self.sediment_thickness / self.bedrock_roughness_scale
        self.cover_factor[np.where(chi > (1-p0)/(p1-p0))] = 1
        self.cover_factor[np.where(chi <= (1-p0)/(p1-p0))] = p0 + (p1-p0)*chi[np.where(chi <= (1-p0)/(p1-p0))]
        self.cover_factor = (self.cover_factor-p0) / (1-p0)
        
    def update_bedrock_erosion_rate_stream_power(self):
        K_r = self.bedrock_erodibility_coefficient
        q = self.water_discharge / self.channel_width
        S = self.slope
        w_cr = self.bedrock_erosion_threshold
        p = self.cover_factor
        
        tmp = (K_r * q * S - w_cr) * (1-p)
        self.bedrock_erosion_rate[self.core_nodes] = tmp[self.core_nodes]
        self.bedrock_erosion_rate[self.downstream_end] = 0
        
    def update_bedrock_erosion_rate_saltation_abrasion(self):
        beta = self.abrasion_coefficient
        q_s = self.sediment_flux_per_unit_width
        p = self.cover_factor
        
        tmp = beta * q_s * (1-p)
        self.bedrock_erosion_rate[self.core_nodes] = tmp[self.core_nodes]
        self.bedrock_erosion_rate[self.downstream_end] = 0
        
    def update_bedrock_elevation(self, dt):
        domain = self.core_nodes
        self.bedrock_elevation[domain] += dt * (self.uplift_rate - self.bedrock_erosion_rate[domain])

    def update_model_state(self):
        if self.model_state is None:
            # Initialize model state
            self.model_state = xr.Dataset()
            x = self.nx*self.dx - (np.arange(self.nx)*self.dx + self.dx/2)
            self.model_state.coords['x'] = (('x'), x, {'units': 'm', 'long_name': 'upstream distance'})
            self.model_state['bedrock_elevation'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['sediment_thickness'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['cover_factor'] = (('x'), np.zeros(self.nx), {'units': '1'})
            self.model_state['channel_width'] = (('x'), np.zeros(self.nx), {'units': 'm'})
            self.model_state['sediment_load'] = (('x'), np.zeros(self.nx), {'units': 'm3 year-1', 'long_name': 'volumetric sediment transport rate'})
            self.model_state['sediment_load_per_unit_width'] = (('x'), np.zeros(self.nx), {'units': 'm2 year-1'})
            self.model_state['bedrock_erosion_rate'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['sediment_entrainment_rate'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})
            self.model_state['sediment_deposition_rate'] = (('x'), np.zeros(self.nx), {'units': 'm year-1'})

            self.model_state['slope'] = (('x'), np.zeros(self.nx), {'units': '1'})
            self.model_state['slope_r'] = (('x'), np.zeros(self.nx), {'units': '1'})
            self.model_state['slope_s'] = (('x'), np.zeros(self.nx), {'units': '1'})

        self.model_state['bedrock_elevation'].data = self.bedrock_elevation
        self.model_state['sediment_thickness'].data = self.sediment_thickness
        self.model_state['cover_factor'].data = self.cover_factor
        self.model_state['channel_width'].data = self.channel_width
        self.model_state['sediment_load'].data = self.sediment_flux
        self.model_state['sediment_load_per_unit_width'].data = self.sediment_flux_per_unit_width
        self.model_state['bedrock_erosion_rate'].data = self.bedrock_erosion_rate
        try:
            self.model_state['sediment_entrainment_rate'].data = self.sediment_entrainment_rate
        except:
            pass
        try:
            self.model_state['sediment_deposition_rate'].data = self.sediment_deposition_rate
        except:
            pass

        self.model_state['slope'].data = self.slope
        self.model_state['slope_r'].data = self.slope_r
        self.model_state['slope_s'].data = self.slope_s
    
    def save_model_state(self, t, var_list=None):
        self.update_model_state()
        if var_list is None:
            var_list = ['bedrock_elevation', 'sediment_thickness']
        state_to_save = self.model_state.copy(deep=True)
        
        for k in state_to_save.keys():
            if k not in var_list:
                state_to_save = state_to_save.drop_vars(k)
        
        state_to_save = state_to_save.expand_dims('time')
        state_to_save.coords['time'] = np.array([t], dtype=float)
        if self.saved_results is None:
            self.saved_results = state_to_save
        else:
            self.saved_results = xr.concat([self.saved_results, state_to_save], 'time')
            
    def write_saved_results_to_file(self, filename):
        #self.saved_results.attrs['input_params'] = json.dumps(self._params)
        self.saved_results['time'].attrs['units'] = 'years'
        #self.saved_results['time'].attrs['calendar'] = '365_day'
        self.saved_results.to_netcdf(filename)

        
class ErosionDeposition(BasicModel):
    def __init__(self, **kwargs):
        BasicModel.__init__(self, **kwargs)
        
        self.sediment_entrainment_rate = np.zeros(self.nx)
        self.sediment_deposition_rate = np.zeros(self.nx)
        
        try:
            self.sediment_erodibility_coefficient = kwargs['sediment_erodibility_coefficient']
        except KeyError:
            self.sediment_erodibility_coefficient = 1e-2
        self.sediment_entrainment_threshold = 0
        
        try:
            self.settling_velocity = kwargs['settling_velocity']
        except KeyError:
            self.settling_velocity = 1
        
    def update_sediment_entrainment_rate(self):
        K_s = self.sediment_erodibility_coefficient
        q = self.water_discharge / self.channel_width
        S = self.slope
        w_cs = self.sediment_entrainment_threshold
        p = self.cover_factor
        
        tmp = (K_s * q * S - w_cs) * p
        domain = self.core_nodes_and_downstream_end
        self.sediment_entrainment_rate[domain] = tmp[domain]
    
    def update_sediment_deposition_rate(self):
        Q_s = self.sediment_flux
        Q = self.water_discharge
        V = self.settling_velocity
        
        tmp = Q_s/Q*V
        domain = self.core_nodes_and_downstream_end
        self.sediment_deposition_rate[domain] = tmp[domain]
        
    def update_sediment_thickness(self, dt):
        domain = self.core_nodes_and_downstream_end
        E_s = self.sediment_entrainment_rate
        D_s = self.sediment_deposition_rate
        
        self.sediment_thickness[domain] += dt * (D_s[domain] - E_s[domain]) / (1-self.sediment_porosity)
        
        if self.sediment_thickness[self.downstream_end] > self.downstream_end_limit:
            self.sediment_thickness[self.downstream_end] = self.downstream_end_limit
        
        self.sediment_thickness[np.where(self.sediment_thickness < 0)] = 0
    
    def update_sediment_flux(self):
        E_r = self.bedrock_erosion_rate
        E_s = self.sediment_entrainment_rate
        D_s = self.sediment_deposition_rate
        
        self.sediment_flux[self.upstream_end] = self.sediment_feed_rate[self.upstream_end]
        for i in range(self.upstream_end+1, self.downstream_end+1):
            self.sediment_flux[i] = (self.sediment_flux[i-1] + E_s[i]*self.dx*self.channel_width[i] + E_r[i]*self.dx*self.channel_width[i] + self.sediment_feed_rate[i])\
                /(1+self.settling_velocity*self.dx*self.channel_width[i]/self.water_discharge[i]) # conserve mass over width not dx 
       
        #self.sediment_flux[self.downstream_end] = self.sediment_flux[-2]
        self.sediment_flux_per_unit_width = self.sediment_flux / self.channel_width
    
    def run_one_step(self, dt):
        self.update_slope()
        if self.cover_factor_approach == 'MRSAA':
            self.update_cover_factor_mrsaa()
        elif self.cover_factor_approach == 'SPACE':
            self.update_cover_factor_space()
        self.update_sediment_flux()
        if self.bedrock_erosion_approach == 'stream-power':
            self.update_bedrock_erosion_rate_stream_power()
        elif self.bedrock_erosion_approach == 'saltation-abrasion':
            self.update_bedrock_erosion_rate_saltation_abrasion()
        self.update_sediment_entrainment_rate()
        self.update_sediment_deposition_rate()
        
        self.update_bedrock_elevation(dt)
        
        self.update_sediment_thickness(dt)
        
        self.elevation = self.bedrock_elevation + self.sediment_thickness
        
class ExnerType(BasicModel):
    def __init__(self, **kwargs):
        BasicModel.__init__(self, **kwargs)
        
        try:
            self.chezy_resistance_coefficient = kwargs['chezy_resistance_coefficient']
        except KeyError:
            self.chezy_resistance_coefficient = 5
            
        self.shields_number = np.zeros(self.nx)
        #self.threshold_shields_number = 0.0495
        self.threshold_shields_number = 0.
        self.flood_intermittency = 0.05

    def update_shields_number(self):
        domain = self.core_nodes_and_downstream_end
        S = self.slope
        aa = np.power(self.chezy_resistance_coefficient, 2) * self.g * np.power(self.channel_width[domain], 2)
        bb = np.power(np.power(self.water_discharge[domain]/self.sec_per_year/self.flood_intermittency, 2)/aa, 1./3.)
        cc = bb * np.power(S[domain], 2./3.) / (self.R * self.grain_size[domain])
        self.shields_number[domain] = cc
        
    def update_sediment_flux(self):
        tmp_shields_number = self.shields_number.copy()
        tmp_shields_number[np.where(tmp_shields_number < self.threshold_shields_number)] = self.threshold_shields_number
        
        domain = self.core_nodes_and_downstream_end
        self.sediment_transport_capacity_per_unit_width[domain] = 4 * np.sqrt(self.R * self.g * self.grain_size[domain]) \
            * self.grain_size[domain] * np.power((tmp_shields_number[domain] - self.threshold_shields_number), 1.5)
        self.sediment_transport_capacity_per_unit_width *= self.sec_per_year*self.flood_intermittency
        
        self.sediment_flux_per_unit_width = self.cover_factor * self.sediment_transport_capacity_per_unit_width
        self.sediment_flux_per_unit_width[self.upstream_end] = self.sediment_feed_rate[self.upstream_end] / self.channel_width[self.upstream_end]
        #self.sediment_flux_per_unit_width[self.downstream_end] = 1e9
        self.sediment_flux = self.sediment_flux_per_unit_width * self.channel_width
        
    def update_sediment_thickness(self, dt):
        p = self.cover_factor
        if self.cover_factor_approach == 'SPACE':
            p += 1e-3
        if self.cover_factor_approach == 'MRSAA':
            p = p*(1-self.cover_factor_lower)+self.cover_factor_lower
        
        domain = self.core_nodes_and_downstream_end
        #domain = self.core_nodes
        left_hand = -np.dot(self.flux_gradient_matrix, self.sediment_flux)
        left_hand += self.bedrock_erosion_rate * self.channel_width  # conserve mass over width not dx 
        left_hand[domain] += self.sediment_feed_rate[domain]/self.dx # sediment_feed_rate[upstream_end] has been included in sediment_flux already
        self.sediment_thickness[domain] += dt*(left_hand[domain]/((1-self.sediment_porosity)*(p[domain])*self.channel_width[domain]))
        
        if self.sediment_thickness[self.downstream_end] > self.downstream_end_limit:
            self.sediment_thickness[self.downstream_end] = self.downstream_end_limit
        
        self.sediment_thickness[np.where(self.sediment_thickness < 0)] = 0
    
    def run_one_step(self, dt):
        self.update_slope()
        if self.cover_factor_approach == 'MRSAA':
            self.update_cover_factor_mrsaa()
        elif self.cover_factor_approach == 'SPACE':
            self.update_cover_factor_space()
        self.update_shields_number()
        self.update_sediment_flux()
        if self.bedrock_erosion_approach == 'stream-power':
            self.update_bedrock_erosion_rate_stream_power()
        elif self.bedrock_erosion_approach == 'saltation-abrasion':
            self.update_bedrock_erosion_rate_saltation_abrasion()
        
        #import pdb;pdb.set_trace()
        self.update_bedrock_elevation(dt)
        self.update_sediment_thickness(dt)
        
        self.elevation = self.bedrock_elevation + self.sediment_thickness



class Variable():
    def __init__(self, value=np.array([]), time=np.array([]), unit=None):
        self.value = value
        self.time = time
        self.unit = unit

class ModelResults():
    def __init__(self):
        self.variables = {}

    def add_field(self, field, unit=None):
        self.variables[field] = Variable(unit=unit)

    def add_value(self, field, z, t):
        if field not in self.variables:
            raise KeyError(field)
        if len(self.variables[field].value) == 0:
            self.variables[field].value = np.array([z])
            self.variables[field].time = np.array([t])
        else:
            self.variables[field].value = np.append(self.variables[field].value, [z], axis=0)
            self.variables[field].time = np.append(self.variables[field].time, t)


def save_object(obj, filename):
    with open(filename, 'xb') as out_file:  # Fail if file exists.
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)