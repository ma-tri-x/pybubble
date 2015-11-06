import abc


class Abc_bubble_models(object):
    __metaclass__=abc.ABCMeta
    
    @abc.abstractmethod
    def get_parameters(self):
        """
        should return dictionary with parameters as items and properties value,increment of each item.
        """
        return
    
    @abc.abstractmethod
    def get_init_cond(self):
        """
        should return dictionary with initial conditions as items and properties value,increment of each item.
        """
        return

    @abc.abstractmethod
    def read_values_from_file(self,filename):
        """
        read_values_from_file(filename):
        should read values of parameters and initial conditions with 
        properties value and increment from file into dictionary.
        """
        return

    @abc.abstractmethod
    def refresh_values(self):
        """
        should refresh dictionary with initial conditions as items and properties value,increment of each item.
        """
        return
    
    @abc.abstractmethod
    def integrate(self):
        """
        should integrate the equation(s).
        """
        return
    
    @abc.abstractmethod
    def return_Rt_curve(self):
        """
        should hand over an array R(t).
        """
        return
    
    @abc.abstractmethod
    def return_pt_curve(self):
        """
        should hand over an array p(t).
        """
        return
    
class gilmore(Abc_bubble_models):
    def __init__(self,filename):
        self.input_dict = filename
        self.parameter_list = self.read_values_from_file(self.input_dict)
        
    def get_parameters(self):
        
    
    def get_init_cond(self):
        

    def read_values_from_file(self,filename):
        

    def refresh_values(self):

    
    def integrate(self):

    
    def return_Rt_curve(self):

    
    def return_pt_curve(self):

    #   Rstart=747e-6 #0.00100015 #m
   #RdotStart=0 #1470. #m/s
   #Rn=250e-6     #m
   #deltaT=1e-7  #s
   #Tstart=0.    #s
   #Tend=120e-6  #s
   #patm=101315  #Pa
   #pac=0.       #Pa
   #f=10        # Hz
   #bVan=0.0000364 # in m³/mol: air according to http://de.wikipedia.org/wiki/Van-der-Waals-Gleichung
   ##mu=0.001     # Pa s (I think)
   ##mu=0.00089008220776922 # Pa s # this is correct for 25°C. taken from: http://www.peacesoftware.de/einigewerte/wasser_dampf_e.html
   #mu=1.002e-3  # was: 0.0000186 # Pa s. correct value at 20°C:1.002mPa*s taken from: http://en.wikipedia.org/wiki/Viscosity
   #sigma=0.0725 # Pa m (I think)
   #BTait=3046e5 # Pa
   #nTait=7.15   # no unit
   
   ## kappa: fractions don't work!!! Only floating point!
   ## down below for gnuplot: put kappa in manually!
   ##kappa=1.3333333333333333333333333333333333333333333333333333333333333333333333333333333333
   #kappa=1.4
   #pv=0 #2337.     # Pa
   ##rho0=998.    # density of liquid in kg/m³
   #rho0=998.20608789369 #<-- is correct for 20°C. This one is correct for 25°C: 997.04802746938
   ##rho0v=0.46638524217467 # density of gas in kg/m³
   ##rho0v=1.2042092351 #0.08737912 # // kg/m^3 of vapour or gas at starting radius!!!!! Has to be adapted for each Rn!!!!!!
   #SpecGasConst=287. #287 for dry air, 462 for H2O vapour. Both in J/molK
   #TempRef=20. # in deg Celsius.
   #T0=1e-2
   #epsilon=1e-2
   def Gilmore(R0_in, v0_in, Requ, \
                t_start, t_end, t_step, \
                T_l = 20.):
    """Run the calculation (Gilmore)
    with the given initial conditions and parameters.
    returns: t, R, R_dot, pg, i
    """

    global p_gas

    # Compute vapour pressure using liquid temperature T_l
    pvapour_in = get_vapour_pressure(T_l)
    print "pv = ", pvapour_in

    # scale initial conditions and parameters
    set_scale(Requ)

    # parameters
    scale_parameters(pvapour_in)
#    print pvapour_in, sc_pvapour

    # initial conditions
    scale_initconds(R0_in, v0_in, Requ, pvapour_in)
#    print scale_R, R0

    # solve system of ODEs
    p_gas = np.zeros(0)
    t_data = create_tdata(t_start, t_end, t_step)

#    print (R0, v0)

    xsol, i = odeint(Gilmore_deriv, (R0, v0), t_data, full_output = True)

    R = xsol[:, 0] * scale_R
    R_dot = xsol[:, 1] * scale_U
    p_gas = np.reshape(p_gas, (-1, 2))
    t = t_data * scale_t

#    np.savetxt('Gilmore_result.dat', (t / 1e-6, R / 1e-6, R_dot))
#    np.savetxt('Gilmore_pg.dat', (p_gas[:, 0], p_gas[:, 1]))

    return (t, R, R_dot, p_gas, i)

    def __init__(self):
        return
    
    def get_parameters(self):
        return
    
    def bar(self):
        print 'Da muss doch was geprintet werden!'
    

def main():
    print 'Subclass:', issubclass(bla, Abc_bubble_models)
    print 'Instance:', isinstance(bla(), Abc_bubble_models)
    c = bla()
    c.get_parameters()
    c.bar()
    
if __name__ == '__main__':
    main()