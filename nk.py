import cmath
import math
import matplotlib
import numpy as np
import scipy.optimize as optimize
from scipy.optimize import minimize
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def r2c(a = 1, b = 0):
  """
  Parameters: 1) a: Real part of complex number (Default value is 1)
              2) b: Imaginary part of complex number (Default value is 0)
  Returns: Complex number
  """
  return complex(a,b)


def transmissionphase_withoutFP(f = 1, d = 1, n1 = 1, n2 = 1, n3 = 1, k1 = 0, k2 = 0, k3 = 0):
    """
    Parameters: 1) f: Frequency is THz (Default value is 1) 
                2) d: Thickness of substrate in mm (Default value is 1) 
                3) n1: Refractive index of medium 1 (Default value is 1) 
                4) n2: Refractive index of sample (Default value is 1) 
                5) n3: Refractive index of medium 3 (Default value is 1)
                6) k1: Absorbtivity of medium 1 (Default value is 0) 
                7) k2: Absorbtivity of sample (Default value is 0) 
                8) k3: Absorbtivity of medium 3 (Default value is 0) 
    Returns: 1) Transmission without FP
             2) Phase without FP
    """
    #Defining Complex Numbers
    n_1 = r2c(n1, -k1)
    n_2 = r2c(n2, -k2)
    n_3 = r2c(n3, -k3)
    n_air = r2c(1, 0)
 
    alpha_constant = 20.958450219516816

    #Defining Amplitude term.
    amp_term = (2*n_2*(n_1+n_3))/((n_2+n_1)*(n_2+n_3)) 

    #Defining Exponential term.
    temp_exp = -r2c(0,1)*(n_2-n_air)*f*d*alpha_constant     
    exp_term  = np.exp(temp_exp.imag*r2c(0,1))*np.exp(temp_exp.real)

    #Returning Transmission and Phase.
    return cmath.polar(amp_term*exp_term)


def transmissionphase_withFP(f = 1, d = 1, n1 = 1, n2 = 1, n3 = 1, k1 = 0, k2 = 0, k3 = 0):
    """
    Parameters: 1) f: Frequency is THz (Default value is 1) 
                2) d: Thickness of substrate in mm (Default value is 1) 
                3) n1: Refractive index of medium 1 (Default value is 1) 
                4) n2: Refractive index of sample (Default value is 1) 
                5) n3: Refractive index of medium 3 (Default value is 1)
                6) k1: Absorbtivity of medium 1 (Default value is 0) 
                7) k2: Absorbtivity of sample (Default value is 0) 
                8) k3: Absorbtivity of medium 3 (Default value is 0) 
    Returns: 1) Transmission with FP
             2) Phase with FP
    """
    #Defining Complex Numbers
    n_1 = r2c(n1, -k1)
    n_2 = r2c(n2, -k2)
    n_3 = r2c(n3, -k3)
    n_air = r2c(1, 0)
 
    alpha_constant = 20.958450219516816

    #Defining Amplitude term.
    amp_term = (2*n_2*(n_1+n_3))/((n_2+n_1)*(n_2+n_3)) 

    #Defining Exponential term.
    temp_exp = -r2c(0,1)*(n_2-n_air)*f*d*alpha_constant     
    exp_term  = np.exp(temp_exp.imag*r2c(0,1))*np.exp(temp_exp.real)
 
    #Defining Fabry-Perot resonance.
    temp_FPexp = -r2c(0,1)*2*n_2*f*d*alpha_constant     
    exp_FPterm  = np.exp(temp_FPexp.imag*r2c(0,1))*np.exp(temp_FPexp.real) 
    FP_term = 1/(1-((n_2-n_1)/(n_2+n_1))*((n_2-n_3)/(n_2+n_3))*(exp_FPterm))

    #Returning Transmission and Phase.
    return cmath.polar(amp_term*exp_term*FP_term)


def loop_transmissionphase_withoutFP(f = np.linspace(1,3,100), d = 1, n1 = 1, n2 = 1, n3 = 1, k1 = 0, k2 = 0, k3 = 0):
    """
    Parameters: 1) f: Frequency array is THz (Default value is 1 to 3 divided into 100 parts) 
                2) d: Thickness of substrate in mm (Default value is 1) 
                3) n1: Refractive index of medium 1 (Default value is 1) 
                4) n2: Refractive index of sample (Default value is 1) 
                5) n3: Refractive index of medium 3 (Default value is 1)
                6) k1: Absorbtivity of medium 1 (Default value is 0) 
                7) k2: Absorbtivity of sample (Default value is 0) 
                8) k3: Absorbtivity of medium 3 (Default value is 0) 
    Returns: 1) Transmission without FP
             2) Phase without FP
    """

    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))

    #Looping over entire frequency range.
    for i in range(len(f)):

      #Calling Function
      transmission[i], phase[i] = transmissionphase_withoutFP(f[i], d, n1, n2, n3, k1, k2, k3)
      
    #Unwrapping phase
    phase = np.unwrap(phase)

    #Returning Transmission and Phase.
    return transmission, phase


def loop_transmissionphase_withFP(f = np.linspace(1,3,100), d = 1, n1 = 1, n2 = 1, n3 = 1, k1 = 0, k2 = 0, k3 = 0):
    """
    Parameters: 1) f: Frequency array is THz (Default value is 1 to 3 divided into 100 parts)  
                2) d: Thickness of substrate in mm (Default value is 1) 
                3) n1: Refractive index of medium 1 (Default value is 1) 
                4) n2: Refractive index of sample (Default value is 1) 
                5) n3: Refractive index of medium 3 (Default value is 1)
                6) k1: Absorbtivity of medium 1 (Default value is 0) 
                7) k2: Absorbtivity of sample (Default value is 0) 
                8) k3: Absorbtivity of medium 3 (Default value is 0) 
    Returns: 1) Transmission with FP
             2) Phase with FP
    """

    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))

    #Looping over entire frequency range.
    for i in range(len(f)):

      #Calling Function
      transmission[i], phase[i] = transmissionphase_withFP(f[i], d, n1, n2, n3, k1, k2, k3)
      
    #Unwrapping phase
    phase = np.unwrap(phase)

    #Returning Transmission and Phase.
    return transmission, phase

def interactive_transmission_without_FP(d, n1, n2, n3, k1, k2, k3):
    
    """
    Parameters: 1) d: Thickness of substrate in mm 
                2) n1: Refractive index of medium 1 
                3) n2: Refractive index of sample 
                4) n3: Refractive index of medium 3
                5) k1: Absorbtivity of medium 1
                6) k2: Absorbtivity of sample
                7) k3: Absorbtivity of medium 3
                
    Returns: 1) Interactive plot for transmission without FP
    
    Use interactive(nk.interactive_transmission_without_FP, d = (0.,1.), n1 = (1.,10.), n2 = (1.,10.), n3 = (1.,10.), k1 = (0.,100.), k2 = (0.,100.), k3 = (0.,100.)) to run the code   
    """   
    f = np.linspace(1,3,100)
    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))
 
    #Looping over entire frequency range.
    for i in range(len(f)):

    #Calling Function
      transmission[i], phase[i] = transmissionphase_withoutFP(f[i], d, n1, n2, n3, k1*0.01, k2*0.01, k3*0.01)

    #Unwrapping phase
    phase = np.unwrap(phase)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Transmission")
    plt.title("Transmission vs Frequency")
    plt.plot(f, transmission)
    plt.show()


def interactive_phase_without_FP(d, n1, n2, n3, k1, k2, k3):
    """
    Parameters: 1) d: Thickness of substrate in mm 
                2) n1: Refractive index of medium 1 
                3) n2: Refractive index of sample 
                4) n3: Refractive index of medium 3
                5) k1: Absorbtivity of medium 1
                6) k2: Absorbtivity of sample
                7) k3: Absorbtivity of medium 3
                
    Returns: 1) Interactive plot for phase without FP
    
    Use interactive(nk.interactive_phase_without_FP, d = (0.,1.), n1 = (1.,10.), n2 = (1.,10.), n3 = (1.,10.), k1 = (0.,100.), k2 = (0.,100.), k3 = (0.,100.)) to run the code   
    """   
    
    f = np.linspace(1,3,100)
    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))

    #Looping over entire frequency range.
    for i in range(len(f)):

      #Calling Function
      transmission[i], phase[i] = transmissionphase_withoutFP(f[i], d, n1, n2, n3, k1*0.01, k2*0.01, k3*0.01)

    #Unwrapping phase
    phase = np.unwrap(phase)
    plt.xlabel("Frequency")
    plt.ylabel("Phase")
    plt.title("Phase vs Frequency")
    plt.plot(f, phase)
    plt.show()




def interactive_transmission_with_FP(d, n1, n2, n3, k1, k2, k3):
    """
    Parameters: 1) d: Thickness of substrate in mm 
                2) n1: Refractive index of medium 1 
                3) n2: Refractive index of sample 
                4) n3: Refractive index of medium 3
                5) k1: Absorbtivity of medium 1
                6) k2: Absorbtivity of sample
                7) k3: Absorbtivity of medium 3
                
    Returns: 1) Interactive plot for transmission with FP
    
    Use interactive(nk.interactive_transmission_with_FP, d = (0.,1.), n1 = (1.,10.), n2 = (1.,10.), n3 = (1.,10.), k1 = (0.,100.), k2 = (0.,100.), k3 = (0.,100.)) to run the code   
    """   
    
    f = np.linspace(1,3,100)
    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))

    #Looping over entire frequency range.
    for i in range(len(f)):

      #Calling Function
      transmission[i], phase[i] = transmissionphase_withFP(f[i], d, n1, n2, n3, k1*0.01, k2*0.01, k3*0.01)

    #Unwrapping phase
    phase = np.unwrap(phase)
    plt.xlabel("Frequency")
    plt.ylabel("Transmission")
    plt.title("Transmission vs Frequency")
    plt.plot(f, transmission)
    plt.show()


def interactive_phase_with_FP(d, n1, n2, n3, k1, k2, k3):
    """
    Parameters: 1) d: Thickness of substrate in mm 
                2) n1: Refractive index of medium 1 
                3) n2: Refractive index of sample 
                4) n3: Refractive index of medium 3
                5) k1: Absorbtivity of medium 1
                6) k2: Absorbtivity of sample
                7) k3: Absorbtivity of medium 3
                
    Returns: 1) Interactive plot for phase with FP
    
    Use interactive(nk.interactive_phase_with_FP, d = (0.,1.), n1 = (1.,10.), n2 = (1.,10.), n3 = (1.,10.), k1 = (0.,100.), k2 = (0.,100.), k3 = (0.,100.)) to run the code   
    """    

    f = np.linspace(1,3,100)
    #Defining transmission and phase array.
    transmission = np.zeros(len(f))
    phase = np.zeros(len(f))

    #Looping over entire frequency range.
    for i in range(len(f)):

      #Calling Function
      transmission[i], phase[i] = transmissionphase_withFP(f[i], d, n1, n2, n3, k1*0.01, k2*0.01, k3*0.01)

    #Unwrapping phase
    phase = np.unwrap(phase)
    plt.xlabel("Frequency")
    plt.ylabel("Phase")
    plt.title("Phase vs Frequency")
    plt.plot(f, phase)
    plt.show()


    
    
    
def shgo_optimizer_withoutFP(trans_temp, phi_temp, f_temp, n1_real, n2, n3_real, k1, k2, k3, d):
    

    alpha_constant = 20.958450219516816
    n_new = []
    k_new = []
    er = []
    fa = []
    trans = []
    phi = []

    #Removing points with consecutive phase difference more than pi/4

    #Looping over entire frequency range
    for i in range(len(phi_temp)-1):

      #Checking if phase difference between 2 consecutive points is less than pi/4
      if (abs(phi_temp[i+1]-phi_temp[i]) < np.pi/4):

        #Appending all the data values satisfying the condition
        fa.append(f_temp[i])
        trans.append(trans_temp[i])
        phi.append(phi_temp[i])

    #Defining Initial Bounds to be considered by Optimizer.
    bounds = ([n2 - 0.05, n2 + 0.05],[k2 - 0.005, k2 + 0.005])
    
    
    for j in range (len(fa)):
        
    
      #Defining Error Function.
        def error(params):
            
            """
            This function is defined to get error term. The function takes Refractive Index and Absorptivity of the sample as an input which we need to determine.
            The functions return error which needs to be converged using optimization algorithms.
            This function will be given as an input to the optimizer so as to calculate Refractive Index and Absorptivity.
            """

            #Defining Refractive Index of Air.
            n_air = 1

            #Defining Refractive Index and Absorptivity as a parameter because our ultimat aim is to find this.
            n2_real, k2 = params

            #Converting Refractive Index in complex forms.

            n1 = complex(1, -0)
            n2 = complex(n2_real, -k2)
            n3 = complex(1, -0)

        #Getting Frequncy, Transmission and Phase from array. 
            f = fa[j]
            transmission = trans[j]
            phase = phi[j]

        #Calling Function as defined in 1 to get Transmission and Phase in terms of parameters. 
            transm, pha = transmissionphase_withoutFP(f, d, n1_real, n2_real, n3_real, k1, k2, k3)

        #Unwrapping phase at every point.
            pha = pha + (((phase + np.pi) // (2 * np.pi))*(2*np.pi))

        #Calulating Error.
            rho = np.log(transmission) - np.log(transm)
            phi1 = phase - pha 
            err = rho**2 + phi1**2
        
        #Returning Error.
            return err
    
        """ 4) Using an optimization algorithm to predict vales of N and k."""

        #Using Optimization function called SHGO to minimize the error
        result = optimize.shgo(error, bounds, iters = 5)

        #Checking whether results are successfully generated. 
        if result.success:

            """If results are successfully generated, we do the following"""

            #Updating the bounds based on previously predicted N and k.
            bounds = ([result.x[0]-0.05,result.x[0]+0.05],[result.x[1]-0.005,result.x[1]+0.005])

            #Appending N, k and error to respective arrays
            fitted_params = result.x
            n_new.append(result.x[0])
            k_new.append(result.x[1])
            er.append(result.funl[0])

        else:

            """If results fail, we do the following"""

            #Print the error
            raise ValueError(result.message)
    return n_new, k_new, er, fa    
    
    
    

    
    
def shgo_optimizer_withFP(trans_temp, phi_temp, f_temp, n1_real, n2, n3_real, k1, k2, k3, d):
    

    alpha_constant = 20.958450219516816
    n_new = []
    k_new = []
    er = []
    fa = []
    trans = []
    phi = []

    #Removing points with consecutive phase difference more than pi/4

    #Looping over entire frequency range
    for i in range(len(phi_temp)-1):

      #Checking if phase difference between 2 consecutive points is less than pi/4
      if (abs(phi_temp[i+1]-phi_temp[i]) < np.pi/4):

        #Appending all the data values satisfying the condition
        fa.append(f_temp[i])
        trans.append(trans_temp[i])
        phi.append(phi_temp[i])

    #Defining Initial Bounds to be considered by Optimizer.
    bounds = ([n2 - 0.05, n2 + 0.05],[k2 - 0.005, k2 + 0.005])
    
    
    for j in range (len(fa)):
        
    
      #Defining Error Function.
        def error(params):
            
            """
            This function is defined to get error term. The function takes Refractive Index and Absorptivity of the sample as an input which we need to determine.
            The functions return error which needs to be converged using optimization algorithms.
            This function will be given as an input to the optimizer so as to calculate Refractive Index and Absorptivity.
            """

            #Defining Refractive Index of Air.
            n_air = 1

            #Defining Refractive Index and Absorptivity as a parameter because our ultimat aim is to find this.
            n2_real, k2 = params

            #Converting Refractive Index in complex forms.

            n1 = complex(1, -0)
            n2 = complex(n2_real, -k2)
            n3 = complex(1, -0)

        #Getting Frequncy, Transmission and Phase from array. 
            f = fa[j]
            transmission = trans[j]
            phase = phi[j]

        #Calling Function as defined in 1 to get Transmission and Phase in terms of parameters. 
            transm, pha = transmissionphase_withFP(f, d, n1_real, n2_real, n3_real, k1, k2, k3)

        #Unwrapping phase at every point.
            pha = pha + (((phase + np.pi) // (2 * np.pi))*(2*np.pi))

        #Calulating Error.
            rho = np.log(transmission) - np.log(transm)
            phi1 = phase - pha 
            err = rho**2 + phi1**2
        
        #Returning Error.
            return err
    
        """ 4) Using an optimization algorithm to predict vales of N and k."""

        #Using Optimization function called SHGO to minimize the error
        result = optimize.shgo(error, bounds, iters = 5)

        #Checking whether results are successfully generated. 
        if result.success:

            """If results are successfully generated, we do the following"""

            #Updating the bounds based on previously predicted N and k.
            bounds = ([result.x[0]-0.05,result.x[0]+0.05],[result.x[1]-0.005,result.x[1]+0.005])

            #Appending N, k and error to respective arrays
            fitted_params = result.x
            n_new.append(result.x[0])
            k_new.append(result.x[1])
            er.append(result.funl[0])

        else:

            """If results fail, we do the following"""

            #Print the error
            raise ValueError(result.message)
    return n_new, k_new, er, fa    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
