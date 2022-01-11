import numpy as np
import scipy.signal as dsp
import matplotlib
import matplotlib.pyplot as plt


class zDomainComplexConjugateRootPair:

    @staticmethod
    def get_math_functions():

        # Basic math equations for mapping a complex conjugate root pair from the squared frequency magnitude cosine 
        # polynomial to the magnitudes of the corresponding complex conjugate root pair within the z-transform
        eta_func     = lambda gamma_real, abs_gamma_imag : 0.5*(np.square(gamma_real) + np.square(abs_gamma_imag) + 1)
        gamma_func   = lambda eta, gamma_real : np.sqrt(eta + np.sqrt(np.square(eta) - np.square(gamma_real)))
        abs_rho_func = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power

        # Array of coefficients for a z-transform with only one specified pair of complex conjugate roots in the squared frequency
        # magnitude cosine polynomial
        z_transform_coeffs_func = lambda gamma_real, gamma, abs_rho : np.array([1, 2*abs_rho*(gamma_real/gamma), np.square(abs_rho)])

        # Frequency magnitude response for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
        magnitude_response_func = lambda omega, gamma_real, abs_gamma_imag, abs_rho : 2*abs_rho*np.sqrt(np.square(np.cos(omega) + gamma_real) + np.square(abs_gamma_imag))

        # Frequency phase response (in degrees) for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
        arctan_num_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.sin(omega)*((gamma_real/(abs_rho*gamma)) + np.cos(omega))
        arctan_den_func            = lambda omega, gamma, gamma_real, abs_rho : 2*np.square(np.cos(omega) + (gamma_real/(2*abs_rho*gamma))) - 2*np.square(gamma_real/(2*abs_rho*gamma)) + np.square(1/abs_rho) - 1
        phase_degree_response_func = lambda arctan_num, arctan_den : -np.rad2deg(np.arctan2(arctan_num, arctan_den))

        return [eta_func, gamma_func, abs_rho_func, 
                z_transform_coeffs_func, 
                magnitude_response_func, 
                arctan_num_func, arctan_den_func, 
                phase_degree_response_func]


class zDomainSingleRealRoot:

    @staticmethod
    def get_math_functions():

        # Basic math equation for mapping a single root from the squared frequency magnitude cosine 
        # polynomial to the magnitudes of the corresponding single real root within the z-transform
        abs_rho_func = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power

        # Array of coefficients for a z-transform with only one specified single real root 
        # in the squared frequency magnitude cosine polynomial
        z_transform_coeffs_func = lambda abs_rho, rho_sign : np.array([1, rho_sign*abs_rho])

        # Frequency magnitude response for the corresponding discrete fourier transform of a z-transform with only 
        # one specified single real root
        magnitude_response_func = lambda omega, gamma, abs_rho, rho_sign : np.sqrt(2*abs_rho*(gamma + rho_sign*np.cos(omega)))

        # Frequency phase response (in degrees) for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
        phase_degree_response_func = lambda omega, abs_rho, rho_sign : -rho_sign*np.rad2deg(np.arctan2(np.sin(omega), (1/abs_rho) + rho_sign*np.cos(omega)))

        return [abs_rho_func,
                z_transform_coeffs_func,
                magnitude_response_func,
                phase_degree_response_func]


def interpretRoot(root_dict):

    # Obtain the information about whether or not the z-domain root lies inside of the unit circle, 
    # and then initialize the corresponding exponent accordingly
    z_domain_root_within_unit_circle = root_dict['z-domain root magnitude within unit circle']
    abs_rho_power = 1 - 2*z_domain_root_within_unit_circle

    # Obtain the information about whether the z-domain root is moving-average (FIR) or
    # auto-regressive (IIR)
    MA_flag = root_dict['moving-average or auto-regressive'] == 'MA'
    AR_flag = root_dict['moving-average or auto-regressive'] == 'AR'

    # If the the root is not either moving-average or auto-regressive, then an error
    # occurred somewhere
    if((not MA_flag) and (not AR_flag)):
        raise ValueError("The 'MA_or_AR' variable must either be 'MA' or 'AR'")
    
    # If an autoregressive z-domain root does not lie within the unit circle, then the resulting difference equation coefficients will lead to
    # an unstable time-domain response which cannot be allowed
    if(AR_flag and (not z_domain_root_within_unit_circle)):
        raise ValueError("Cannot have autoregressive (IIR) coefficients while the magnitude of the roots of the corresponding z-transform are outside the unit circle")

    # Get the root from the magnitude-domain (i.e., the plane where the roots of the squared magnitude frequency cosine polynomial lie),
    # and then convert it to the corresponding addend (which is more compatible with the upcoming equations) 
    magnitude_domain_root = root_dict['magnitude-domain root']
    magnitude_domain_addend = -1*magnitude_domain_root

    # If the magnitude-domain addend is complex,...
    if(isinstance(magnitude_domain_addend, complex)):

        # Get the real and imaginary components
        gamma_real     = np.real(magnitude_domain_addend)
        abs_gamma_imag = np.imag(magnitude_domain_addend)

        # If the imaginary component is negative, the upcoming equations will not make sense, so raise a ValueError
        if(abs_gamma_imag <= 0):
            raise ValueError('The given complex root of a squared frequency magnitude cosine polynomial must have a negative imaginary component.')

        quant_1 = gamma_real
        quant_2 = abs_gamma_imag

        # Set the boolean flag to say the magnitude-domain root is complex
        complex_flag = True
        
    else:

        # Get the sign and absolute value of the magnitude-domain addend
        rho_sign = -1*np.sign(magnitude_domain_addend)
        gamma     =    np.abs(magnitude_domain_addend)

        # If the absolute value of the magnitude-domain addend is less than one, the upcoming equations will not make
        # sense, so raise a ValueError
        if(gamma <= 1):
            raise ValueError('The given single real root must have an absolute value greater than one.')
        
        quant_1 = rho_sign
        quant_2 = gamma

        # Set the boolean flag to say the magnitude-domain root is NOT complex
        complex_flag = False

    return [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2]


def z_trans_coefs(root_dict):
    
    [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2] = interpretRoot(root_dict)
    
    if(not complex_flag):

        rho_sign = quant_1
        gamma    = quant_2

        [abs_rho_func,
         z_transform_coeffs_func,
         _, _] = \
            zDomainSingleRealRoot.get_math_functions()

        z_coefs = z_transform_coeffs_func(abs_rho_func(gamma, abs_rho_power), rho_sign)
    
    else:

        gamma_real     = quant_1
        abs_gamma_imag = quant_2

        [eta_func, gamma_func, abs_rho_func, 
         z_transform_coeffs_func, _, _, _, _] = \
            zDomainComplexConjugateRootPair.get_math_functions()
        
        gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)

        z_coefs = z_transform_coeffs_func(gamma_real, gamma, abs_rho_func(gamma, abs_rho_power))

    if(AR_flag):
        MA_z_coefs = np.array([1])
        AR_z_coefs = z_coefs
    else:
        MA_z_coefs = z_coefs
        AR_z_coefs = np.array([1])

    return [MA_z_coefs, AR_z_coefs]


def freqz(omega, root_dict):
    
    [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2] = interpretRoot(root_dict)

    if(not complex_flag):

        rho_sign = quant_1
        gamma    = quant_2

        [abs_rho_func, _,
         magnitude_response_func,
         phase_degree_response_func] = \
            zDomainSingleRealRoot.get_math_functions()

        abs_rho = abs_rho_func(gamma, abs_rho_power)

        abs_h_f = magnitude_response_func(omega, gamma, abs_rho, rho_sign)
        angle_deg_h_f = phase_degree_response_func(omega, abs_rho, rho_sign)

    else:

        [eta_func, gamma_func, abs_rho_func, 
         _, magnitude_response_func, 
         arctan_num_func, arctan_den_func, 
         phase_degree_response_func] = \
            zDomainComplexConjugateRootPair.get_math_functions()

        gamma_real     = quant_1
        abs_gamma_imag = quant_2

        gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)
        abs_rho = abs_rho_func(gamma, abs_rho_power)

        arctan_num = arctan_num_func(omega, gamma, gamma_real, abs_rho)
        arctan_den = arctan_den_func(omega, gamma, gamma_real, abs_rho)

        abs_h_f = magnitude_response_func(omega, gamma_real, abs_gamma_imag, abs_rho)
        angle_deg_h_f = phase_degree_response_func(arctan_num, arctan_den)

    if(AR_flag):
        abs_h_f = 1/abs_h_f
        angle_deg_h_f = -angle_deg_h_f

    return [abs_h_f, angle_deg_h_f]


def generateSpectralPlots(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo):

    usetex = matplotlib.checkdep_usetex(True)

    f = omega/(2*np.pi)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    [fig1, axs] = plt.subplots(2, 2)

    axs[0, 0].plot(f, abs_h_f_emp)
    axs[0, 0].set_xlim([0, 0.5])
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].grid()
    axs[0, 0].set_xlabel('Frequency (Hz)')
    if(usetex):
        axs[0, 0].set_ylabel(r'$\big|h(f)\big|$')
    axs[0, 0].set_title('Empirical Magnitude')
    
    axs[1, 0].plot(f, angle_deg_h_f_emp)
    axs[1, 0].set_xlim([0, 0.5])
    axs[1, 0].grid()
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel(r'${\angle}h(f)^{\circ}$')
    axs[1, 0].set_title('Empirical Phase')

    axs[0, 1].plot(f, abs_h_f_theo)
    axs[0, 1].set_xlim([0, 0.5])
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].grid()
    axs[0, 1].set_xlabel('Frequency (Hz)')
    if(usetex):
        axs[0, 1].set_ylabel(r'$\big|h(f)\big|$')
    axs[0, 1].set_title('Theoretical Magnitude')
    
    axs[1, 1].plot(f, angle_deg_h_f_theo)
    axs[1, 1].set_xlim([0, 0.5])
    axs[1, 1].grid()
    axs[1, 1].set_xlabel('Frequency (Hz)')
    if(usetex):
        axs[1, 1].set_ylabel(r'${\angle}h(f)^{\circ}$')
    axs[1, 1].set_title('Theoretical Phase')

    fig1.tight_layout()

    plt.show()


if(__name__=='__main__'):

    root_dicts_list = \
        [{'z-domain root magnitude within unit circle' :  False, 'moving-average or auto-regressive' : 'MA', 'magnitude-domain root' : 1.1 - 1.9j}]

    root_dict = root_dicts_list[0]

    [MA_z_coefs, AR_z_coefs] = z_trans_coefs(root_dict)

    [omega, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs)
    abs_h_f_emp = np.abs(h_f_emp)
    angle_deg_h_f_emp = np.rad2deg(np.angle(h_f_emp))

    [abs_h_f_theo, angle_deg_h_f_theo] = freqz(omega, root_dict)

    generateSpectralPlots(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo)
