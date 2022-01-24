import numpy as np
import scipy.signal as dsp
import matplotlib as mpl
import matplotlib.pyplot as plt


class chebyshevSpectrumCalculations:

    @staticmethod
    def calculatePartialChebyshevPowerSpectrum(omega, MA_or_AR_root_dicts_list):

        # Temporarily separate the real roots from the complex roots
        real_roots = []
        complex_roots = []
        for root_dict in MA_or_AR_root_dicts_list:
            if(isinstance(root_dict['magnitude-domain root'], complex)):
                complex_roots.append(root_dict['magnitude-domain root'])
            else:
                real_roots.append(root_dict['magnitude-domain root'])

        # Combine all the real roots, complex roots, and complex conjugate roots back 
        # into one big array
        real_roots = np.array(real_roots)
        complex_roots = np.array(complex_roots)
        complex_roots = np.hstack((complex_roots, np.conjugate(complex_roots)))
        roots = np.hstack((real_roots, complex_roots))

        # Get the corresponding cosine polynomial coefficients, and then get the equivalent
        # Chebyshev series coefficients
        cheb_poly_coefs = np.real(np.polynomial.polynomial.polyfromroots(roots))
        cheb_series_coefs = np.polynomial.chebyshev.poly2cheb(cheb_poly_coefs)
    
        # Calculate the squared frequency magnitude response from the Chebyshev series coefficients
        squared_abs_h_f_cheb_theo = np.zeros(omega.shape)
        for n in np.arange(0, len(cheb_series_coefs), 1):
            squared_abs_h_f_cheb_theo = squared_abs_h_f_cheb_theo + cheb_series_coefs[n]*np.cos(n*omega)
    
        # If there is an odd number of positive real roots, then that means that the squared magnitude
        # frequency response needs to be multiplied by -1
        odd_number_of_negatives_flag = np.sum(np.sign(real_roots) == 1) % 2  == 1
        if(odd_number_of_negatives_flag):
            squared_abs_h_f_cheb_theo = -1*squared_abs_h_f_cheb_theo
    
        # Take the square root of the squared frequency magnitude response
        abs_h_f_cheb_theo = np.sqrt(squared_abs_h_f_cheb_theo)
    
        return abs_h_f_cheb_theo

    @staticmethod
    def calculateEntireChebyshevPowerSpectrum(omega, root_dicts_list):  

        # Separate the auto-regressive roots (i.e., the poles) from the
        # moving-average roots (i.e., the zeros)
        AR_root_dicts_list = []
        MA_root_dicts_list = []
        for root_dict in root_dicts_list:
            if(root_dict['moving-average or auto-regressive'] == 'MA'):
                MA_root_dicts_list.append(root_dict)
            elif(root_dict['moving-average or auto-regressive'] == 'AR'):
                AR_root_dicts_list.append(root_dict)
            else:
                raise ValueError('Unexpected value for MA/AR description for the following root: ' + str(root_dict))

        # Calculate the moving-average and auto-regressive frequency spectrums separately, and then
        # divide the moving-average spectrum by the auto-regressive spectrum to get the combined total
        # Chebyshev spectrum
        AR_abs_h_f_cheb_theo = chebyshevSpectrumCalculations.calculatePartialChebyshevPowerSpectrum(omega, AR_root_dicts_list)
        MA_abs_h_f_cheb_theo = chebyshevSpectrumCalculations.calculatePartialChebyshevPowerSpectrum(omega, MA_root_dicts_list)
        abs_h_f_cheb_theo = MA_abs_h_f_cheb_theo/AR_abs_h_f_cheb_theo

        return abs_h_f_cheb_theo

    @staticmethod
    def calculateChebyshevSpectrumPolynomialRoots(f, delta_f, sq_abs_h_f, cutoff = 10**-3, N_max=40):

        # Initialize while-loop variables
        cheb_series = []
        n = 0
        stop_flag = n >= N_max

        # Approximate the Chebyshev series coefficents up to a cutoff or max number of coefficients
        while(not stop_flag):

            # Calculate the current coefficient of chebyshev series
            if(n==0):
                cheb_series_coef = 2*np.sum(sq_abs_h_f)*delta_f
            else:
                cheb_series_coef = 4*np.sum(sq_abs_h_f*np.cos(2*np.pi*n*f))*delta_f

            # If the coefficient is above the cutoff, continue the while-loop
            cheb_series_coef_not_small = np.abs(cheb_series_coef) >= cutoff
            if(cheb_series_coef_not_small):
                cheb_series.append(cheb_series_coef)
                n = n + 1
            
            # Update on whether or not the stop-flag should continue
            stop_flag = (n >= N_max) or (not cheb_series_coef_not_small)
        
        # Convert the list to an array
        cheb_series = np.array(cheb_series)

        # Convert the approximated Chebyshev series into the corresponding Chebyshev Polynomial
        cheb_poly = np.polynomial.chebyshev.cheb2poly(cheb_series)

        # Normalize the coefficients of the chebyshev polynomial
        cheb_poly = cheb_poly/cheb_poly[0]
        
        # Get the roots of the Chebyshev polynomial roots
        cheb_poly_roots = np.polynomial.polynomial.polyroots(cheb_poly)

        return cheb_poly_roots


class fundamentalFrequencyResponseAndZTransformEquations:
    
    @staticmethod
    def get_z_domain_complex_conjugate_root_pair_math_functions():

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

    @staticmethod
    def get_z_domain_single_real_root_math_functions():

        # Basic math equation for mapping a single root from the squared frequency magnitude cosine 
        # polynomial to the magnitudes of the corresponding single real root within the z-transform
        abs_rho_func = lambda gamma, abs_rho_power : (gamma + np.sqrt(np.square(gamma) - 1))**abs_rho_power

        # Array of coefficients for a z-transform with only one specified single real root 
        # in the squared frequency magnitude cosine polynomial
        z_transform_coeffs_func = lambda abs_rho, rho_sign : np.array([1, rho_sign*abs_rho])

        # Frequency magnitude response for the corresponding discrete fourier transform of a z-transform with only 
        # one specified single real root
        magnitude_response_func = lambda omega, gamma, abs_rho, rho_sign : np.sqrt(rho_sign*2*abs_rho*(np.cos(omega) + rho_sign*gamma))

        # Frequency phase response (in degrees) for the corresponding discrete fourier transform of a z-transform with only one specified pair of complex conjugate roots
        phase_degree_response_func = lambda omega, abs_rho, rho_sign : -rho_sign*np.rad2deg(np.arctan2(np.sin(omega), (1/abs_rho) + rho_sign*np.cos(omega)))

        return [abs_rho_func,
                z_transform_coeffs_func,
                magnitude_response_func,
                phase_degree_response_func]


class frequencyResponseAndZTransformCalculations:

    @staticmethod
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
            rho_sign = np.sign(magnitude_domain_addend)
            gamma    =  np.abs(magnitude_domain_addend)

            # If the absolute value of the magnitude-domain addend is less than one, the upcoming equations will not make
            # sense, so raise a ValueError
            if(gamma <= 1):
                raise ValueError('The given single real root must have an absolute value greater than one.')
        
            quant_1 = rho_sign
            quant_2 = gamma

            # Set the boolean flag to say the magnitude-domain root is NOT complex
            complex_flag = False

        return [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2]

    @staticmethod
    def z_trans_coefs(root_dict):
    
        # Figure out how to interpret the given root
        [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2] = frequencyResponseAndZTransformCalculations.interpretRoot(root_dict)
    
        # If the given root is not complex,...
        if(not complex_flag):

            # Reinterpret these generic quantities as describing the frequency magnitude response 
            # of a corresponding z-transform with only one single real root
            rho_sign = quant_1
            gamma    = quant_2

            # Get the appropriate math functions
            [abs_rho_func,
             z_transform_coeffs_func,
             _, _] = \
                fundamentalFrequencyResponseAndZTransformEquations.get_z_domain_single_real_root_math_functions()

            # Calculate the z-transform coefficients based on the math functions which were just retrieved
            z_coefs = z_transform_coeffs_func(abs_rho_func(gamma, abs_rho_power), rho_sign)
    
        # Otherwise, if the given root is complex,...
        else:
        
            # Reinterpret these generic quantities as describing the frequency magnitude response 
            # of a corresponding z-transform with only one complex conjugate root pair
            gamma_real     = quant_1
            abs_gamma_imag = quant_2

            # Get the appropriate math functions
            [eta_func, gamma_func, abs_rho_func, 
             z_transform_coeffs_func, _, _, _, _] = \
                fundamentalFrequencyResponseAndZTransformEquations.get_z_domain_complex_conjugate_root_pair_math_functions()
        
            # Calculate the z-transform coefficients based on the math functions which were just retrieved
            gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)
            z_coefs = z_transform_coeffs_func(gamma_real, gamma, abs_rho_func(gamma, abs_rho_power))

        # If the given root is auto-regressive,...
        if(AR_flag):
            # Put the calculated z-transform coefficients in the AR coefficients,
            # and put the number "1" in the MA coefficients
            MA_z_coefs = np.array([1])
            AR_z_coefs = z_coefs
        # Otherwise, if the given root is moving-average,...
        else:
            # Put the calculated z-transform coefficients in the MA coefficients,
            # and put the number "1" in the MA coefficients
            MA_z_coefs = z_coefs
            AR_z_coefs = np.array([1])

        return [MA_z_coefs, AR_z_coefs]

    @staticmethod
    def freqz(omega, root_dict):
    
        # Figure out how to interpret the given root
        [AR_flag, abs_rho_power, complex_flag, quant_1, quant_2] = frequencyResponseAndZTransformCalculations.interpretRoot(root_dict)

        # If the given root is not complex,...
        if(not complex_flag):

            # Reinterpret these generic quantities as describing the frequency magnitude response 
            # of a corresponding z-transform with only one single real root
            rho_sign = quant_1
            gamma    = quant_2

            # Get the appropriate math functions
            [abs_rho_func, _,
             magnitude_response_func,
             phase_degree_response_func] = \
                fundamentalFrequencyResponseAndZTransformEquations.get_z_domain_single_real_root_math_functions()
        
            # Calculate the quantity which define the frequency response for both magnitude and phase
            abs_rho = abs_rho_func(gamma, abs_rho_power)

            # Calculate the frequency magnitude and phase response (degrees)
            abs_h_f = magnitude_response_func(omega, gamma, abs_rho, rho_sign)
            angle_deg_h_f = phase_degree_response_func(omega, abs_rho, rho_sign)

        # Otherwise, if the given root is complex,...
        else:

            # Reinterpret these generic quantities as describing the frequency magnitude response 
            # of a corresponding z-transform with only one complex conjugate root pair
            gamma_real     = quant_1
            abs_gamma_imag = quant_2

            # Get the appropriate math functions
            [eta_func, gamma_func, abs_rho_func, 
             _, magnitude_response_func, 
             arctan_num_func, arctan_den_func, 
             phase_degree_response_func] = \
                fundamentalFrequencyResponseAndZTransformEquations.get_z_domain_complex_conjugate_root_pair_math_functions()

            # Calculate the quantities which define the frequency response for both magnitude
            # and phase
            gamma = gamma_func(eta_func(gamma_real, abs_gamma_imag), gamma_real)
            abs_rho = abs_rho_func(gamma, abs_rho_power)

            # Calculate the numerator and denominator of the value inside the arctan function
            # which defines the frequency phase response
            arctan_num = arctan_num_func(omega, gamma, gamma_real, abs_rho)
            arctan_den = arctan_den_func(omega, gamma, gamma_real, abs_rho)

            # Calculate the frequency magnitude and phase response (degrees)
            abs_h_f = magnitude_response_func(omega, gamma_real, abs_gamma_imag, abs_rho)
            angle_deg_h_f = phase_degree_response_func(arctan_num, arctan_den)

        # If the given root is auto-regressive,...
        if(AR_flag):
            # Recalculate the magnitude as the reciprocal, and the phase as the negative
            abs_h_f = 1/abs_h_f
            angle_deg_h_f = -angle_deg_h_f

        return [abs_h_f, angle_deg_h_f]


class magnitudeDomainRoots:

    @staticmethod
    def convertLimitedRootsArrayToRootsDictList(withinUnitCircle, MA_or_AR, roots):
        
        all_non_complex_roots_bool_array =     np.squeeze(np.imag(roots) == 0)
        all_complex_roots_bool_array     = np.logical_not(all_non_complex_roots_bool_array)
        real_roots_exist_flag            =         np.any(all_non_complex_roots_bool_array)
        imag_roots_exist_flag            =         np.any(    all_complex_roots_bool_array)

        # Get all the real roots
        if(real_roots_exist_flag):
            real_roots = np.real(roots[all_non_complex_roots_bool_array])
        else:
            real_roots = np.array([])

        # Get all of the complex roots, and then get the roots with only 
        # negative imaginary components
        if(imag_roots_exist_flag):
            complex_roots = roots[np.logical_not(all_non_complex_roots_bool_array)]
            complex_roots = complex_roots[np.imag(complex_roots) < 0]
        else:
            complex_roots = np.array([]).astype(complex)

        # Convert the real roots and remaining complex roots into a list of dictionaries, each
        # of which describes an individual root
        root_dicts_list = []
        for root in real_roots:
            root_dicts_list.append({'z-domain root magnitude within unit circle' : withinUnitCircle, 'moving-average or auto-regressive' : MA_or_AR, 'magnitude-domain root' : root})
        for root in complex_roots:
            root_dicts_list.append({'z-domain root magnitude within unit circle' : withinUnitCircle, 'moving-average or auto-regressive' : MA_or_AR, 'magnitude-domain root' : root})
    
        return root_dicts_list


def generateTheoreticalAndEmpiricalResponses(root_dicts_list):

    # Separately calculate the corresponding z-transform coefficients of the 
    # auto-regressive and moving-average roots, respectively
    MA_z_coefs = np.array([1])
    AR_z_coefs = np.array([1])
    for root_dict in root_dicts_list:
        [tmp_MA_z_coefs, tmp_AR_z_coefs] = frequencyResponseAndZTransformCalculations.z_trans_coefs(root_dict)
        MA_z_coefs = np.polynomial.polynomial.polymul(MA_z_coefs, tmp_MA_z_coefs)
        AR_z_coefs = np.polynomial.polynomial.polymul(AR_z_coefs, tmp_AR_z_coefs)

    # Empirically calculate the frequency magnitude response, the frequency phase
    # response, and the frequency axis itself
    [omega, h_f_emp] = dsp.freqz(MA_z_coefs, AR_z_coefs)
    abs_h_f_emp = np.abs(h_f_emp)
    angle_deg_h_f_emp = np.rad2deg(np.angle(h_f_emp))

    # Calculate the theoretical frequency magnitude and phase responses based on the
    # individual theoretical responses of each of the roots
    abs_h_f_theo = np.ones(abs_h_f_emp.shape)
    angle_deg_h_f_theo = np.zeros(angle_deg_h_f_emp.shape)
    for root_dict in root_dicts_list:
        [tmp_abs_h_f_theo, tmp_angle_deg_h_f_theo] = frequencyResponseAndZTransformCalculations.freqz(omega, root_dict)
        abs_h_f_theo = tmp_abs_h_f_theo*abs_h_f_theo
        angle_deg_h_f_theo = angle_deg_h_f_theo + tmp_angle_deg_h_f_theo
    angle_deg_h_f_theo = np.rad2deg(np.arctan2(np.sin(np.deg2rad(angle_deg_h_f_theo)), np.cos(np.deg2rad(angle_deg_h_f_theo)))) # this is done to wrap the phase response

    # Calculate the theoretical Chebyshev spectrum
    abs_h_f_cheb_theo = chebyshevSpectrumCalculations.calculateEntireChebyshevPowerSpectrum(omega, root_dicts_list)
    
    return [omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo]


def generateSpectralPlots(omega, abs_h_f_emp, angle_deg_h_f_emp, abs_h_f_theo, angle_deg_h_f_theo, abs_h_f_cheb_theo):

    f = omega/(2*np.pi)
    
    usetex = mpl.checkdep_usetex(True)
    if(usetex):
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    abs_h_f_emp_idxs        = np.array([0, 0])
    abs_h_f_theo_idxs       = np.array([1, 0])
    angle_deg_h_f_emp_idxs  = np.array([0, 1])
    angle_deg_h_f_theo_idxs = np.array([1, 1])
    abs_h_f_cheb_theo_idxs  = np.array([2, 0])

    idxs_matrix = np.vstack((abs_h_f_emp_idxs, angle_deg_h_f_emp_idxs, abs_h_f_theo_idxs, angle_deg_h_f_theo_idxs, abs_h_f_cheb_theo_idxs))
    num_rows = np.amax(idxs_matrix[:, 0]) + 1
    num_cols = np.amax(idxs_matrix[:, 1]) + 1
    [fig, axs] = plt.subplots(num_rows, num_cols, figsize=(7, 7))
    if(len(axs.shape) == 2):
        abs_h_f_emp_axis        = axs[       abs_h_f_emp_idxs[0],        abs_h_f_emp_idxs[1]]
        angle_deg_h_f_emp_axis  = axs[ angle_deg_h_f_emp_idxs[0],  angle_deg_h_f_emp_idxs[1]]
        abs_h_f_theo_axis       = axs[      abs_h_f_theo_idxs[0],       abs_h_f_theo_idxs[1]]
        angle_deg_h_f_theo_axis = axs[angle_deg_h_f_theo_idxs[0], angle_deg_h_f_theo_idxs[1]]
        abs_h_f_cheb_theo_axis  = axs[ abs_h_f_cheb_theo_idxs[0],  abs_h_f_cheb_theo_idxs[1]]
    else:
        abs_h_f_emp_axis        = axs[np.amax(       abs_h_f_emp_idxs)]
        angle_deg_h_f_emp_axis  = axs[np.amax( angle_deg_h_f_emp_idxs)]
        abs_h_f_theo_axis       = axs[np.amax(      abs_h_f_theo_idxs)]
        angle_deg_h_f_theo_axis = axs[np.amax(angle_deg_h_f_theo_idxs)]
        abs_h_f_cheb_theo_axis  = axs[np.amax( abs_h_f_cheb_theo_idxs)]

    abs_h_f_emp_axis.plot(f, abs_h_f_emp)
    abs_h_f_emp_axis.set_xlim([0, 0.5])
    abs_h_f_emp_axis.grid()
    abs_h_f_emp_axis.set_xlabel('Frequency (Hz)')
    if(usetex):
        abs_h_f_emp_axis.set_ylabel(r'$\big|h(f)\big|$')
    else:
        abs_h_f_emp_axis.set_ylabel(r'$|h(f)|$')
    abs_h_f_emp_axis.set_title('Empirical Magnitude')
    
    angle_deg_h_f_emp_axis.plot(f, angle_deg_h_f_emp)
    angle_deg_h_f_emp_axis.set_xlim([0, 0.5])
    angle_deg_h_f_emp_axis.grid()
    angle_deg_h_f_emp_axis.set_xlabel('Frequency (Hz)')
    angle_deg_h_f_emp_axis.set_ylabel(r'${\angle}h(f)^{\circ}$')
    angle_deg_h_f_emp_axis.set_title('Empirical Phase')

    abs_h_f_theo_axis.plot(f, abs_h_f_theo)
    abs_h_f_theo_axis.set_xlim([0, 0.5])
    abs_h_f_theo_axis.grid()
    abs_h_f_theo_axis.set_xlabel('Frequency (Hz)')
    if(usetex):
        abs_h_f_theo_axis.set_ylabel(r'$\big|h(f)\big|$')
    else:
        abs_h_f_theo_axis.set_ylabel(r'$|h(f)|$')
    abs_h_f_theo_axis.set_title('Theoretical Magnitude')
    
    angle_deg_h_f_theo_axis.plot(f, angle_deg_h_f_theo)
    angle_deg_h_f_theo_axis.set_xlim([0, 0.5])
    angle_deg_h_f_theo_axis.grid()
    angle_deg_h_f_theo_axis.set_xlabel('Frequency (Hz)')
    if(usetex):
        angle_deg_h_f_theo_axis.set_ylabel(r'${\angle}h(f)^{\circ}$')
    else:
        angle_deg_h_f_theo_axis.set_ylabel(r'${\angle}h(f)^\circ$')
    angle_deg_h_f_theo_axis.set_title('Theoretical Phase')

    abs_h_f_cheb_theo_axis.plot(f, abs_h_f_cheb_theo)
    abs_h_f_cheb_theo_axis.set_xlim([0, 0.5])
    abs_h_f_cheb_theo_axis.grid()
    abs_h_f_cheb_theo_axis.set_xlabel('Frequency (Hz)')
    if(usetex):
        abs_h_f_cheb_theo_axis.set_ylabel(r'$\big|h(f)\big|$')
    else:
        abs_h_f_cheb_theo_axis.set_ylabel(r'$|h(f)|$')
    abs_h_f_cheb_theo_axis.set_title('Chebyshev Magnitude')

    fig.tight_layout()

    plt.show()


if(__name__=='__main__'):

    root_dicts_list = \
        magnitudeDomainRoots.convertLimitedRootsArrayToRootsDictList(False, 'MA', np.array([-1.1, 1.1 - 1.9j, 1.0 + 1.9j])) + \
        magnitudeDomainRoots.convertLimitedRootsArrayToRootsDictList( True, 'MA', np.array([ 1.4, 4.5 - 0.4j, 4.5 + 0.4j])) + \
        magnitudeDomainRoots.convertLimitedRootsArrayToRootsDictList( True, 'AR', np.array([ 3.1, 0.4 - 9.2j, 0.4 + 9.2j]))

    #root_dicts_list = convertLimitedRootsArrayToRootsDictList(False, 'MA', np.array([-1.1, 1.1 - 1.9j, 1.4, 4.5 - 0.4j]))

    [omega, 
     abs_h_f_emp, angle_deg_h_f_emp, 
     abs_h_f_theo, angle_deg_h_f_theo, 
     abs_h_f_cheb_theo] = \
        generateTheoreticalAndEmpiricalResponses(root_dicts_list)

    generateSpectralPlots(omega,
                          abs_h_f_emp/np.amax(abs_h_f_emp), 
                          angle_deg_h_f_emp, 
                          abs_h_f_theo/np.amax(abs_h_f_theo), 
                          angle_deg_h_f_theo, 
                          abs_h_f_cheb_theo/np.amax(abs_h_f_cheb_theo))
