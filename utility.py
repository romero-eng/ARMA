import numpy as np


def complexConjugateRootPairZTransformCoefficients(gamma_real, gamma_imag_abs, z_domain_root_within_unit_circle):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    if(gamma_imag_abs < 0):
        raise ValueError('All positive roots in the frequency magnitude must have positive imaginary components')

    eta = 0.5*(np.square(gamma_real) + np.square(gamma_imag_abs) + 1)
    gamma = np.sqrt(eta + np.sqrt(np.square(eta) - np.square(gamma_real)))
    rho_abs = (gamma + np.sqrt(np.square(gamma) - 1))**rho_abs_power

    z_transform_coeff_array = [1, 2*rho_abs*gamma_real/gamma, np.square(rho_abs)]

    if(z_domain_root_within_unit_circle):
        AR_z_transform_coeff_array = z_transform_coeff_array
        MA_z_transform_coeff_array = [1]
    else:
        AR_z_transform_coeff_array = [1]
        MA_z_transform_coeff_array = z_transform_coeff_array

    return [AR_z_transform_coeff_array, MA_z_transform_coeff_array]


def complexConjugateRootPairFreqz(gamma_real, gamma_imag_abs, z_domain_root_within_unit_circle, omega):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    if(gamma_imag_abs < 0):
        raise ValueError('All positive roots in the frequency magnitude must have positive imaginary components')

    eta = 0.5*(np.square(gamma_real) + np.square(gamma_imag_abs) + 1)
    gamma = np.sqrt(eta + np.sqrt(np.square(eta) - np.square(gamma_real)))
    rho_abs = (gamma + np.sqrt(np.square(gamma) - 1))**rho_abs_power

    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    quant_1 = gamma_real/(rho_abs*gamma)
    quant_2 = quant_1/2

    theo_mag = 2*rho_abs*np.sqrt(np.square(cos_omega + gamma_real) + np.square(gamma_imag_abs))
    if(z_domain_root_within_unit_circle):
        theo_mag = 1/theo_mag

    arctan_num = 2*sin_omega*(cos_omega + quant_1)
    arctan_den = 2*np.square(cos_omega + quant_2) - 2*np.square(quant_2) + (1/np.square(rho_abs)) - 1
    theo_phase_deg = -np.rad2deg(np.arctan2(arctan_num, arctan_den))
    if(z_domain_root_within_unit_circle):
        theo_phase_deg = -theo_phase_deg

    squared_freq_mag_coeffs = 4*np.square(rho_abs)*np.array([1, 2*gamma_real, np.square(gamma_real) + np.square(gamma_imag_abs)])

    if(z_domain_root_within_unit_circle):
        AR_squared_freq_mag_coeffs = squared_freq_mag_coeffs
        MA_squared_freq_mag_coeffs = [1]
    else:
        AR_squared_freq_mag_coeffs = [1]
        MA_squared_freq_mag_coeffs = squared_freq_mag_coeffs

    return [theo_mag, theo_phase_deg, AR_squared_freq_mag_coeffs, MA_squared_freq_mag_coeffs]


def realNegativeRootZTransformCoefficents(Gamma, z_domain_root_within_unit_circle):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    rho_abs = (Gamma + np.sqrt(np.square(Gamma) - 1))**rho_abs_power

    z_transform_coeff_array = [1, rho_abs]

    if(z_domain_root_within_unit_circle):
        AR_z_transform_coeff_array = z_transform_coeff_array
        MA_z_transform_coeff_array = [1]
    else:
        AR_z_transform_coeff_array = [1]
        MA_z_transform_coeff_array = z_transform_coeff_array

    return [AR_z_transform_coeff_array, MA_z_transform_coeff_array]


def realNegativeRootFreqz(Gamma, z_domain_root_within_unit_circle, omega):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    rho_abs = (Gamma + np.sqrt(np.square(Gamma) - 1))**rho_abs_power

    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    theo_mag = np.sqrt(2*rho_abs*(Gamma + cos_omega))
    if(z_domain_root_within_unit_circle):
        theo_mag = 1/theo_mag

    arctan_num = sin_omega
    arctan_den = (1/rho_abs) + cos_omega
    theo_phase_deg = -np.rad2deg(np.arctan2(arctan_num, arctan_den))
    if(z_domain_root_within_unit_circle):
        theo_phase_deg = -theo_phase_deg

    squared_freq_mag_coeffs = 2*rho_abs*np.array([1, Gamma])

    return [theo_mag, theo_phase_deg, squared_freq_mag_coeffs]


def realPositiveRootZTransformCoefficents(Gamma, z_domain_root_within_unit_circle):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    rho_abs = (Gamma + np.sqrt(np.square(Gamma) - 1))**rho_abs_power

    z_transform_coeff_array = [1, -rho_abs]

    return z_transform_coeff_array


def realPositiveRootFreqz(Gamma, z_domain_root_within_unit_circle, omega):

    if(z_domain_root_within_unit_circle):
        rho_abs_power = -1
    else:
        rho_abs_power = 1

    rho_abs = (Gamma + np.sqrt(np.square(Gamma) - 1))**rho_abs_power

    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    theo_mag = np.sqrt(-2*rho_abs*(cos_omega - Gamma))
    if(z_domain_root_within_unit_circle):
        theo_mag = 1/theo_mag

    arctan_num = sin_omega
    arctan_den = (1/rho_abs) - cos_omega
    theo_phase_deg = np.rad2deg(np.arctan2(arctan_num, arctan_den))
    if(z_domain_root_within_unit_circle):
        theo_phase_deg = -theo_phase_deg

    squared_freq_mag_coeffs = 2*rho_abs*np.array([-1, Gamma])

    return [theo_mag, theo_phase_deg, squared_freq_mag_coeffs]

