# interpretation of the base of the FB with an SNR or Cygnus-like region
# python scripts/dima/SN_cygnus_interp.py

import numpy as np

GeV2MeV = 1000.
pc2cm = 1.e18
km2cm = 1.e5
y2s = 3.e7
Omega0 = np.deg2rad(10.) * np.deg2rad(12.)

def F2I(F, Omega=Omega0, Rratio=1.):
    return F / Omega / GeV2MeV / Rratio**2

Fcyg = 1.6e-5 # MeV cm^-2 s^-1
Rcyg = 1. / 3.
Omega_cyg = np.pi * np.deg2rad(6.)**2

Icyg = F2I(Fcyg, Omega=Omega_cyg, Rratio=Rcyg)

print 'Cygnus intensity: %.2g' % Icyg


dist = 1000 # pc
H0 = dist * np.tan(np.deg2rad(6.))

Tcool = 1.e6 * y2s # s
v = 700 * km2cm # cm/s
Lcool = v * Tcool / pc2cm

Tesc_FB = dist * pc2cm / v / y2s
Tesc_base = H0 * pc2cm / v / y2s

print 'Lcool = %.3g pc' % (Lcool)
print 'Escape time from the FB Tesc = %.3g ky' % (Tesc_FB/1000.)
print 'Escape time from the FB Tesc = %.3g ky' % (Tesc_base/1000.)