
import numpy as np
import scipy.stats as stats
from scipy.stats import truncexpon
from scipy.stats import truncnorm
from iminuit import Minuit
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14

# define pdf and generate data
np.random.seed(seed=1234567)        # fix random seed
theta = 0.2                         # fraction of signal
mu = 10.                            # mean of Gaussian
sigma = 2.                          # std. dev. of Gaussian
xi = 5.                             # mean of exponential
xMin = 0.
xMax = 20.

def f(x, par):
    theta   = par[0]
    mu      = par[1]
    sigma   = par[2]
    xi      = par[3]
    fs = stats.truncnorm.pdf(x, a=(xMin-mu)/sigma, b=(xMax-mu)/sigma, loc=mu, scale=sigma)
    fb = stats.truncexpon.pdf(x, b=(xMax-xMin)/xi, loc=xMin, scale=xi)
    return theta*fs + (1-theta)*fb
        
numVal = 200
xData = np.empty([numVal])
for i in range (numVal):
    r = np.random.uniform();
    if r < theta:
        xData[i] = stats.truncnorm.rvs(a=(xMin-mu)/sigma, b=(xMax-mu)/sigma, loc=mu, scale=sigma)
    else:
        xData[i] = stats.truncexpon.rvs(b=(xMax-xMin)/xi, loc=xMin, scale=xi)

# Function to be minimized is negative log-likelihood
def negLogL(par):
    pdf = f(xData, par)
    return -np.sum(np.log(pdf))

# Initialize Minuit and set up fit:
#     initial parameter values are guesses,
#     error values set initial step size in search algorithm,
#     limit_param to set limits on parameters (needed here to keep pdf>0),
#     fix_param=True to fix a parameter,
#     errordef=0.5 means errors correspond to logL = logLmax - 0.5,
#     pedantic=False to turn off verbose messages.

par     = np.array([theta, mu, sigma, xi])      # initial values (here equal true values)
parname = ['theta', 'mu', 'sigma', 'xi']
parstep = np.array([0.1, 1., 1., 1.])           # initial setp sizes
parfix  = [False, True, True, False]            # change these to fix/free parameters
parlim  = [(0.,1), (None, None), (0., None), (0., None)]
m = Minuit.from_array_func(negLogL, par, parstep, name=parname,
    limit=parlim, fix=parfix, errordef=0.5, pedantic=False)

# do the fit, get errors, extract results
m.migrad()                                             # minimize -logL
MLE = m.np_values()                                    # max-likelihood estimates
sigmaMLE = m.np_errors()                               # standard deviations
cov = m.np_matrix(skip_fixed=True)                     # covariance matrix
rho = m.np_matrix(skip_fixed=True, correlation=True)   # correlation coeffs.
npar = len(m.np_values())
nfreepar = len(cov[0])
    
npar = len(m.np_values())
print(r'par index, name, estimate, standard deviation:')
for i in range(npar):
    if not(m.is_fixed(i)):
        print("{:4d}".format(i), "{:<10s}".format(m.parameters[i]), " = ",
         "{:.6f}".format(MLE[i]), " +/- ", "{:.6f}".format(sigmaMLE[i]))

print()
print(r'free par indices, covariance, correlation coeff.:')
for i in range(nfreepar):
    for j in range(nfreepar):
        print(i, j, "{:.6f}".format(cov[i,j]), "{:.6f}".format(rho[i,j]))
    
# Plot fitted pdf
yMin = 0.
yMax = f(0., MLE)*1.1
fig = plt.figure(figsize=(8,6))
xCurve = np.linspace(xMin, xMax, 100)
yCurve = f(xCurve, MLE)
plt.plot(xCurve, yCurve, color='dodgerblue')

# Plot data as tick marks
tick_height = 0.05*(yMax - yMin)
xvals = [xData, xData]
yvals = [np.zeros_like(xData), tick_height * np.ones_like(xData)]
plt.plot(xvals, yvals, color='black', linewidth=1)
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x; \theta)$')
plt.figtext(0.6, 0.8, r'$\hat{\theta} = $' + f'{MLE[0]:.4f}' +
            r'$\pm$' + f'{sigmaMLE[0]:.4f}')
plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)
plt.show()

# Make scan of lnL (for theta, if free)
if not(m.is_fixed('theta')):
    plt.figure()
    m.draw_mnprofile('theta')

# Make a contour plot of lnL = lnLmax - 1/2 (here for theta and xi)
if not(m.is_fixed('theta') | m.is_fixed('xi')):
    plt.figure()
    m.draw_mncontour('theta', 'xi', nsigma=1, numpoints=200);
    plt.show()