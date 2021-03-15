import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import scipy.optimize as optimize
np.set_printoptions(precision=3)

###Global functions #########################################################################
def H2(x) :                 #Binary entropy Function
    if (x>1/2): return 1
    return - x*np.log2(x) - (1-x)*np.log2(1-x)

def gamma(a,b,c,d):         #statistical uncertainty for phase error
    factor1 = (c+d)*(1-b)*b/c/d/np.log(2)
    factor2 = np.log2( (c+d)/c/d/(1-b)/b*21**2/a**2 )
    return np.sqrt(factor1*factor2)

mysigma = 0.9
def PDTC (eta, LOSS, sigma=mysigma) :
    eta0 = 10**(-LOSS/10)
    return 1/np.sqrt(2*np.pi)/sigma/eta * np.exp( - (np.log(eta/eta0) + sigma*sigma/2)**2 / (2*sigma*sigma) )

def avg_eta(eta_thres,LOSS) :
    upper = 1
    return  integrate.quad( lambda x: x*PDTC(x,LOSS),eta_thres,upper)[0] / integrate.quad(lambda x: PDTC(x,LOSS),eta_thres,upper)[0]

def avg_error(detector_error,eta_thres,LOSS):
    upper = 1
    return  integrate.quad( lambda x: detector_error(x)*PDTC(x,LOSS),eta_thres,upper)[0]/ integrate.quad(lambda x: PDTC(x,LOSS),eta_thres,upper)[0]
#############################################################################################

# Detector Parameters (simplified)
etaBOBH,etaBOBV,etaBOBD,etaBOBA = [0.1]*4
b_H,b_V,b_A,b_D = [2e-4]*4
Y0_H,Y0_V,Y0_A,Y0_D = [4e-5]*4

fEC    = 1.16
mu3    = 2.0e-3
esec   = 1e-10   # kappa*ell
ecor   = 1e-15

# Secure Rate Optimization
def OptimumRate(xVector,loss,sigma=0.9,N=3e10):
   qX   = xVector[0]   # the parameters to optimize over
   Pmu1 = xVector[1]   #
   Pmu2 = xVector[2]   #
   mu1  = xVector[3]   #
   mu2  = xVector[4]   #
   etaTH = xVector[5]  #

   if qX<0 or qX>1 or mu1<0 or Pmu1<0 or Pmu1>1 or mu2<0 or Pmu2<0 or Pmu2>1 or etaTH<0: return 0

   Pmu3 = 1-Pmu1-Pmu2
   qZ = 1-qX
   if mu2 <= mu3 or Pmu3 < 0 : return 0
   muavg = Pmu1*mu1 + Pmu2*mu2 +Pmu3*mu3
   emis = 0.003
   lostGates = 227

   eta0 = 10**(-loss/10)

   sqrt2 = np.sqrt(2)

   I0 =    1/2*( special.erf( (-np.log(eta0)+sigma**2/2)/sqrt2/sigma ) - special.erf( (np.log(etaTH)-np.log(eta0)+sigma**2/2)/sqrt2/sigma) )
   I1 = eta0/2*( special.erf( (-np.log(eta0)-sigma**2/2)/sqrt2/sigma ) - special.erf( (np.log(etaTH)-np.log(eta0)-sigma**2/2)/sqrt2/sigma) )
   eta_ch_avg = I1/I0

   muavg = Pmu1*mu1 + Pmu2*mu2 + Pmu3*mu3
   muavg_H = 0.5*muavg
   muavg_V = 0.5*muavg
   muavg_D = 0.5*muavg*qX + qZ*(1-emis)*muavg
   muavg_A = 0.5*muavg*qX + qZ*   emis *muavg
   # Available detector gates
   gates_H = 1/( 1 + (Y0_H + b_H*eta_ch_avg + eta_ch_avg*etaBOBH*muavg_H)*lostGates )
   gates_V = 1/( 1 + (Y0_V + b_V*eta_ch_avg + eta_ch_avg*etaBOBV*muavg_V)*lostGates )
   gates_D = 1/( 1 + (Y0_D + b_D*eta_ch_avg + eta_ch_avg*etaBOBD*muavg_D)*lostGates )
   gates_A = 1/( 1 + (Y0_A + b_A*eta_ch_avg + eta_ch_avg*etaBOBA*muavg_A)*lostGates )

   nXmu1 = N*Pmu1*qX*0.5*( gates_H*(2*Y0_H*I0 + (2*b_H + mu1*etaBOBH)*I1) + gates_V*(2*Y0_V*I0 + (2*b_V + mu1*etaBOBV)*I1) )
   nXmu2 = N*Pmu2*qX*0.5*( gates_H*(2*Y0_H*I0 + (2*b_H + mu2*etaBOBH)*I1) + gates_V*(2*Y0_V*I0 + (2*b_V + mu2*etaBOBV)*I1) )
   nXmu3 = N*Pmu3*qX*0.5*( gates_H*(2*Y0_H*I0 + (2*b_H + mu3*etaBOBH)*I1) + gates_V*(2*Y0_V*I0 + (2*b_V + mu3*etaBOBV)*I1) )

   nZmu1 = N*Pmu1*qZ*( gates_D*( Y0_D*I0 + (b_D + (1-emis)*mu1*etaBOBD)*I1 ) + gates_A*( Y0_A*I0 + (b_A + emis*mu1*etaBOBA)*I1 ) )
   nZmu2 = N*Pmu2*qZ*( gates_D*( Y0_D*I0 + (b_D + (1-emis)*mu2*etaBOBD)*I1 ) + gates_A*( Y0_A*I0 + (b_A + emis*mu2*etaBOBA)*I1 ) )
   nZmu3 = N*Pmu3*qZ*( gates_D*( Y0_D*I0 + (b_D + (1-emis)*mu3*etaBOBD)*I1 ) + gates_A*( Y0_A*I0 + (b_A + emis*mu3*etaBOBA)*I1 ) )

   mXmu1 = N*Pmu1*0.5*qX*( gates_H*( Y0_H*I0 + (b_H + emis*mu1*etaBOBH)*I1 ) + gates_V*( Y0_V*I0 + (b_V + emis*mu1*etaBOBV)*I1 ) )
   mXmu2 = N*Pmu2*0.5*qX*( gates_H*( Y0_H*I0 + (b_H + emis*mu2*etaBOBH)*I1 ) + gates_V*( Y0_V*I0 + (b_V + emis*mu2*etaBOBV)*I1 ) )
   mXmu3 = N*Pmu3*0.5*qX*( gates_H*( Y0_H*I0 + (b_H + emis*mu3*etaBOBH)*I1 ) + gates_V*( Y0_V*I0 + (b_V + emis*mu3*etaBOBV)*I1 ) )

   mZmu1 = N*Pmu1*qZ*gates_A*( Y0_A*I0 + (emis*etaBOBA*mu1+b_A)*I1 )
   mZmu2 = N*Pmu2*qZ*gates_A*( Y0_A*I0 + (emis*etaBOBA*mu2+b_A)*I1 )
   mZmu3 = N*Pmu3*qZ*gates_A*( Y0_A*I0 + (emis*etaBOBA*mu3+b_A)*I1 )

   nX = nXmu1 + nXmu2 + nXmu3
   mX = mXmu1 + mXmu2 + mXmu3
   nZ = nZmu1 + nZmu2 + nZmu3
   mZ = mZmu1 + mZmu2 + mZmu3

   nXfin = np.sqrt(nX/2*np.log(21/esec))
   mXfin = np.sqrt(mX/2*np.log(21/esec))
   nZfin = np.sqrt(nZ/2*np.log(21/esec))
   mZfin = np.sqrt(mZ/2*np.log(21/esec))

   fin1, fin2, fin3 = np.exp(mu1)/Pmu1, np.exp(mu2)/Pmu2, np.exp(mu3)/Pmu3
   nX_min_Mu1 = fin1*( max(0 ,nXmu1 - nXfin) )
   nX_min_Mu2 = fin2*( max(0 ,nXmu2 - nXfin) )
   nX_min_Mu3 = fin3*( max(0 ,nXmu3 - nXfin) )
   nX_plu_Mu1 = fin1*( min(nX,nXmu1 + nXfin) )
   nX_plu_Mu2 = fin2*( min(nX,nXmu2 + nXfin) )
   nX_plu_Mu3 = fin3*( min(nX,nXmu3 + nXfin) )
   nZ_min_Mu1 = fin1*( max(0 ,nZmu1 - nZfin) )
   nZ_min_Mu2 = fin2*( max(0 ,nZmu2 - nZfin) )
   nZ_min_Mu3 = fin3*( max(0 ,nZmu3 - nZfin) )
   nZ_plu_Mu1 = fin1*( min(nZ,nZmu1 + nZfin) )
   nZ_plu_Mu2 = fin2*( min(nZ,nZmu2 + nZfin) )
   nZ_plu_Mu3 = fin3*( min(nZ,nZmu3 + nZfin) )
   mZ_min_Mu1 = fin1*( max(0 ,mZmu1 - mZfin) )
   mZ_min_Mu2 = fin2*( max(0 ,mZmu2 - mZfin) )
   mZ_min_Mu3 = fin3*( max(0 ,mZmu3 - mZfin) )
   mZ_plu_Mu1 = fin1*( min(mZ,mZmu1 + mZfin) )
   mZ_plu_Mu2 = fin2*( min(mZ,mZmu2 + mZfin) )
   mZ_plu_Mu3 = fin3*( min(mZ,mZmu3 + mZfin) )

   tau0 = np.exp(-mu1)*Pmu1 + np.exp(-mu2)*Pmu2 + np.exp(-mu3)*Pmu3
   tau1 = np.exp(-mu1)*mu1*Pmu1 + np.exp(-mu2)*mu2*Pmu2 + np.exp(-mu3)*mu3*Pmu3

   sX0 = max(0, tau0*(mu2*nX_min_Mu3 - mu3*nX_plu_Mu2)/(mu2-mu3) )  # Vacuum Pulses Contribution
   sZ0 = max(0, tau0*(mu2*nZ_min_Mu3 - mu3*nZ_plu_Mu2)/(mu2-mu3) )
   sX1 = max(0, tau1*mu1*(nX_min_Mu2 - nX_plu_Mu3 - (mu2**2-mu3**2)/(mu1**2)*(nX_plu_Mu1 - sX0/tau0)) / (mu1*(mu2-mu3) - mu2**2 + mu3**2) ) # single photon pulses
   sZ1 = max(0, tau1*mu1*(nZ_min_Mu2 - nZ_plu_Mu3 - (mu2**2-mu3**2)/(mu1**2)*(nZ_plu_Mu1 - sZ0/tau0)) / (mu1*(mu2-mu3) - mu2**2 + mu3**2) )
   uZ1 = max(0, tau1*(mZ_plu_Mu2 - mZ_min_Mu3)/(mu2-mu3) )

   if sZ1 < 1 or sX1 < 1 or uZ1 < 1 or uZ1/sZ1 >= 1: return 0

   ga   = gamma(esec,uZ1/sZ1,sZ1,sX1)
   phiX = uZ1/sZ1 + ga           # Phase Error
   eobs = mX/nX

   privacy = sX1*H2(phiX)   # Privacy Amplification
   lambdaEC = nX*fEC*H2(eobs)   # Error Correction

   distilled = max(0,sX0 + sX1 - privacy - lambdaEC)

   rate = - distilled/N
   return rate

StaticRs = []
OptimiRs = []
x1 = [0.88,0.60,0.24,0.6,0.23,0.02]
losses = np.arange(13,24,0.25)
for lo in losses:
    res = optimize.minimize(OptimumRate, x1 ,lo, method='nelder-mead',options={'xtol': 1e-10 , 'disp':False})   # Optimization Routine
    x1  = res.x     # The intial optimization guess for each length is the result of the previous length.
    xNo = np.array([res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],1e-10])
    StaticRate = - OptimumRate(xNo,lo)
    PRTSthRate = - OptimumRate(res.x,lo)
    StaticRs.append(np.log10(StaticRate))
    OptimiRs.append(np.log10(PRTSthRate))
    cut = res.x[5]
    print("loss: {0:5.2f} Rate: {1:14.11f}  Static: {2:14.11f} ||   qX:{3:6.4f}   pmu1: {4:5.3f}   pmu2: {5:5.3f}   mu1: {6:8.5f}   mu2: {7:8.5f}   etaT:{8:9.6f} ".\
       format(lo,PRTSthRate,StaticRate,res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5]))

plt.plot( losses,StaticRs,'r-',lw=3.0,label='Static')
plt.plot( losses,OptimiRs,'b-',lw=3.0,label='Optimum Cutoff')
plt.title('Key generation improvement in turbulent channels')
plt.xlabel('mean channel loss (dB)')
plt.ylabel('$\log_{{10}}(Rate)$')
plt.legend()
plt.show()
