# Python: 3.11.4
# numpy: 1.25.2
# scipy: 1.11.3
# matplotlib: 3.8.0
import numpy as np
import scipy
import matplotlib.pyplot as plt
class FPDE:
    def __init__(self, alpha, f, r, sigma, phi, fu, fv):
        self.alpha = alpha
        self.f = f
        self.sigma = sigma
        self.phi = phi
        self.fu = fu
        self.fv = fv
        self.r = r
    def time(T): # 生成时间节点
        np.random.seed(2024)
        t = np.array([0])
        while t[-1] < T:
            # 追加1000个0.005-.05均匀分布的时间间隔
            t = np.hstack((t, t[-1]+np.cumsum(np.random.uniform(.005,.05,1000))))
        t = np.delete(t, slice(np.searchsorted(t, T, side='left') + 1, None))
        t[-1] = T
        return t
    def L1Caputo(self, t):
        x0 = self.phi(0)
        # 定义x
        x = np.empty((len(t), len(x0)))
        x[0] = x0
        # 预定义常量
        gamma = scipy.special.gamma(2-self.alpha)
        for n in range(1,len(t)):
            # 生成 a[n,k], 其中 k=1,...,n
            a = np.diff([(t[n]-t[k])**(1-self.alpha) for k in range(0,n+1)]) / np.diff(t[0:n+1])
            # 生成 a[n,k+1]-a[n,k], 其中k = 0,...,n-1
            diffa = np.hstack((a[0], np.diff(a)))
            # 计算 sum_{k=0}^{n-1} (a[n,k+1]-a[n,k])x[k]
            s = x[0:n].transpose().dot(diffa)
            # 预定义
            halpha = (t[n]-t[n-1])**self.alpha
            sigmat = self.sigma(t[n])
            if sigmat < t[n]:
                if sigmat > 0:
                    # 寻找对应时间大于等于sigma(t[n])的最小下标
                    m = np.searchsorted(t, sigmat)
                    sigmax = ((t[m] - sigmat) * x[m-1] + (sigmat - t[m-1]) * x[m]) / (t[m] - t[m-1])
                else:
                    sigmax = self.phi(sigmat)
                f = lambda x: self.f(t[n], x, sigmax)
                fx = lambda x: self.fu(t[n], x, sigmax)
            else:
                f = lambda x: self.f(t[n], x, x)
                fx = lambda x: self.fu(t[n], x, x) + self.fv(t[n], x, x)
            x[n] = scipy.optimize.root(
                fun=lambda x: x - halpha * (gamma * f(x) + s)
                , x0=x[n-1]
                , jac=lambda x: np.eye(len(x0)) - halpha * gamma * fx(x)).x
        return x
    def initt(self):
        return np.linspace(-self.r, 0, np.ceil(50*self.r).astype(np.int64)+1)
    def initx(self, initialt):
        return np.array([self.phi(t) for t in (self.initialt() if initialt is None else initialt)])
A = np.array([[-3,1],[1,-3]])
g = lambda x: x / (1+np.abs(x)**2)
dg = lambda x: (1-np.abs(x)**2) / (1+np.abs(x)**2)**2
f = lambda t,u,v: A.dot(u) + np.array([np.sin(t)+g(v[1]),np.cos(t)+g(v[0])]).transpose()
fu = lambda t,u,v: A
fv = lambda t,u,v: np.array([[0,dg(v[1])],[dg(v[0]),0]])
T = 30

t = FPDE.time(T)
phi = lambda t:np.array([np.sin(t),np.cos(t)])
psi = lambda t:np.array([np.cos(t),np.sin(t)])
fpde = FPDE(alpha=.3, f=f, r=2*np.pi, sigma=lambda t:t-2*np.pi, phi=phi, fu=fu, fv=fv)
initialt = fpde.initt()[:-1]
totalt = np.hstack((initialt, t))
initialx = fpde.initx(initialt)
x = fpde.L1Caputo(t)
plt.figure()
plt.plot(totalt, np.vstack((initialx, x)))
plt.legend(('$x^1$', '$x^2$'))
fpde.phi = psi
initiale = initialx - fpde.initx(initialt)
e = x - fpde.L1Caputo(t)
plt.figure()
plt.plot(totalt, np.vstack((initiale, e)))
plt.legend(('$e^1$', '$e^2$'))

fpde.alpha = .7
fpde.phi = phi
x = fpde.L1Caputo(t)
plt.figure()
plt.plot(totalt, np.vstack((initialx, x)))
plt.legend(('$x^1$', '$x^2$'))
fpde.phi = psi
e = x - fpde.L1Caputo(t)
plt.figure()
plt.plot(totalt, np.vstack((initiale, e)))
plt.legend(('$e^1$', '$e^2$'))

phi = lambda t:np.array([ .8,-.8])
psi = lambda t:np.array([1.2,-.5])
fpde.sigma = lambda t:.5*t
fpde.alpha = .3
fpde.phi = phi
x = fpde.L1Caputo(t)
plt.figure()
plt.plot(t, np.log10(np.abs(x)))
plt.legend(('$\log_{10}x^1$', '$\log_{10}x^2$'))
fpde.phi = psi
e = x - fpde.L1Caputo(t)
plt.figure()
plt.plot(t, np.log10(np.abs(e)))
plt.legend(('$\log_{10}e^1$', '$\log_{10}e^2$'))

fpde.alpha = .7
fpde.phi = phi
x = fpde.L1Caputo(t)
plt.figure()
plt.plot(t, np.log10(np.abs(x)))
plt.legend(('$\log_{10}x^1$', '$\log_{10}x^2$'))
fpde.phi = psi
e = x - fpde.L1Caputo(t)
plt.figure()
plt.plot(t, np.log10(np.abs(e)))
plt.legend(('$\log_{10}e^1$', '$\log_{10}e^2$'))

plt.show()
