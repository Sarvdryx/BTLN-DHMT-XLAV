# BTLN XLAV - tuan 7. Requirements: numpy, scipy, opencv-python, scikit-image, matplotlib
# Run from XLAV/code/ ; figures are written to XLAV/images/
import numpy as np, matplotlib, time
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import skimage.data as D
from scipy.signal import convolve2d, fftconvolve
import os
OUT=os.environ.get('OUT','../images/')
f=D.camera().astype(float); M,N=f.shape
def spec(img):  # centred log-magnitude spectrum for display
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
def show(ax,im,t,cmap='gray',**kw):
    ax.imshow(im,cmap=cmap,**kw); ax.set_title(t,fontsize=8); ax.axis('off')
u_=np.arange(M)-M/2; v_=np.arange(N)-N/2; U,V=np.meshgrid(u_,v_,indexing='ij'); Dist=np.sqrt(U**2+V**2)
def apply_H(img,H):  # H centred, same size as image (periodic boundary)
    Fc=np.fft.fftshift(np.fft.fft2(img)); return np.real(np.fft.ifft2(np.fft.ifftshift(Fc*H)))
def apply_H_pad(img,Hfun):  # with zero padding to 2M x 2N (avoids wrap-around), slide 27 pipeline
    P,Q=2*M,2*N; fp=np.pad(img,((M//2,M//2),(N//2,N//2)),mode='symmetric')  # mirror padding: no dark border
    x=np.arange(P)[:,None]; y=np.arange(Q)[None,:]; fp=fp*(-1.0)**(x+y)        # centre spectrum
    F=np.fft.fft2(fp); uu=np.arange(P)[:,None]-P/2; vv=np.arange(Q)[None,:]-Q/2
    H=Hfun(np.sqrt(uu**2+vv**2)/2)        # distances in units of the unpadded grid
    g=np.real(np.fft.ifft2(F*H))*(-1.0)**(x+y)
    return g[M//2:M//2+M,N//2:N//2+N]
ideal=lambda D0: (lambda d:(d<=D0).astype(float))
butter=lambda D0,n: (lambda d:1/(1+(d/D0)**(2*n)))
gauss=lambda D0: (lambda d:np.exp(-d**2/(2*D0**2)))
P_tot=np.sum(np.abs(np.fft.fft2(f))**2)
def power_kept(D0):
    Fc=np.fft.fftshift(np.fft.fft2(f)); return 100*np.sum(np.abs(Fc[Dist<=D0])**2)/P_tot

# 1) sinusoids and spectra
x=np.arange(100); X,Y=np.meshgrid(x,x,indexing='xy')
sins=[('u=3, v=0',128+127*np.sin(2*np.pi*3*X/99)),('u=6, v=0',128+127*np.sin(2*np.pi*6*X/99)),
      ('u=0, v=4',128+127*np.sin(2*np.pi*4*Y/99)),('u=3, v=5',128+127*np.sin(2*np.pi*(3*X+5*Y)/99))]
fig,ax=plt.subplots(2,4,figsize=(9,4.6))
for k,(t,s) in enumerate(sins):
    show(ax[0,k],s,t,vmin=0,vmax=255)
    S=np.abs(np.fft.fftshift(np.fft.fft2(s-s.mean())))
    c=50; show(ax[1,k],S[c-12:c+13,c-12:c+13],'phổ biên độ (vùng tâm)',cmap='magma')
plt.tight_layout(); plt.savefig(OUT+'w7_sinusoids.png',dpi=110); plt.close()

# 2) periodic noise + notch filter
noise=40*np.sin(2*np.pi*(40*np.arange(M)[:,None]/M+60*np.arange(N)[None,:]/N))
g=f+noise
Fc=np.fft.fftshift(np.fft.fft2(g)); S=np.log1p(np.abs(Fc))
peaks=[(M//2+40,N//2+60),(M//2-40,N//2-60)]
Hn=np.ones((M,N))
for (a,b) in peaks: Hn*=1-np.exp(-((U+M/2-a)**2+(V+N/2-b)**2)/(2*3.0**2))
r=np.real(np.fft.ifft2(np.fft.ifftshift(Fc*Hn)))
def psnr(a,b): return 10*np.log10(255**2/np.mean((a-b)**2))
gm=convolve2d(g,np.ones((5,5))/25,mode='same',boundary='symm')
print('periodic noise PSNR: noisy %.2f  mean5x5 %.2f  notch %.2f'%(psnr(f,np.clip(g,0,255)),psnr(f,np.clip(gm,0,255)),psnr(f,np.clip(r,0,255))))
fig,ax=plt.subplots(1,5,figsize=(12,2.7))
show(ax[0],np.clip(g,0,255),'Ảnh có nhiễu tuần hoàn',vmin=0,vmax=255); show(ax[1],S,'Phổ: 2 đỉnh nhiễu (khoanh đỏ)')
for (a_,b_) in peaks: ax[1].add_patch(plt.Circle((b_,a_),14,fill=False,color='r',lw=1.2))
show(ax[2],np.clip(gm,0,255),'Lọc trung bình 5×5 (không gian)',vmin=0,vmax=255)
show(ax[3],Hn,'Bộ lọc chắn khấc H(u,v)'); [ax[3].add_patch(plt.Circle((b_,a_),14,fill=False,color='r',lw=1.2)) for (a_,b_) in peaks]; show(ax[4],np.clip(r,0,255),'Sau lọc chắn khấc',vmin=0,vmax=255)
plt.tight_layout(); plt.savefig(OUT+'w7_periodic.png',dpi=110); plt.close()

# 3) timings: direct vs FFT convolution
rng=np.random.default_rng(0)
for k in (3,7,15,31,63):
    h=rng.random((k,k)); h/=h.sum()
    t0=time.perf_counter(); a=convolve2d(f,h,mode='same'); t1=time.perf_counter(); b=fftconvolve(f,h,mode='same'); t2=time.perf_counter()
    print('k=%d direct %.3fs fft %.4fs maxdiff %.2e'%(k,t1-t0,t2-t1,np.abs(a-b).max()))
# circular convolution theorem check (unnormalised DFT): ifft(F*H) == circular conv
h=np.zeros((M,N)); h[:3,:3]=1/9
circ=np.real(np.fft.ifft2(np.fft.fft2(f)*np.fft.fft2(h)))
direct=sum(np.roll(np.roll(f,i,0),j,1)/9 for i in range(3) for j in range(3))
print('conv theorem max abs diff',np.abs(circ-direct).max())

# 4) ideal LPF with several D0
D0s=[10,30,60,120]
fig,ax=plt.subplots(1,5,figsize=(12,2.7)); show(ax[0],f,'Ảnh gốc',vmin=0,vmax=255)
for k,D0 in enumerate(D0s,1):
    show(ax[k],apply_H_pad(f,ideal(D0)),'ILPF $D_0=%d$ (giữ %.1f%% năng lượng)'%(D0,power_kept(D0)),vmin=0,vmax=255)
    print('D0',D0,'power %.2f'%power_kept(D0))
plt.tight_layout(); plt.savefig(OUT+'w7_ilpf.png',dpi=110); plt.close()

# 5) H profiles and spatial kernels (1-D cut through the centre)
d=np.linspace(0,150,600); D0=30
fig,ax=plt.subplots(1,2,figsize=(10,3.2))
ax[0].plot(d,ideal(D0)(d),'k',lw=1.2,label='Lý tưởng')
for n,c in ((1,'tab:blue'),(2,'tab:green'),(5,'tab:orange'),(20,'tab:purple')):
    ax[0].plot(d,butter(D0,n)(d),color=c,label='Butterworth n=%d'%n)
ax[0].plot(d,gauss(D0)(d),'r--',label='Gauss ($\\sigma=D_0$)')
ax[0].axvline(D0,color='0.6',ls=':'); ax[0].set_xlabel('D(u,v)'); ax[0].set_ylabel('H'); ax[0].legend(fontsize=7); ax[0].set_title('Hàm truyền H theo khoảng cách, $D_0=30$',fontsize=9)
Lk=512; uu=np.arange(Lk)-Lk/2; UU,VV=np.meshgrid(uu,uu,indexing='ij'); DD=np.sqrt(UU**2+VV**2)
for name,Hf,c in (('Lý tưởng',ideal(D0),'k'),('Butterworth n=2',butter(D0,2),'tab:green'),('Butterworth n=5',butter(D0,5),'tab:orange'),('Gauss',gauss(D0),'r')):
    hk=np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Hf(DD)))))
    prof=hk[Lk//2,Lk//2:Lk//2+40]; ax[1].plot(prof/prof[0],color=c,label=name)
    neg=prof.min()/prof[0]; print('kernel',name,'min/peak %.3f'%neg)
ax[1].axhline(0,color='0.6',lw=0.8); ax[1].set_xlabel('khoảng cách tới tâm (điểm ảnh)'); ax[1].legend(fontsize=7); ax[1].set_title('Nhân không gian h(x) tương ứng (chuẩn hóa đỉnh = 1)',fontsize=9)
plt.tight_layout(); plt.savefig(OUT+'w7_profiles.png',dpi=110); plt.close()

# 6) ILPF vs BLPF vs GLPF at the same D0 (zoom on an edge region to see ringing)
D0=30; crop=(slice(60,260),slice(180,380))
res=[('ILPF',apply_H_pad(f,ideal(D0))),('BLPF n=2',apply_H_pad(f,butter(D0,2))),('BLPF n=5',apply_H_pad(f,butter(D0,5))),('GLPF',apply_H_pad(f,gauss(D0)))]
fig,ax=plt.subplots(1,5,figsize=(12,2.7)); show(ax[0],f[crop],'Ảnh gốc (vùng cắt)',vmin=0,vmax=255)
for k,(t,r) in enumerate(res,1): show(ax[k],r[crop],t+', $D_0=30$',vmin=0,vmax=255); print(t,'psnr vs orig %.2f'%psnr(f,r))
plt.tight_layout(); plt.savefig(OUT+'w7_lp_compare.png',dpi=110); plt.close()

# 7) edge detection result: spatial vs frequency
Hl=-4*np.pi**2*((U/M)**2+(V/N)**2)        # Laplacian in frequency domain (cycles/pixel)
lap=apply_H(f,Hl)
ghp=apply_H_pad(f,lambda d:1-np.exp(-d**2/(2*30**2)))
Fl=np.fft.fftshift(np.fft.fft2(lap)); Ff=np.fft.fftshift(np.fft.fft2(f))
print('DC of laplacian image %.3e, mean %.3e'%(abs(Fl[M//2,N//2]),lap.mean()))
def frac_low(Fc,R=30): return 100*np.sum(np.abs(Fc[Dist<=R])**2)/np.sum(np.abs(Fc)**2)
print('energy within D<=30: original %.2f%%, laplacian %.2f%%, ghpf %.2f%%'%(frac_low(Ff),frac_low(Fl),frac_low(np.fft.fftshift(np.fft.fft2(ghp)))))
fig,ax=plt.subplots(1,4,figsize=(10,2.8))
show(ax[0],spec(f),'Phổ biên độ ảnh gốc (log)')
show(ax[1],np.clip(np.abs(ghp)*3,0,255),'|kết quả GHPF $D_0=30$| (không gian)')
show(ax[2],spec(ghp),'Phổ của kết quả GHPF (log)')
show(ax[3],np.log1p(np.abs(Hl)),'|H| của Laplace trong miền tần số (log)')
plt.tight_layout(); plt.savefig(OUT+'w7_edge_freq.png',dpi=110); plt.close()
