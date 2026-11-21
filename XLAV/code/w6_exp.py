# BTLN XLAV - tuan 5/6. Requirements: numpy, opencv-python, scikit-image, matplotlib
# Run from XLAV/code/ ; figures are written to XLAV/images/
import numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import skimage.data as D
from w6_lib import *
OUT='../images/'; rng=np.random.default_rng(2026)
cam=D.camera(); g=lambda ax,im,t: (ax.imshow(im,cmap='gray',vmin=0,vmax=255),ax.set_title(t,fontsize=8),ax.axis('off'))

# 1) forward vs backward mapping
f=cam[40:168,190:318]
H,W=f.shape; t=np.deg2rad(30); s=1.3
A=np.array([[s*np.cos(t),-s*np.sin(t),0],[s*np.sin(t),s*np.cos(t),0]],float)
c=np.array([W/2,H/2]); Ho,Wo=int(H*1.6),int(W*1.6); co=np.array([Wo/2,Ho/2])
A[:,2]=co-A[:,:2]@c
fwd=np.zeros((Ho,Wo)); hit=np.zeros((Ho,Wo),bool)
yy,xx=np.mgrid[0:H,0:W]; xp=np.round(A[0,0]*xx+A[0,1]*yy+A[0,2]).astype(int); yp=np.round(A[1,0]*xx+A[1,1]*yy+A[1,2]).astype(int)
ok=(xp>=0)&(xp<Wo)&(yp>=0)&(yp<Ho); fwd[yp[ok],xp[ok]]=f[yy[ok],xx[ok]]; hit[yp[ok],xp[ok]]=True
bwd=warp_affine(f,A,(Ho,Wo))
# nearest backward
M=np.linalg.inv(np.vstack([A,[0,0,1]])); Y,X=np.mgrid[0:Ho,0:Wo]
xs=M[0,0]*X+M[0,1]*Y+M[0,2]; ys=M[1,0]*X+M[1,1]*Y+M[1,2]
inn=(xs>=-0.5)&(xs<W-0.5)&(ys>=-0.5)&(ys<H-0.5)
nn=np.where(inn,f[np.clip(np.round(ys).astype(int),0,H-1),np.clip(np.round(xs).astype(int),0,W-1)],0)
inside_bwd=inn
holes=(~hit)&inside_bwd
print('forward mapping: %.1f%% of target pixels inside the image are holes'%(100*holes.sum()/inside_bwd.sum()))
fig,ax=plt.subplots(1,4,figsize=(10,2.8))
g(ax[0],f,'Ảnh gốc f (128×128)'); rgb=np.stack([fwd]*3,-1).astype(np.uint8); rgb[holes]=[255,40,40]; ax[1].imshow(rgb); ax[1].set_title('Ánh xạ thuận + làm tròn (lỗ: đỏ)',fontsize=8); ax[1].axis('off'); g(ax[2],nn,'Ánh xạ ngược + láng giềng gần'); g(ax[3],bwd,'Ánh xạ ngược + song tuyến')
plt.tight_layout(); plt.savefig(OUT+'w6_forward_backward.png',dpi=120); plt.close()

# 2) augmentation by bilinear warping + mixup
ast=D.astronaut(); cat=D.chelsea()
ast=cv2.resize(ast,(256,256),interpolation=cv2.INTER_AREA); cat=cv2.resize(cat[:,60:360],(256,256),interpolation=cv2.INTER_AREA)
fig,ax=plt.subplots(2,5,figsize=(10,4.3))
ax[0,0].imshow(ast); ax[0,0].set_title('Ảnh gốc',fontsize=8)
for k in range(1,5):
    A=random_affine(256,256,rng); ax[0,k].imshow(np.clip(warp_affine(ast,A),0,255).astype(np.uint8)); ax[0,k].set_title('Biến đổi affine ngẫu nhiên %d'%k,fontsize=8)
for k,lam in enumerate([1.0,0.75,0.5,0.25,0.0]):
    ax[1,k].imshow((lam*ast+(1-lam)*cat).astype(np.uint8)); ax[1,k].set_title(r'mixup $\lambda=%.2f$'%lam,fontsize=8)
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.savefig(OUT+'w6_augment.png',dpi=110); plt.close()

# 3) noise from arbitrary pdf
z=np.linspace(-100,100,2001)
pdfs={'Gauss $\\sigma=20$':np.exp(-z**2/(2*20**2)),
      'Đều $[-40,40]$':(np.abs(z)<=40).astype(float),
      'Laplace $b=12$':np.exp(-np.abs(z)/12),
      'Hai đỉnh (tự định nghĩa)':np.exp(-(z-35)**2/50)+np.exp(-(z+35)**2/50)}
f=cam[100:356,140:396]
fig,ax=plt.subplots(2,5,figsize=(11,4.6)); g(ax[0,0],f,'Ảnh gốc'); ax[1,0].axis('off')
for k,(name,p) in enumerate(pdfs.items(),1):
    n=sample_from_pdf(p,z,200000,rng)
    gimg=add_noise(f,p,z,rng=rng); g(ax[0,k],gimg,name)
    ax[1,k].hist(n,bins=100,density=True,color='0.6'); ax[1,k].plot(z,p/np.trapezoid(p,z),'r',lw=1); ax[1,k].set_yticks([]); ax[1,k].tick_params(labelsize=6)
    ax[1,k].set_title('lược đồ mẫu (xám) và hàm mật độ (đỏ)',fontsize=6)
plt.tight_layout(); plt.savefig(OUT+'w6_noise.png',dpi=110); plt.close()

# 4) smoothing comparison (PSNR)
f=cam.astype(np.uint8)
zg=np.linspace(-80,80,1601); gauss=np.exp(-zg**2/(2*20**2))
noisy={'Gauss ($\\sigma=20$)':add_noise(f,gauss,zg,rng=rng),
       'Muối tiêu (10\\%)':add_noise(f,np.array([1,0,0,1.]),np.array([0,1,254,255.]),mode='impulse',amount=0.10,rng=rng)}
# exact salt & pepper: values 0 or 255
sp=f.copy(); m=rng.random(f.shape); sp[m<0.05]=0; sp[(m>=0.05)&(m<0.10)]=255; noisy['Muối tiêu (10\\%)']=sp
filters={'Trung bình 3×3':lambda x: conv2(x,np.ones((3,3))/9),
         'Gauss 5×5 ($\\sigma=1{,}1$)':lambda x: conv2(x,gaussian_kernel(5,1.1)),
         'Trung vị 3×3':lambda x: median_filter(x,3),
         'TB trọng số gradient':gradient_weighted_mean,
         'Mặt nạ quay':rotating_mask}
res={}
for nk,nv in noisy.items():
    res[nk]={'(ảnh nhiễu)':psnr(f,nv)}
    for fk,fn in filters.items(): res[nk][fk]=psnr(f,np.clip(fn(nv),0,255))
for nk in res: print(nk, {k:round(v,2) for k,v in res[nk].items()})
# edge sharpness: response on a step edge profile
step=np.tile(np.r_[np.full(10,50.),np.full(10,200.)],(15,1))
for fk,fn in filters.items(): print('step',fk,np.round(fn(step)[7,6:14],1))
nv=noisy['Muối tiêu (10\\%)']; crop=(slice(60,250),slice(180,370))
fig,ax=plt.subplots(1,5,figsize=(11,2.6)); g(ax[0],nv[crop],'Nhiễu muối tiêu 10%')
for k,(fk,fn) in enumerate(list(filters.items())[:3],1): g(ax[k],np.clip(fn(nv),0,255)[crop],fk.replace('{,}',',').replace('$\\sigma=','σ=').replace('$',''))
g(ax[4],np.clip(rotating_mask(nv),0,255)[crop],'Mặt nạ quay')
plt.tight_layout(); plt.savefig(OUT+'w6_smoothing.png',dpi=110); plt.close()

# 5) dirt removal with constrained averaging
f=cam.astype(float).copy(); d=f.copy()
img=np.zeros_like(f,np.uint8)
for (p,q) in [((60,80),(200,140)),((300,60),(470,180)),((80,400),(260,330)),((350,300),(480,470))]:
    cv2.line(img,p,q,255,6)
cv2.circle(img,(400,120),40,255,5)
d[img>0]=8                       # dark scribbles, grey level in [0,15]
g1,dirt=constrained_mean(d,0,15,n=7)
for _ in range(3):              # wide strokes: iterate until all dirt replaced
    g1,dirt2=constrained_mean(g1,0,15,n=7)
mean7=conv2(d,np.ones((7,7))/49); med7=median_filter(d,7)
print('dirt PSNR: dirty %.2f mean %.2f median %.2f constrained %.2f'%(psnr(cam,d),psnr(cam,mean7),psnr(cam,med7),psnr(cam,g1)))
fig,ax=plt.subplots(1,4,figsize=(10,2.8)); g(ax[0],d,'Ảnh có vết dơ'); g(ax[1],mean7,'Trung bình 7×7'); g(ax[2],med7,'Trung vị 7×7'); g(ax[3],g1,'TB có ràng buộc [0,15]')
plt.tight_layout(); plt.savefig(OUT+'w6_dirt.png',dpi=110); plt.close()

# 6) edge detection demo
f=cam.astype(float)
e,Eg=gradient_edges(f,30)
L=conv2(f,LAP8); Ez=zero_crossings(L,120)
Llog=conv2(f,log_kernel(9,1.4)); Elog=zero_crossings(Llog,6)
can=cv2.Canny(cam,60,140)
print('edge pixel %:',[round(100*x.mean(),1) for x in (Eg,Ez,Elog,can>0)])
fig,ax=plt.subplots(1,5,figsize=(12,2.7)); g(ax[0],f,'Ảnh gốc')
g(ax[1],np.clip(e*3,0,255),'|∇f| (Sobel)'); g(ax[2],255*Eg,'Ngưỡng |∇f| ≥ 30')
g(ax[3],255*Ez,'Qua 0 của Laplace 8 hướng (T=120)'); g(ax[4],255*Elog,'Qua 0 của LoG (σ=1,4)')
plt.tight_layout(); plt.savefig(OUT+'w6_edges.png',dpi=110); plt.close()

# 7) kernels
np.set_printoptions(precision=4,suppress=True,linewidth=200)
for n in (3,5,7,11):
    s=0.3*((n-1)/2-1)+0.8; h=gaussian_kernel(n,s)
    print('GAUSS',n,'sigma=%.2f'%s); print(h)
    print('int', np.round(h/h.min()).astype(int), 'sum',np.round(h/h.min()).sum())
    r=n//2; v=np.exp(-np.arange(-r,r+1)**2/(2*s*s)); v/=v.sum(); print('1D',np.round(v,4), 'int1D', np.round(v/v.min()).astype(int))
for n,s in ((3,0.5),(5,0.8),(7,1.1)):
    h=log_kernel(n,s); print('LOG',n,s); print(np.round(h,4)); q=np.round(h/np.abs(h).max()*40).astype(int); c=n//2; q[c,c]-=q.sum(); print('scaled40', q, q.sum())
