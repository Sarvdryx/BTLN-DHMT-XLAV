# BTLN XLAV - tuan 5/6. Requirements: numpy, opencv-python, scikit-image, matplotlib
# Run from XLAV/code/ ; figures are written to XLAV/images/
import numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import skimage.data as D
OUT='../images/'
nG=256

def histogram(f):
    H=np.zeros(nG,np.int64)
    np.add.at(H,f.ravel(),1)          # Step 2: H[f(x,y)]++
    return H

def equalize_lut(H, NM):
    T=np.cumsum(H)                    # Step 3: cumulative histogram
    return np.round((nG-1)/NM*T).astype(np.int64)  # Step 4

def specify(f, pg):
    """Histogram specification g = G^{-1}[T[f]] (slides 38-39).
    f: uint8 grey image, pg: target pdf over nG levels (sum=1)."""
    NM=f.size
    T=equalize_lut(histogram(f),NM)                 # Steps 1-4
    Hg=pg*NM                                        # target histogram (counts)
    G=np.round((nG-1)/NM*np.cumsum(Hg)).astype(np.int64)  # Steps 5-6
    # Step 7: G^{-1}: smallest g such that G[g] >= s (G is non-decreasing)
    Ginv=np.searchsorted(G, np.arange(nG), side='left').clip(0,nG-1)
    lut=Ginv[T]
    return lut[f].astype(np.uint8), lut

def apply_color(img, fn):
    """Apply a grey-level mapping on the V channel only, keep H and S."""
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hsv[...,2]=fn(hsv[...,2])
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

# predefined target histogram: Gaussian N(128, 45^2), truncated to [0,255]
z=np.arange(nG); pg=np.exp(-(z-128)**2/(2*45.0**2)); pg/=pg.sum()

names=['camera','moon','coins','clock','cell','page','astronaut','coffee','chelsea','rocket']
imgs=[getattr(D,n)() for n in names]
def V(im): return im if im.ndim==2 else cv2.cvtColor(im,cv2.COLOR_RGB2HSV)[...,2]
rows=[]
for n,im in zip(names,imgs):
    if im.ndim==2:
        he=equalize_lut(histogram(im),im.size)[im].astype(np.uint8)
        hs,_=specify(im,pg)
    else:
        he=apply_color(im,lambda v: equalize_lut(histogram(v),v.size)[v].astype(np.uint8))
        hs=apply_color(im,lambda v: specify(v,pg)[0])
    rows.append((n,im,he,hs))
    print(n, 'mean/std  f=%.1f/%.1f  HE=%.1f/%.1f  HS=%.1f/%.1f'%(V(im).mean(),V(im).std(),V(he).mean(),V(he).std(),V(hs).mean(),V(hs).std()))

def show(ax,im):
    ax.imshow(im,cmap='gray',vmin=0,vmax=255); ax.axis('off')
def hist(ax,im,color,ref=None):
    h=histogram(V(im)).astype(float); h/=h.sum()
    # display with 8-level bins so that sparse (comb-like) histograms stay readable
    hb=h.reshape(32,8).sum(1)/8
    ax.bar(z[::8]+4,hb,width=8,color=color)
    if ref is not None: ax.plot(z,ref,'r-',lw=1)
    ax.set_xlim(0,255); ax.set_ylim(0,max(hb.max(),pg.max())*1.1); ax.set_yticks([]); ax.tick_params(labelsize=5)
for part in range(2):
    fig,axs=plt.subplots(5,6,figsize=(10,8.6))
    for r,(n,im,he,hs) in enumerate(rows[5*part:5*part+5]):
        show(axs[r,0],im); hist(axs[r,1],im,'0.3')
        show(axs[r,2],he); hist(axs[r,3],he,'tab:blue',np.full(nG,1/nG))
        show(axs[r,4],hs); hist(axs[r,5],hs,'tab:green',pg)
        axs[r,0].text(-0.08,0.5,n,fontsize=7,rotation=90,va='center',ha='right',transform=axs[r,0].transAxes)
    for c,t in enumerate(['Ảnh gốc f','Lược đồ của f','Cân bằng (HE)','Lược đồ sau HE','Đặc tả (HS)','Lược đồ sau HS']):
        axs[0,c].set_title(t,fontsize=8)
    plt.tight_layout(); plt.savefig(OUT+'w5_histspec_%d.png'%(part+1),dpi=110); plt.close()
