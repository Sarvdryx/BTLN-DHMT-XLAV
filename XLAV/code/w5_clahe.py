# BTLN XLAV - tuan 5/6. Requirements: numpy, opencv-python, scikit-image, matplotlib
# Run from XLAV/code/ ; figures are written to XLAV/images/
import numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import skimage.data as D
OUT='../images/'
ims={'moon':D.moon(),'cell':D.cell(),'retina (kênh L)':cv2.cvtColor(cv2.resize(D.retina(),(512,512)),cv2.COLOR_RGB2LAB)[...,0]}
cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
fig,ax=plt.subplots(3,3,figsize=(7.5,7.5))
for r,(n,f) in enumerate(ims.items()):
    for c,(t,g) in enumerate([('Ảnh gốc',f),('Cân bằng toàn cục (HE)',cv2.equalizeHist(f)),('CLAHE (clip 2,0; lưới 8×8)',cl.apply(f))]):
        ax[r,c].imshow(g,cmap='gray',vmin=0,vmax=255); ax[r,c].axis('off')
        if r==0: ax[r,c].set_title(t,fontsize=8)
    ax[r,0].text(-0.05,0.5,n,rotation=90,fontsize=8,va='center',ha='right',transform=ax[r,0].transAxes)
    print(n,[round(float(x.std()),1) for x in (f,cv2.equalizeHist(f),cl.apply(f))])
plt.tight_layout(); plt.savefig(OUT+'w5_clahe.png',dpi=110)
