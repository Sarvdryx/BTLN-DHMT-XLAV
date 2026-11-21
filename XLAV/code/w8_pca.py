# BTLN XLAV - tuan 8. Requirements: numpy, scikit-image, matplotlib (scikit-learn optional)
# Run from XLAV/code/ ; figures are written to XLAV/images/
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.environ.get('OUT', '../images/')

# ---------------------------------------------------------------- PCA core
def load_faces(folder, size=None):
    """Read all grey-level images in a folder -> array (m, r, c)."""
    import glob, cv2
    imgs = []
    for p in sorted(glob.glob(folder + '/*')):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if size is not None: im = cv2.resize(im, size)   # size = (c, r)
        imgs.append(im)
    return np.stack(imgs).astype(np.float64)

def pca_fit(images, k=None, var_ratio=0.95):
    """images: array (m, r, c). Returns mean (n,), basis U (n, k), eigenvalues (k,), all eigenvalues."""
    m = images.shape[0]
    X = images.reshape(m, -1).astype(np.float64).T          # n x m, one image per column
    mu = X.mean(axis=1, keepdims=True)                       # mean face
    A = X - mu                                               # centred data
    n = A.shape[0]
    if m < n:                                                # Turk-Pentland trick: eig of the small m x m matrix
        L = A.T @ A / m
        mu_s, V = np.linalg.eigh(L)                          # ascending order
        idx = np.argsort(mu_s)[::-1]; mu_s, V = mu_s[idx], V[:, idx]
        keep = mu_s > 1e-10 * mu_s[0]                        # at most m-1 non-zero eigenvalues
        mu_s, V = mu_s[keep], V[:, keep]
        U = A @ V                                            # C (A v) = mu (A v)
        U /= np.linalg.norm(U, axis=0)                       # normalise: a_i^T a_i = 1
        lam = mu_s
    else:
        C = A @ A.T / m
        lam, U = np.linalg.eigh(C); idx = np.argsort(lam)[::-1]; lam, U = lam[idx], U[:, idx]
    if k is None:                                            # smallest k keeping var_ratio of the variance
        k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var_ratio) + 1)
    return mu.ravel(), U[:, :k], lam[:k], lam

def pca_transform(images, mu, U):
    """Feature vectors y = U^T (x - mu), one row per image."""
    X = images.reshape(images.shape[0], -1).astype(np.float64)
    return (X - mu) @ U

def pca_reconstruct(Y, mu, U, shape):
    return (Y @ U.T + mu).reshape((-1,) + shape)

if __name__ == '__main__':
    import skimage.data as D
    faces = D.lfw_subset()[:100]                             # m = 100 face images, 25 x 25
    m, r, c = faces.shape
    mu, U, lam_k, lam = pca_fit(faces, var_ratio=0.95)
    k = U.shape[1]
    Y = pca_transform(faces, mu, U)
    print('m=%d r=%d c=%d n=%d, non-zero eigenvalues=%d, k(95%%)=%d' % (m, r, c, r * c, len(lam), k))
    cum = np.cumsum(lam) / lam.sum()
    for kk in (1, 5, 10, 20, 30, 50):
        print('k=%d cumulative variance %.3f' % (kk, cum[kk - 1]))
    # checks: orthonormal basis, uncorrelated features, variance of y_i = lambda_i
    print('max |U^T U - I| = %.2e' % np.abs(U.T @ U - np.eye(k)).max())
    CY = np.cov(Y.T, bias=True)
    print('max |offdiag cov(Y)| = %.2e, max |diag - lambda| = %.2e' % (np.abs(CY - np.diag(np.diag(CY))).max(), np.abs(np.diag(CY) - lam_k).max()))
    # reconstruction error = sum of discarded eigenvalues
    for kk in (5, 20, k):
        Uk = U[:, :kk] if kk <= k else None
        mu2, Ukk, _, _ = pca_fit(faces, k=kk)
        R = pca_reconstruct(pca_transform(faces, mu2, Ukk), mu2, Ukk, (r, c))
        err = np.mean(np.sum((faces - R).reshape(m, -1) ** 2, axis=1))
        print('k=%d mean ||x-x_hat||^2 = %.4f, sum discarded lambda = %.4f' % (kk, err, lam[kk:].sum()))
    # compare with scikit-learn (optional)
    try:
        from sklearn.decomposition import PCA
        p = PCA(n_components=k, svd_solver='full').fit(faces.reshape(m, -1))
        Ys = p.transform(faces.reshape(m, -1))
        print('sklearn match (up to sign): %.2e' % np.abs(np.abs(Ys) - np.abs(Y)).max())
    except ImportError:
        pass

    # ---------------- figures
    fig, ax = plt.subplots(3, 8, figsize=(10, 4.2))
    for j in range(8):
        ax[0, j].imshow(faces[j], cmap='gray'); ax[0, j].set_title('ảnh %d' % (j + 1), fontsize=7)
    ax[1, 0].imshow(mu.reshape(r, c), cmap='gray'); ax[1, 0].set_title('mặt trung bình', fontsize=7)
    for j in range(1, 8):
        ax[1, j].imshow(U[:, j - 1].reshape(r, c), cmap='gray'); ax[1, j].set_title('$a_{%d}$, $\\lambda$=%.2f' % (j - 1, lam[j - 1]), fontsize=7)
    x0 = faces[0]
    ax[2, 0].imshow(x0, cmap='gray'); ax[2, 0].set_title('ảnh 1 (gốc)', fontsize=7)
    for j, kk in enumerate((1, 3, 5, 10, 20, 40, 60), 1):
        mu2, Ukk, _, _ = pca_fit(faces, k=kk)
        ax[2, j].imshow(pca_reconstruct(pca_transform(x0[None], mu2, Ukk), mu2, Ukk, (r, c))[0], cmap='gray', vmin=0, vmax=1)
        ax[2, j].set_title('$k=%d$' % kk, fontsize=7)
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout(); plt.savefig(OUT + 'w8_eigenfaces.png', dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 2.9))
    ax[0].bar(np.arange(1, 31), lam[:30] / lam.sum(), color='tab:blue'); ax[0].set_xlabel('chỉ số thành phần $i+1$'); ax[0].set_ylabel('$\\lambda_i / \\sum\\lambda$'); ax[0].set_title('Tỉ lệ phương sai của từng thành phần', fontsize=9)
    ax[1].plot(np.arange(1, len(lam) + 1), cum, 'k'); ax[1].axhline(0.95, color='r', ls='--', lw=0.8); ax[1].axvline(k, color='r', ls=':', lw=0.8)
    ax[1].set_xlabel('số thành phần giữ lại $k$'); ax[1].set_title('Phương sai tích lũy (đường đỏ: 95%%, $k=%d$)' % k, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT + 'w8_variance.png', dpi=120); plt.close()

    # 2-D illustration: principal axes and decorrelation
    rng = np.random.default_rng(1)
    P = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2.2]], 400)
    Cp = np.cov(P.T, bias=True); lp, Vp = np.linalg.eigh(Cp); o = np.argsort(lp)[::-1]; lp, Vp = lp[o], Vp[:, o]
    Q = (P - P.mean(0)) @ Vp
    print('2D: cov(x)=', np.round(Cp, 2).tolist(), 'eig=', np.round(lp, 2).tolist(), 'cov(y)=', np.round(np.cov(Q.T, bias=True), 3).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].scatter(P[:, 0], P[:, 1], s=4, c='0.5')
    for i, col in enumerate(('r', 'b')):
        v = Vp[:, i] * 2 * np.sqrt(lp[i]); ax[0].annotate('', xy=v, xytext=(0, 0), arrowprops=dict(color=col, width=1.5, headwidth=7))
        ax[0].text(*(v * 1.12), '$a_%d$ ($\\lambda_%d$=%.2f)' % (i, i, lp[i]), color=col, fontsize=8)
    ax[0].set_aspect('equal'); ax[0].set_title('Dữ liệu tương quan và hai trục chính', fontsize=9); ax[0].set_xlabel('$x(0)$'); ax[0].set_ylabel('$x(1)$')
    ax[1].scatter(Q[:, 0], Q[:, 1], s=4, c='0.5'); ax[1].set_aspect('equal'); ax[1].set_title('Sau phép chiếu $y = A^T(x-\\mu)$: không tương quan', fontsize=9)
    ax[1].set_xlabel('$y(0)$'); ax[1].set_ylabel('$y(1)$')
    plt.tight_layout(); plt.savefig(OUT + 'w8_axes2d.png', dpi=120); plt.close()
