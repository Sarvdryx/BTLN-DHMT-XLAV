# BTLN XLAV - tuan 5/6. Requirements: numpy, opencv-python, scikit-image, matplotlib
# Run from XLAV/code/ ; figures are written to XLAV/images/
import numpy as np

# ---------- Geometric transform with backward mapping + bilinear interpolation ----------
def bilinear(f, x, y):
    """Sample image f (H x W [x C]) at real coordinates (x = column, y = row).
    l = floor(x), a = x - l in [0,1); k = floor(y), b = y - k in [0,1)."""
    H, W = f.shape[:2]
    l = np.floor(x).astype(int); k = np.floor(y).astype(int)
    a = x - l; b = y - k
    inside = (l >= 0) & (k >= 0) & (l <= W - 1) & (k <= H - 1)
    l1 = np.clip(l + 1, 0, W - 1); k1 = np.clip(k + 1, 0, H - 1)
    l = np.clip(l, 0, W - 1); k = np.clip(k, 0, H - 1)
    if f.ndim == 3:
        a = a[..., None]; b = b[..., None]; inside = inside[..., None]
    f = f.astype(np.float64)
    g = ((1 - a) * (1 - b) * f[k, l] + a * (1 - b) * f[k, l1]
         + (1 - a) * b * f[k1, l] + a * b * f[k1, l1])
    return np.where(inside, g, 0.0)

def warp_affine(f, A, out_shape=None):
    """g(x', y') = f(T^{-1}(x', y')), A is the 2x3 forward affine matrix."""
    H, W = out_shape or f.shape[:2]
    M = np.vstack([A, [0, 0, 1]]); Minv = np.linalg.inv(M)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = Minv[0, 0] * xx + Minv[0, 1] * yy + Minv[0, 2]
    y = Minv[1, 0] * xx + Minv[1, 1] * yy + Minv[1, 2]
    return bilinear(f, x, y)

def random_affine(H, W, rng, max_rot=20, scale=(0.85, 1.15), shift=0.08, shear=0.1):
    t = np.deg2rad(rng.uniform(-max_rot, max_rot)); s = rng.uniform(*scale)
    sh = rng.uniform(-shear, shear)
    R = s * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.array([[1, sh], [0, 1]])
    c = np.array([W / 2, H / 2]); d = rng.uniform(-shift, shift, 2) * [W, H]
    return np.hstack([R, (c - R @ c + d)[:, None]])

def mixup(x1, y1, x2, y2, alpha, rng):
    lam = rng.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2, lam

# ---------- Noise from an arbitrary distribution (inverse transform sampling) ----------
def sample_from_pdf(pdf, support, size, rng):
    """Draw samples from any 1-D pdf given on a grid `support`."""
    p = np.clip(np.asarray(pdf, float), 0, None)
    cdf = np.cumsum(p); cdf /= cdf[-1]
    u = rng.random(size)
    return np.interp(u, cdf, support)          # x = F^{-1}(u)

def add_noise(f, pdf, support, mode='additive', amount=1.0, rng=None):
    """mode='additive':  g = f + n,  n ~ pdf
       mode='multiplicative': g = f * (1 + n)
       mode='impulse': a fraction `amount` of pixels is replaced by n ~ pdf"""
    rng = rng or np.random.default_rng()
    f = f.astype(np.float64)
    n = sample_from_pdf(pdf, support, f.shape, rng)
    if mode == 'additive':
        g = f + n
    elif mode == 'multiplicative':
        g = f * (1 + n)
    elif mode == 'impulse':
        mask = rng.random(f.shape) < amount
        g = np.where(mask, n, f)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)

# ---------- Kernels ----------
def gaussian_kernel(n, sigma):
    r = n // 2; y, x = np.mgrid[-r:r + 1, -r:r + 1]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return h / h.sum()

def log_kernel(n, sigma):
    r = n // 2; y, x = np.mgrid[-r:r + 1, -r:r + 1]; q = (x**2 + y**2) / (2 * sigma**2)
    h = (q - 1) / (np.pi * sigma**4) * np.exp(-q)
    return h - h.mean()          # force sum = 0: flat regions give 0 response

def conv2(f, h):
    """g(x,y) = sum_{i,j} f(x+i, y+j) h(i,j)  (correlation form used in the slides),
    replicate border."""
    r = h.shape[0] // 2; fp = np.pad(f.astype(np.float64), r, mode='edge')
    g = np.zeros(f.shape, np.float64); H, W = f.shape
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            g += h[i + r, j + r] * fp[r + i:r + i + H, r + j:r + j + W]
    return g

def windows(f, r):
    fp = np.pad(f.astype(np.float64), r, mode='edge'); H, W = f.shape
    return np.stack([fp[r + i:r + i + H, r + j:r + j + W] for i in range(-r, r + 1) for j in range(-r, r + 1)])

def median_filter(f, n=3):
    return np.median(windows(f, n // 2), axis=0)

def constrained_mean(f, lo, hi, n=5):
    """Constrained averaging (L5-cont, slide 4): only pixels whose grey level lies in [lo,hi]
    (the 'dirt') are replaced, and only pixels outside [lo,hi] contribute to the average."""
    Wd = windows(f, n // 2); valid = (Wd < lo) | (Wd > hi)
    s = (Wd * valid).sum(0); c = valid.sum(0)
    g = f.astype(np.float64).copy(); dirt = (f >= lo) & (f <= hi) & (c > 0)
    g[dirt] = s[dirt] / c[dirt]
    return g, dirt

def gradient_weighted_mean(f):
    """Gradient inverse weighted smoothing (L5-cont, slide 5), 3x3 window."""
    Wd = windows(f, 1); c = Wd[4]
    d = np.abs(Wd - c); delta = np.where(d == 0, 2.0, 1.0 / np.maximum(d, 1e-12))
    delta[4] = 0
    w = 0.5 * delta / delta.sum(0)
    return 0.5 * c + (w * Wd).sum(0)

def rotating_mask(f):
    """Rotating-mask averaging (L5-cont, slides 6-8): 9 masks of size 3x3 inside the 5x5
    window (centre + 8 shifted); choose the one with minimum variance."""
    fp = np.pad(f.astype(np.float64), 2, mode='edge'); H, W = f.shape
    means, vars_ = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            S = np.stack([fp[2 + di + i:2 + di + i + H, 2 + dj + j:2 + dj + j + W]
                          for i in (-1, 0, 1) for j in (-1, 0, 1)])
            means.append(S.mean(0)); vars_.append(S.var(0))
    means = np.stack(means); vars_ = np.stack(vars_)
    idx = vars_.argmin(0)
    return np.take_along_axis(means, idx[None], 0)[0]

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float))**2)
    return 10 * np.log10(255**2 / mse)

# ---------- Edges ----------
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
SOBEL_Y = SOBEL_X.T
LAP4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
LAP8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float)

def gradient_edges(f, T):
    fx = conv2(f, SOBEL_X); fy = conv2(f, SOBEL_Y)
    e = np.hypot(fx, fy); return e, (e >= T)

def zero_crossings(L, T):
    """Mark (y,x) when L changes sign between (y,x) and a neighbour in one of 4 directions
    (right, down, two diagonals) and the jump |L1 - L2| >= T (rejects weak noise crossings)."""
    H, W = L.shape; E = np.zeros((H, W), bool)
    P = np.pad(L, 1, mode='edge')
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        N = P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        E |= (L * N < 0) & (np.abs(L - N) >= T)
    return E
