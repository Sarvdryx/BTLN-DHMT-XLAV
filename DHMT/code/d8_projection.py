# BTLN DHMT - tuan 8: phep chieu phoi canh / song song.
# Requirements: numpy, matplotlib.  Run from DHMT/code/ ; figures -> DHMT/images/
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.environ.get('OUT', '../images/')

def per_matrix(C, R0, N):
    """Perspective projection with centre C onto plane (R0, N) -- slide 14."""
    C, R0, N = map(np.asarray, (C, R0, N))
    d0, d1 = N @ R0, N @ C; d = d0 - d1
    M = np.zeros((4, 4))
    M[:3, :3] = d * np.eye(3) + np.outer(C, N)
    M[:3, 3] = -C * d0
    M[3, :3] = N; M[3, 3] = -d1
    return M

def par_matrix(V, R0, N):
    """Parallel projection along V onto plane (R0, N): P' = P + k V, N.(P'-R0) = 0."""
    V, R0, N = map(np.asarray, (V, R0, N))
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - np.outer(V, N) / (N @ V)
    M[:3, 3] = V * (N @ R0) / (N @ V)
    return M

def apply(M, P):
    """P: (k, 3) points -> projected (k, 3) points (homogeneous division)."""
    H = np.c_[P, np.ones(len(P))] @ M.T
    return H[:, :3] / H[:, 3:4]

def camera(eye, target, up=(0, 0, 1), d=1.0):
    """Centre of projection = eye; view plane perpendicular to the viewing direction at
    distance d in front of the eye. Returns (M, R0, u, v): 3-D projection matrix and the
    2-D frame (R0; u = right, v = up) of the view plane."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    w = target - eye; w /= np.linalg.norm(w)          # viewing direction = plane normal
    u = np.cross(w, up); u /= np.linalg.norm(u)       # screen x (right)
    v = np.cross(u, w)                                # screen y (up)
    R0 = eye + d * w
    return per_matrix(eye, R0, w), R0, u, v

def to_screen(P3, R0, u, v):
    """3-D points on the view plane -> 2-D screen coordinates."""
    Q = P3 - R0
    return np.c_[Q @ u, Q @ v]

def project_view(P, eye, target, up=(0, 0, 1), d=1.0):
    M, R0, u, v = camera(eye, target, up, d)
    return to_screen(apply(M, P), R0, u, v)

# ------------------------------------------------------------ simple 3-D objects
def sphere_wire(c, r, nlat=12, nlon=24, k=60):
    c = np.asarray(c, float); lines = []
    t = np.linspace(-np.pi, np.pi, k)
    for phi in np.linspace(-np.pi / 2, np.pi / 2, nlat + 1)[1:-1]:          # parallels
        lines.append(c + r * np.c_[np.cos(phi) * np.cos(t), np.cos(phi) * np.sin(t), np.full(k, np.sin(phi))])
    p = np.linspace(-np.pi / 2, np.pi / 2, k)
    for th in np.linspace(-np.pi, np.pi, nlon, endpoint=False):              # meridians
        lines.append(c + r * np.c_[np.cos(p) * np.cos(th), np.cos(p) * np.sin(th), np.sin(p)])
    return lines

def cube_edges(o, s):
    o = np.asarray(o, float); V = o + s * np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    E = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count('1') == 1]
    return [V[[a, b]] for a, b in E]

if __name__ == '__main__':
    # ---- 1) sphere under the slide-4 perspective (C(0,0,-d), view plane z = 0)
    d = 3.0; c = np.array([3.5, 2.5, 3.0]); r = 1.5
    M = per_matrix([0, 0, -d], [0, 0, 0], [0, 0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for L in sphere_wire(c, r):
        Q = apply(M, L); ax[0].plot(Q[:, 0], Q[:, 1], color='0.6', lw=0.6)
    # analytic silhouette: (|s|^2 - r^2)(x^2 + y^2 + d^2) = (cx x + cy y + (cz + d) d)^2
    s = c - np.array([0, 0, -d]); allq = np.vstack([apply(M, L) for L in sphere_wire(c, r)]); lo, hi = allq[:, :2].min(0) - 0.3, allq[:, :2].max(0) + 0.3
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 700), np.linspace(lo[1], hi[1], 700))
    F = (s @ s - r * r) * (X**2 + Y**2 + d * d) - (s[0] * X + s[1] * Y + s[2] * d) ** 2
    ax[0].contour(X, Y, F, [0], colors='r', linewidths=1.5)
    cc = apply(M, c[None])[0]; ax[0].plot(*cc[:2], 'b+', ms=10)
    ax[0].set_aspect('equal'); ax[0].set_title('Mặt cầu c=(3,5; 2,5; 3), r=1,5, d=3:\nlưới chiếu (xám), đường viền ellipse (đỏ), ảnh của tâm (+)', fontsize=8)
    # on-axis sphere at several depths: circle radius rho = d r / sqrt((cz+d)^2 - r^2)
    for cz, col in ((2, 'tab:red'), (6, 'tab:green'), (14, 'tab:blue')):
        for L in sphere_wire([0, 0, cz], 1.0, 8, 16):
            Q = apply(M, L); ax[1].plot(Q[:, 0], Q[:, 1], color=col, lw=0.5)
        rho = d * 1.0 / np.sqrt((cz + d) ** 2 - 1.0)
        print('on-axis sphere cz=%g: silhouette radius %.4f' % (cz, rho))
        t = np.linspace(0, 2 * np.pi, 200); ax[1].plot(rho * np.cos(t), rho * np.sin(t), color=col, lw=1.6, label='$c_z=%g$, $\\rho=%.3f$' % (cz, rho))
    ax[1].set_aspect('equal'); ax[1].legend(fontsize=7, loc='upper right'); ax[1].set_title('Cùng mặt cầu r=1 trên trục, xa dần: ảnh nhỏ dần', fontsize=8)
    plt.tight_layout(); plt.savefig(OUT + 'd8_sphere.png', dpi=120); plt.close()
    # numerical check of the silhouette: max |F| on projected tangent points
    e = np.array([0, 0, -d]); ok = []
    for th in np.linspace(0, 2 * np.pi, 50, endpoint=False):
        # tangent points: those P on sphere with (P - c).(P - e) = 0  (radius _|_ line of sight)
        sn = s / np.linalg.norm(s); a = np.cross(sn, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(sn, a)
        D = np.linalg.norm(s); rr = r * np.sqrt(D * D - r * r) / D; cen = c - sn * r * r / D
        P = cen + rr * (np.cos(th) * a + np.sin(th) * b); q = apply(M, P[None])[0]
        ok.append((s @ s - r * r) * (q[0]**2 + q[1]**2 + d * d) - (s[0] * q[0] + s[1] * q[1] + s[2] * d) ** 2)
    print('silhouette check max |F| = %.2e' % np.max(np.abs(ok)))

    # ---- 2) look-at camera: same scene from three user-chosen viewpoints
    scene = cube_edges([-1, -1, 0], 2) + sphere_wire([3.5, 0, 1], 1, 8, 16)
    views = [((8, -6, 4), (1.2, 0, 1)), ((0, -10, 1.5), (1.2, 0, 1)), ((1.2, 0.01, 12), (1.2, 0, 0))]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, (eye, tgt) in zip(ax, views):
        up = (0, 1, 0) if eye[2] > 10 else (0, 0, 1)
        for L in scene:
            Q = project_view(L, eye, tgt, up, d=1.0); a.plot(Q[:, 0], Q[:, 1], 'k', lw=0.7)
        a.set_aspect('equal'); a.axis('off'); a.set_title('mắt C=%s, nhìn tới T=%s' % (eye, tgt), fontsize=8)
    plt.tight_layout(); plt.savefig(OUT + 'd8_lookat.png', dpi=120); plt.close()

    # ---- 3) perspective vs parallel: rails and a row of equal cubes
    rails = [np.array([[x, 0, z0], [x, 0, 40]]) for x in (-1, 1) for z0 in (0,)]
    sleepers = [np.array([[-1, 0, z], [1, 0, z]]) for z in range(0, 41, 2)]
    cubes = sum([cube_edges([2, 0, z], 1) for z in (0, 5, 10, 20)], [])
    objs = rails + sleepers + cubes
    Mp = per_matrix([0, 1.5, -4], [0, 0, 0], [0, 0, 1])
    Mo = par_matrix([0.5, -0.3, 1], [0, 0, 0], [0, 0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for L in objs:
        Q = apply(Mp, L); ax[0].plot(Q[:, 0], Q[:, 1], 'k', lw=0.7)
        Q = apply(Mo, L); ax[1].plot(Q[:, 0], Q[:, 1], 'k', lw=0.7)
    vp = apply(Mp, np.array([[0, 0, 1e6]]))[0]; ax[0].plot(*vp[:2], 'ro', ms=4); ax[0].annotate('điểm tụ', vp[:2], xytext=(6, -12), textcoords='offset points', color='r', fontsize=8)
    ax[0].set_aspect('equal'); ax[0].set_title('Phối cảnh: ray song song hội tụ, khối lập phương xa nhỏ dần', fontsize=8)
    ax[1].set_aspect('equal'); ax[1].set_title('Song song (xiên) theo V=(0,5; -0,3; 1): ray vẫn song song, kích thước không đổi', fontsize=8)
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.savefig(OUT + 'd8_per_vs_par.png', dpi=120); plt.close()
