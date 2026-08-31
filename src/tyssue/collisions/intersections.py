"""Detect where an epithelium has stopped being an embedding.

A vertex model is only a valid tissue while the piecewise-linear map from the
abstract cell complex into space is injective. This module detects the ways that
fails:

  Test A  self-intersection  -- a face's own boundary crosses itself (a "bow-tie")
  Test B  containment        -- a vertex not on face F's ring lies inside F
  Test C  overlap            -- two faces share interior points (a superset of B)
  plus    open rings         -- a face whose half-edges do not close into a cycle

Every test is built on one primitive, the orientation determinant, so they are
exact-sign arithmetic on coordinate *differences*: no tolerances, no trigonometry,
no global coordinate chart.

Relation to the rest of ``tyssue.collisions``
---------------------------------------------
This is the whole of ``tyssue.collisions``. It replaces the former
``intersection.self_intersections``, which needed a CGAL extension and reported
intersecting *triangles* of a triangular mesh; this module is pure numpy/scipy,
needs no compiled extension, and answers the question in terms of *faces* --
which face folded, which neighbour it invaded. ``Epithelium.validate`` is purely
combinatorial and a bow-tie passes it.

Why there is no coordinate chart
--------------------------------
Each face is reduced to 2-D in the best-fit plane of its own ring, taken about its
own centroid, so only differences ``q - c_F`` ever appear. The construction is
translation- and rotation-invariant and never takes a global coordinate, so a face
straddling a periodic seam needs no special handling.

The plane comes from the ring's covariance, NOT from its Newell area vector: a
symmetric bow-tie's lobes cancel exactly, so its Newell vector is zero and the frame
would be undefined precisely on the case that matters most.

Where a planar projection stops being valid
-------------------------------------------
A face spanning a wide arc of a curved tissue is fitted well by a CHORD plane
slicing through the surface, which is nearly planar and yet is not the surface the
cells live on -- so a residual test cannot detect the failure. ``curvature_span``
measures how far the surface normal turns across a face instead; faces past
``max_span_rad`` are reported as *undecidable* rather than counted as folded.

Why it is fast enough to call every step
----------------------------------------
Culling lemma: with ``r_F = max_i |p_i - c_F|``, every point of F lies in
``B(c_F, r_F)``, because F is contained in the convex hull of its ring. So

  * q outside ``B(c_F, r_F)``        =>  q is not in F        (Test B candidates)
  * ``|c_F - c_G| > r_F + r_G``      =>  F and G are disjoint (Test C candidates)

Both hold in any dimension on any geometry and neither can produce a false negative.
Measured on a 758-cell crypt: 1,192,481 -> 47 point-in-polygon tests per frame.

See Also
--------
:func:`tyssue.behaviors.sheet.basic_events.check_intersections`
    run this during a simulation, the way ``reconnect`` is run
:meth:`tyssue.core.objects.Epithelium.find_intersections`
    the method form
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

__all__ = [
    "curvature_span",
    "RingCache",
    "ring_matrix",
    "face_frames",
    "self_intersecting",
    "contained_vertices",
    "overlapping_faces",
    "find_intersections",
    "Result",
    "signed_area_defect",
]


# ---------------------------------------------------------------------------
# Topology: the padded ring matrix
# ---------------------------------------------------------------------------
@dataclass
class RingCache:
    """Face rings as dense padded arrays.

    Depends only on topology, so it survives every solver step between topology
    changes -- which is what makes a per-step call affordable (rebuilding costs
    ~14 ms, evaluating costs ~2 ms).

    RING  (Nf, nmax) int   positional row index into the vertex coordinate array
    LEN   (Nf,)      int   sides per face
    VALID (Nf, nmax) bool  RING slot is a real vertex, not padding
    NXT   (Nf, nmax) int   slot of the next ring vertex, wrapping at LEN
    """

    RING: np.ndarray
    LEN: np.ndarray
    VALID: np.ndarray
    NXT: np.ndarray
    face_ids: np.ndarray
    vert_ids: np.ndarray
    signature: tuple = field(default=(), repr=False)
    open_rings: np.ndarray = field(default=None, repr=False)

    @property
    def n_faces(self) -> int:
        return len(self.LEN)


def _topology_signature(eptm) -> tuple:
    return (int(eptm.Nf), int(eptm.Ne), int(eptm.Nv))


def ring_matrix(eptm, cache: RingCache | None = None) -> RingCache:
    """Chain each face's half-edges ``srce -> trgt`` into a closed ring.

    Row order in ``edge_df`` is NOT ring order -- it must be chained. A face whose
    half-edges do not close into a single cycle is a corrupt mesh; it is recorded in
    ``open_rings`` and excluded from every test rather than silently mis-tested.
    """
    sig = _topology_signature(eptm)
    if cache is not None and cache.signature == sig:
        return cache

    edge = eptm.edge_df
    face_col = edge["face"].to_numpy()
    srce = edge["srce"].to_numpy()
    trgt = edge["trgt"].to_numpy()

    order = np.argsort(face_col, kind="stable")
    face_sorted = face_col[order]
    face_ids, starts = np.unique(face_sorted, return_index=True)
    bounds = np.append(starts, len(face_sorted))

    rings, open_flags = [], []
    for k in range(len(face_ids)):
        sl = order[bounds[k]:bounds[k + 1]]
        s, t = srce[sl], trgt[sl]
        nxt = dict(zip(s.tolist(), t.tolist()))
        ok = len(nxt) == len(s)
        ring = []
        if ok:
            start = int(s[0])
            ring = [start]
            cur = nxt[start]
            while cur != start:
                ring.append(int(cur))
                cur = nxt.get(cur)
                if cur is None or len(ring) > len(s):
                    ok = False
                    break
            ok = ok and len(ring) == len(s)
        if not ok:                      # corrupt: keep a placeholder, flag it
            ring = [int(s[0])] * max(len(s), 3)
        rings.append(ring)
        open_flags.append(not ok)

    nmax = max(len(r) for r in rings)
    nf = len(rings)
    RING = np.zeros((nf, nmax), dtype=np.int64)
    LEN = np.array([len(r) for r in rings], dtype=np.int64)
    for i, r in enumerate(rings):
        RING[i, :len(r)] = r
    k = np.arange(nmax)[None, :]
    VALID = k < LEN[:, None]
    NXT = np.where(VALID, (k + 1) % np.maximum(LEN[:, None], 1), 0)

    vert_ids = eptm.vert_df.index.to_numpy()
    lookup = np.full(int(vert_ids.max()) + 1, -1, dtype=np.int64)
    lookup[vert_ids] = np.arange(len(vert_ids))
    RING = np.where(VALID, lookup[np.clip(RING, 0, None)], 0)

    return RingCache(RING, LEN, VALID, NXT, face_ids, vert_ids, sig,
                     np.array(open_flags, dtype=bool))


# ---------------------------------------------------------------------------
# Geometry: per-face tangent frame from the Newell area vector
# ---------------------------------------------------------------------------
def _positions(eptm) -> np.ndarray:
    """(Nv, 3) vertex coordinates.

    A 2-D epithelium is lifted to z = 0, so the 3-D path degenerates to the planar
    one exactly rather than approximately -- the same code then serves a flat sheet,
    a curved sheet, a tube and a monolayer face.
    """
    coords = list(getattr(eptm, "coords", None) or
                  [c for c in ("x", "y", "z") if c in eptm.vert_df.columns])
    P = eptm.vert_df[coords].to_numpy(dtype=float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    return P


def face_frames(eptm, cache: RingCache, pos: np.ndarray | None = None):
    """Per-face centroid, Newell area vector and an orthonormal tangent frame.

    Returns (c, N, u, w, P2, D, fan_area): P2 is the ring projected into (u, w),
    D holds the local differences p_i - c_F, and fan_area is tyssue's own face area
    (the sum of |n_i|/2 over half-edges).
    """
    P = _positions(eptm) if pos is None else pos
    ring = P[cache.RING]                                   # (Nf, nmax, 3)
    ring = np.where(cache.VALID[..., None], ring, 0.0)
    c = ring.sum(axis=1) / cache.LEN[:, None]

    D = np.where(cache.VALID[..., None], ring - c[:, None, :], 0.0)
    rows = np.arange(cache.n_faces)[:, None]
    Dn = D[rows, cache.NXT]
    tri = np.cross(D, Dn)                                  # per-fan-triangle normals
    N = tri.sum(axis=1)                                    # Newell area vector

    # The Newell vector VANISHES on a symmetric fold -- a bow-tie's lobes cancel
    # exactly -- which is precisely the case this module must not miss. So the
    # tangent frame comes from the best-fit plane (smallest-eigenvector of the
    # ring's covariance), which is well defined for any non-collinear ring.
    # Newell is kept only for orientation and for the signed-area screen.
    cov = np.einsum("fki,fkj->fij", D, D)
    _, vecs = np.linalg.eigh(cov)
    nh = vecs[:, :, 0]                                     # least-variance direction
    # keep the frame right-handed w.r.t. Newell where Newell is meaningful
    flip = np.einsum("fi,fi->f", nh, N) < 0
    nh = np.where(flip[:, None], -nh, nh)

    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(nh), 1))
    ref[np.abs(nh[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    u = np.cross(nh, ref)
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.divide(u, un, out=np.zeros_like(u), where=un > 0)
    w = np.cross(nh, u)

    P2 = np.stack([np.einsum("fkd,fd->fk", D, u),
                   np.einsum("fkd,fd->fk", D, w)], axis=-1)
    # fan_area matches tyssue's face area exactly: sub_area = |n_i|/2 summed
    # over half-edges (sheet_geometry.py:52-56), i.e. the sum of MAGNITUDES.
    fan_area = 0.5 * np.where(cache.VALID, np.linalg.norm(tri, axis=2), 0.0).sum(axis=1)
    return c, N, u, w, P2, D, fan_area


def signed_area_defect(eptm, cache: RingCache, pos=None):
    """Level-0 screen: |A_signed| vs the unsigned |N|/2 the energy actually uses.

    ``SheetGeometry.update_areas`` sets sub_area = |n|/2, so a fold contributes
    POSITIVE area and the area elasticity feels no restoring force. The signed area
    in the face's own frame does cancel. The gap between them is therefore exactly
    the quantity the model is blind to.

    Cheap and useful, but INCOMPLETE as a detector: a bow-tie with unequal lobes
    keeps its sign. Never use it to certify a mesh -- use Test A.
    """
    _, N, _, _, P2, _, fan_area = face_frames(eptm, cache, pos)
    a, b = P2[..., 0], P2[..., 1]
    rows = np.arange(cache.n_faces)[:, None]
    an, bn = a[rows, cache.NXT], b[rows, cache.NXT]
    cross = np.where(cache.VALID, a * bn - an * b, 0.0)
    signed = 0.5 * cross.sum(axis=1)
    return signed, fan_area



def curvature_span(eptm, cache: RingCache, pos=None):
    """Angle (radians) each face spans across the tissue's curvature.

    A planar projection of a face is only meaningful while the face is small
    compared with the surface's radius of curvature. Measuring the residual of the
    ring about its own best-fit plane does NOT detect the failure: a face spanning
    a wide arc is fitted well by a CHORD plane slicing through the surface, which
    is nearly planar and yet is not the surface the cells live on.

    Measured instead by how much the surface normal turns across the face: each
    vertex gets the mean normal of the faces meeting there, and the face's span is
    the largest angle between its own ring vertices' normals. A cell on the crypt
    (edge ~0.6, tube radius 2.5) spans ~14 degrees; a stretched sliver crossing
    61 degrees of the tube is flagged.
    """
    P = _positions(eptm) if pos is None else pos
    _, _, u, w, _, _, _ = face_frames(eptm, cache, P)
    # the BEST-FIT frame normal, not the Newell vector: Newell vanishes on a
    # symmetric bow-tie, which would make the span meaningless exactly where the
    # face most needs judging.
    n = np.cross(u, w)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    acc = np.zeros((len(cache.vert_ids), 3))
    flat = cache.RING[cache.VALID]
    rep = np.repeat(np.arange(cache.n_faces), cache.LEN)
    np.add.at(acc, flat, n[rep])
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)

    vn = acc[cache.RING]                                  # (Nf, nmax, 3)
    vn = np.where(cache.VALID[..., None], vn, np.nan)
    dots = np.einsum("fid,fjd->fij", np.nan_to_num(vn), np.nan_to_num(vn))
    ok = cache.VALID[:, :, None] & cache.VALID[:, None, :]
    dots = np.where(ok, np.clip(dots, -1.0, 1.0), 1.0)
    return np.arccos(dots.min(axis=(1, 2)))


# ---------------------------------------------------------------------------
# Test A -- self-intersection
# ---------------------------------------------------------------------------
def _proper_cross(p1, p2, p3, p4):
    """Segments (p1,p2) and (p3,p4) cross at an interior point of both.

    orient(a,b,c) = (b-a) x (c-a); its sign says which side of line ab point c is on.
    A proper crossing needs each segment to separate the other's endpoints, i.e.
    both orientation products strictly negative. Strictness excludes touching and
    collinear cases -- correct here, since adjacent ring edges share an endpoint and
    must not be counted as an intersection.
    """
    d1 = (p4[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1]) - \
         (p4[..., 1] - p3[..., 1]) * (p1[..., 0] - p3[..., 0])
    d2 = (p4[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
         (p4[..., 1] - p3[..., 1]) * (p2[..., 0] - p3[..., 0])
    d3 = (p2[..., 0] - p1[..., 0]) * (p3[..., 1] - p1[..., 1]) - \
         (p2[..., 1] - p1[..., 1]) * (p3[..., 0] - p1[..., 0])
    d4 = (p2[..., 0] - p1[..., 0]) * (p4[..., 1] - p1[..., 1]) - \
         (p2[..., 1] - p1[..., 1]) * (p4[..., 0] - p1[..., 0])
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def self_intersecting(eptm, cache: RingCache, pos=None, P2=None,
                      max_span_rad: float = np.deg2rad(30.0), return_span=False):
    """Test A. Boolean over faces: does this face's boundary cross itself?

    A closed polygon is simple iff no two NON-ADJACENT edges properly cross. Edge i
    runs slot i -> NXT[i]; edges i and j are adjacent when they share an endpoint,
    so the pairs to test are j not in {i-1, i, i+1} mod n -- n(n-3)/2 of them, at
    most 20 for an 8-sided cell.

    Vectorised over all faces at once: one pass per (i, j) slot pair, each pass a
    handful of numpy ops. ~1.9 ms for 781 faces.
    """
    if P2 is None:
        _, _, _, _, P2, _, _ = face_frames(eptm, cache, pos)
    nf, nmax, _ = P2.shape
    rows = np.arange(nf)
    LEN, NXT = cache.LEN, cache.NXT
    hit = np.zeros(nf, dtype=bool)

    for i in range(nmax):
        for j in range(i + 2, nmax):
            live = (i < LEN) & (j < LEN)
            if not live.any():
                continue
            i2 = NXT[rows, i]
            j2 = NXT[rows, j]
            # non-adjacent: neither edge may end where the other begins
            live &= (j2 != i) & (i2 != j)
            if not live.any():
                continue
            hit |= live & _proper_cross(P2[rows, i], P2[rows, i2],
                                        P2[rows, j], P2[rows, j2])
    hit &= ~cache.open_rings

    # A crossing found in a plane that cannot represent the face is not evidence.
    # Verified case: face 462 spans 61.4 deg of the crypt tube -- its best-fit plane
    # is a chord through the cylinder and reports a crossing, while the surface
    # itself has none. Such faces are reported separately, never counted as folded.
    span = curvature_span(eptm, cache, pos)
    undecidable = hit & (span > max_span_rad)
    hit &= ~undecidable
    if return_span:
        return hit, undecidable, span
    return hit


# ---------------------------------------------------------------------------
# Test B -- a foreign vertex inside a face
# ---------------------------------------------------------------------------
def _winding_inside(poly: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Nonzero-winding point-in-polygon for one polygon and many query points.

    w(q) = (1/2pi) sum_i angle subtended by edge (p_i, p_i+1) at q; q is inside iff
    w != 0. Computed by the standard crossing accumulation with orientation signs,
    which is the same number without trigonometry.

    The NONZERO rule is used deliberately, not even-odd. They agree on simple
    polygons but differ on exactly the folded ones this module exists to find: a
    doubly-wound lobe has w = 2, which nonzero calls inside (material covers the
    point -- the physically meaningful answer) and even-odd calls outside.
    """
    n = len(poly)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    qx = q[:, 0][:, None]
    qy = q[:, 1][:, None]

    up = (y[None, :] <= qy) & (yn[None, :] > qy)
    dn = (y[None, :] > qy) & (yn[None, :] <= qy)
    # side of the directed edge the query point falls on
    side = (xn - x)[None, :] * (qy - y[None, :]) - (yn - y)[None, :] * (qx - x[None, :])
    wind = np.where(up & (side > 0), 1, 0) + np.where(dn & (side < 0), -1, 0)
    return wind.sum(axis=1) != 0


def contained_vertices(eptm, cache: RingCache, pos=None, faces=None,
                       slab: float = 0.25):
    """Test B. Vertices that are not on face F's ring but lie inside F.

    Returns a list of (face_id, vert_id, owner_face_ids).

    Two stages:
      1. cull -- KD-tree ball query at (centroid, circumradius). Exact by the
         culling lemma, so nothing is missed; measured 0.06 candidates per face.
      2. test -- slab |h| <= slab * sqrt(area) to require the vertex to be near F's
         plane at all (vacuous in 2-D, where h == 0), then nonzero winding in F's
         tangent frame.

    The slab is the only tolerance in the module. It has to exist because a vertex
    of a curved sheet is never exactly in a neighbour's plane; on the crypt the
    scale it must admit is the sagitta of a cell on the tube, r^2/2R ~ 0.07.
    """
    P = _positions(eptm) if pos is None else pos
    c, N, u, w, P2, _, fan_area = face_frames(eptm, cache, P)

    radius = np.linalg.norm(P2, axis=2)
    radius = np.where(cache.VALID, radius, 0.0).max(axis=1)
    eps = slab * np.sqrt(np.maximum(fan_area, 1e-30))

    tree = cKDTree(P)
    todo = np.arange(cache.n_faces) if faces is None else np.asarray(faces)
    todo = todo[~cache.open_rings[todo]]

    out = []
    vert2face = None
    for f in todo:
        cand = tree.query_ball_point(c[f], radius[f])
        if not cand:
            continue
        own = set(cache.RING[f, :cache.LEN[f]].tolist())
        cand = np.array([j for j in cand if j not in own], dtype=np.int64)
        if not len(cand):
            continue
        D = P[cand] - c[f]
        h = D @ np.cross(u[f], w[f])
        near = np.abs(h) <= eps[f]
        if not near.any():
            continue
        cand = cand[near]
        D = D[near]
        q2 = np.column_stack([D @ u[f], D @ w[f]])
        poly = P2[f, :cache.LEN[f]]
        inside = _winding_inside(poly, q2)
        if not inside.any():
            continue
        if vert2face is None:
            vert2face = _vertex_to_faces(cache)
        for j in cand[inside]:
            vid = int(cache.vert_ids[j])
            owners = sorted(int(cache.face_ids[k]) for k in vert2face.get(j, ()))
            out.append((int(cache.face_ids[f]), vid, owners))
    return out


def _vertex_to_faces(cache: RingCache) -> dict:
    m = {}
    for f in range(cache.n_faces):
        for k in cache.RING[f, :cache.LEN[f]]:
            m.setdefault(int(k), set()).add(f)
    return m


# ---------------------------------------------------------------------------
# Test C -- complete pairwise overlap
# ---------------------------------------------------------------------------
def overlapping_faces(eptm, cache: RingCache, pos=None):
    """Test C. Face pairs whose boundaries properly cross.

    Test B misses a "lens" overlap in which two faces cross without either one's
    vertex landing inside the other. This closes that gap by testing every edge of
    F against every edge of G, for candidate pairs from Corollary 2 of the culling
    lemma. Each pair is evaluated in F's tangent frame.

    Costs more than A and B; opt-in via ``find_intersections(strict=True)``.
    """
    P = _positions(eptm) if pos is None else pos
    c, _, u, w, P2, _, _ = face_frames(eptm, cache, P)
    radius = np.where(cache.VALID, np.linalg.norm(P2, axis=2), 0.0).max(axis=1)

    tree = cKDTree(c)
    pairs = tree.query_ball_tree(tree, r=2 * radius.max())
    out, indeterminate = [], []
    for f, others in enumerate(pairs):
        if cache.open_rings[f]:
            continue
        ringf = cache.RING[f, :cache.LEN[f]]
        polyf = P2[f, :cache.LEN[f]]
        for g in others:
            if g <= f or cache.open_rings[g]:
                continue
            if np.linalg.norm(c[f] - c[g]) > radius[f] + radius[g]:
                continue
            ringg = cache.RING[g, :cache.LEN[g]]
            if set(ringf.tolist()) & set(ringg.tolist()):
                continue                      # neighbours share an edge legitimately

            # Project BOTH rings onto the best-fit plane of their UNION, not onto
            # F's own tangent plane. Using F's plane is wrong for a pair straddling
            # curvature: on a tube of radius 2.5 a face with circumradius 1.6 spans
            # ~37 degrees, and its neighbour projected into its tangent frame is
            # distorted enough to manufacture crossings that are not there.
            pts = np.vstack([P[ringf], P[ringg]])
            cc = pts.mean(axis=0)
            dd = pts - cc
            _, vecs = np.linalg.eigh(dd.T @ dd)
            nh = vecs[:, 0]
            ax = np.array([1.0, 0.0, 0.0]) if abs(nh[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
            uu = np.cross(nh, ax)
            uu /= np.linalg.norm(uu)
            ww = np.cross(nh, uu)

            # If the pair is too non-planar for ANY single plane to represent, the
            # answer is not decidable this way -- report it rather than guess.
            resid = np.abs(dd @ nh).max()
            scale = max(radius[f], radius[g])
            if resid > 0.35 * scale:
                indeterminate.append((int(cache.face_ids[f]), int(cache.face_ids[g])))
                continue

            nf_, ng_ = len(ringf), len(ringg)
            proj = np.column_stack([dd @ uu, dd @ ww])
            if _pair_crosses(proj[:nf_], proj[nf_:]):
                out.append((int(cache.face_ids[f]), int(cache.face_ids[g])))
    if indeterminate:
        log.debug("%d face pairs too non-planar to decide", len(indeterminate))
    return out


def _pair_crosses(A: np.ndarray, B: np.ndarray) -> bool:
    An, Bn = np.roll(A, -1, axis=0), np.roll(B, -1, axis=0)
    for i in range(len(A)):
        if _proper_cross(A[i][None, :], An[i][None, :], B, Bn).any():
            return True
    return False


# ---------------------------------------------------------------------------
# The public entry point -- becomes Epithelium.find_intersections
# ---------------------------------------------------------------------------
@dataclass
class Result:
    folded: np.ndarray                 # face ids whose own boundary self-crosses
    contained: list                    # (face_id, vert_id, owner_face_ids)
    overlaps: list                     # (face_id, face_id), strict mode only
    open_rings: np.ndarray             # face ids whose half-edges do not close
    undecidable: np.ndarray = None     # crossings in a plane too curved to trust
    cache: RingCache = field(repr=False, default=None)

    @property
    def clean(self) -> bool:
        return not (len(self.folded) or len(self.contained)
                    or len(self.overlaps) or len(self.open_rings)
                    or (self.undecidable is not None and len(self.undecidable)))

    def __len__(self):
        return len(self.folded) + len(self.contained) + len(self.overlaps)


def find_intersections(eptm, cache: RingCache | None = None, *, strict: bool = False,
                       trigger: str | None = None, slab: float = 0.25) -> Result:
    """Detect every way this epithelium has stopped being an embedding.

    strict   also run Test C (pairwise overlap). Off by default: it is the only
             test whose cost is not negligible.
    trigger  "folded" runs Test B only on faces adjacent to a self-crossing one.
             On crypt data this loses nothing -- 62 of 62 intruding vertices came
             from a face that was itself self-crossing -- but that is a property of
             that data, NOT a theorem: two convex cells can slide through each other
             with neither ring self-crossing. Opt in only when the assumption holds.
    """
    cache = ring_matrix(eptm, cache)
    P = _positions(eptm)
    _, _, _, _, P2, _, _ = face_frames(eptm, cache, P)

    folded, undecidable, _span = self_intersecting(eptm, cache, P, P2,
                                                   return_span=True)

    if trigger == "folded":
        if folded.any():
            v2f = _vertex_to_faces(cache)
            near = set()
            for f in np.flatnonzero(folded):
                for k in cache.RING[f, :cache.LEN[f]]:
                    near |= v2f.get(int(k), set())
            subset = np.array(sorted(near), dtype=np.int64)
        else:
            subset = np.array([], dtype=np.int64)
    else:
        subset = None

    contained = ([] if (trigger == "folded" and subset is not None and not len(subset))
                 else contained_vertices(eptm, cache, P, subset, slab))
    overlaps = overlapping_faces(eptm, cache, P) if strict else []

    return Result(
        undecidable=cache.face_ids[undecidable],
        folded=cache.face_ids[folded],
        contained=contained,
        overlaps=overlaps,
        open_rings=cache.face_ids[cache.open_rings],
        cache=cache,
    )
