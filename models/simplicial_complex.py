import numpy as np
from itertools import combinations
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist, squareform


class SimplicialComplex:
    """Vietoris-Rips simplicial complex with properly oriented boundary matrices.

    Constructs a simplicial complex from a point cloud by adding all simplices
    whose vertices are pairwise within a distance threshold. Boundary matrices
    use standard orientation (±1 entries) so that d^2 = 0.

    Parameters
    ----------
    points : array-like, shape (N, m)
        Point coordinates (typically Diffusion Maps embedding).
    max_dim : int
        Maximum simplex dimension to construct. 1=edges only, 2=up to triangles,
        3=up to tetrahedra.
    threshold : float or None
        Distance threshold for the Rips complex. If None, uses the median of
        pairwise distances.
    max_simplices : dict or None
        Optional cap on the number of simplices at each dimension,
        e.g. {2: 1000, 3: 250}. If exceeded, simplices are randomly subsampled.
    """

    def __init__(self, points, max_dim=2, threshold=None, max_simplices=None):
        if hasattr(points, "numpy"):
            points = points.detach().cpu().numpy()
        self.points = np.asarray(points, dtype=np.float64)
        self.max_dim = max_dim
        self.max_simplices = max_simplices or {}
        self.N = self.points.shape[0]

        # Pairwise distances
        self._dist_matrix = squareform(pdist(self.points))

        if threshold is None:
            flat = self._dist_matrix[self._dist_matrix > 0]
            threshold = np.percentile(flat, 10)
        self.threshold = threshold

        # Build the complex
        self._simplices = {}  # dim -> list of sorted tuples
        self._simplex_to_idx = {}  # dim -> dict(tuple -> index)
        self._boundary_matrices = {}  # k -> sparse matrix B_k
        self._build_rips()

    def _build_rips(self):
        """Build the Vietoris-Rips complex up to max_dim."""
        # 0-simplices (vertices)
        self._simplices[0] = [(i,) for i in range(self.N)]
        self._simplex_to_idx[0] = {(i,): i for i in range(self.N)}

        # 1-simplices (edges): pairs within threshold
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self._dist_matrix[i, j] <= self.threshold:
                    edges.append((i, j))
        edges = self._subsample(edges, 1)
        self._simplices[1] = edges
        self._simplex_to_idx[1] = {e: idx for idx, e in enumerate(edges)}

        if self.max_dim < 2:
            return

        # Build neighbor sets for efficient clique finding
        neighbors = [set() for _ in range(self.N)]
        for i, j in edges:
            neighbors[i].add(j)
            neighbors[j].add(i)

        # 2-simplices (triangles)
        triangles = []
        for i, j in edges:
            common = neighbors[i].intersection(neighbors[j])
            for k in common:
                if k > j:
                    triangles.append(tuple(sorted((i, j, k))))
        triangles = list(set(triangles))  # deduplicate
        triangles.sort()
        triangles = self._subsample(triangles, 2)
        self._simplices[2] = triangles
        self._simplex_to_idx[2] = {t: idx for idx, t in enumerate(triangles)}

        if self.max_dim < 3:
            return

        # 3-simplices (tetrahedra)
        tetrahedra = []
        for tri in triangles:
            i, j, k = tri
            common = neighbors[i].intersection(neighbors[j]).intersection(neighbors[k])
            for l in common:
                if l > k:
                    tetrahedra.append(tuple(sorted((i, j, k, l))))
        tetrahedra = list(set(tetrahedra))
        tetrahedra.sort()
        tetrahedra = self._subsample(tetrahedra, 3)
        self._simplices[3] = tetrahedra
        self._simplex_to_idx[3] = {t: idx for idx, t in enumerate(tetrahedra)}

    def _subsample(self, simplices, dim):
        """Subsample simplices if exceeding max_simplices cap.

        For dim >= 2, keeps the simplices with the smallest max edge length
        (earliest in the Rips filtration = most important topologically).
        For dim 1 (edges), falls back to random subsampling.
        """
        cap = self.max_simplices.get(dim)
        if cap is None or len(simplices) <= cap:
            return simplices

        if dim >= 2:
            # Sort by filtration value: max pairwise edge length among vertices
            def filtration_value(simplex):
                verts = list(simplex)
                return max(
                    self._dist_matrix[verts[a], verts[b]]
                    for a in range(len(verts))
                    for b in range(a + 1, len(verts))
                )
            simplices = sorted(simplices, key=filtration_value)[:cap]
        else:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(simplices), size=cap, replace=False)
            indices.sort()
            simplices = [simplices[i] for i in indices]
        return simplices

    def num_simplices(self, k):
        """Number of k-simplices."""
        return len(self._simplices.get(k, []))

    @property
    def simplices(self):
        """Dict mapping dimension k -> list of sorted vertex-index tuples."""
        return self._simplices

    def boundary_matrix(self, k):
        """Compute the oriented boundary matrix B_k.

        B_k maps k-chains to (k-1)-chains. It has shape
        (n_{k-1}-simplices, n_k-simplices) with entries ±1.

        Orientation convention: for a k-simplex [v_0, v_1, ..., v_k] (sorted),
        the i-th face (obtained by removing v_i) gets sign (-1)^i.

        B_k @ B_{k+1} = 0 (the fundamental property of boundary operators).

        Returns
        -------
        B_k : scipy.sparse.csr_matrix
            The oriented boundary matrix.
        """
        if k in self._boundary_matrices:
            return self._boundary_matrices[k]

        if k < 1 or k > self.max_dim:
            return None

        k_simplices = self._simplices.get(k, [])
        km1_simplices_idx = self._simplex_to_idx.get(k - 1, {})

        n_rows = len(km1_simplices_idx)
        n_cols = len(k_simplices)

        if n_rows == 0 or n_cols == 0:
            B = csr_matrix((n_rows, n_cols))
            self._boundary_matrices[k] = B
            return B

        rows = []
        cols = []
        data = []

        for col_idx, simplex in enumerate(k_simplices):
            # Each face is obtained by removing one vertex
            for face_idx in range(len(simplex)):
                face = simplex[:face_idx] + simplex[face_idx + 1 :]
                row_idx = km1_simplices_idx.get(face)
                if row_idx is not None:
                    sign = (-1) ** face_idx
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(sign)

        B = coo_matrix(
            (data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float64
        ).tocsr()
        self._boundary_matrices[k] = B
        return B

    def verify_boundary_squared_zero(self, k):
        """Verify that B_k @ B_{k+1} = 0 (up to numerical tolerance).

        Returns True if the property holds, False otherwise.
        """
        B_k = self.boundary_matrix(k)
        B_kp1 = self.boundary_matrix(k + 1)
        if B_k is None or B_kp1 is None:
            return True
        product = B_k @ B_kp1
        return np.abs(product.toarray()).max() < 1e-12
