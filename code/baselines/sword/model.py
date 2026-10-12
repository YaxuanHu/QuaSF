"""
SWORD (SWift and Optimized Recognition of protein Domains)

Structure-based domain segmentation using spectral clustering on
the residue contact graph derived from Cα distance matrices.

Reference: Postic et al., Bioinformatics, 2017
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def extract_ca_coords(pdb_path):
    """Extract Cα coordinates and residue indices from a PDB file.

    Returns:
        coords: np.array [N, 3] Cα coordinates
        resi: np.array [N] residue indices
    """
    coords = []
    resi = []
    seen = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    res_seq = int(line[22:26].strip())
                    if res_seq not in seen:
                        seen.add(res_seq)
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                        resi.append(res_seq)

    if len(coords) == 0:
        return None, None

    return np.array(coords), np.array(resi)


def build_contact_map(coords, contact_threshold=8.0):
    """Build binary contact map from Cα coordinates.

    Args:
        coords: np.array [N, 3]
        contact_threshold: float, distance cutoff in Angstroms

    Returns:
        contact_map: np.array [N, N] binary adjacency matrix
        dist_matrix: np.array [N, N] distance matrix
    """
    dist_matrix = squareform(pdist(coords))
    contact_map = (dist_matrix < contact_threshold).astype(float)
    # Remove self-contacts on diagonal
    np.fill_diagonal(contact_map, 0)
    # Remove sequential neighbors (±1, ±2) to focus on long-range contacts
    N = len(coords)
    for i in range(N):
        for offset in [-2, -1, 1, 2]:
            j = i + offset
            if 0 <= j < N:
                contact_map[i, j] = 0
    return contact_map, dist_matrix


def estimate_num_domains(laplacian_eigenvalues, max_domains=10):
    """Estimate number of domains from eigenvalue gap.

    Uses the eigengap heuristic: find the largest gap between
    consecutive eigenvalues of the graph Laplacian.

    Args:
        laplacian_eigenvalues: sorted eigenvalues
        max_domains: maximum number of domains to consider

    Returns:
        n_domains: estimated number of domains (>= 1)
    """
    n = min(len(laplacian_eigenvalues), max_domains + 1)
    eigenvalues = laplacian_eigenvalues[:n]

    if len(eigenvalues) < 2:
        return 1

    gaps = np.diff(eigenvalues)
    # Skip the first eigenvalue (always ~0 for connected graph)
    if len(gaps) < 2:
        return 1

    # Find largest gap starting from index 1
    best_k = np.argmax(gaps[1:]) + 2  # +2 because we skip index 0 and argmax is 0-indexed

    return max(1, min(best_k, max_domains))


def spectral_clustering_domains(contact_map, n_domains=None, max_domains=10):
    """Perform spectral clustering on the contact graph.

    Args:
        contact_map: np.array [N, N] adjacency matrix
        n_domains: int or None. If None, auto-estimate.
        max_domains: int, max domains for auto-estimation

    Returns:
        labels: np.array [N] domain labels (1-indexed, 0 not used)
        n_domains: int, number of domains found
    """
    N = contact_map.shape[0]

    if N < 3:
        return np.ones(N, dtype=int), 1

    # Compute normalized Laplacian
    L = laplacian(contact_map, normed=True)

    # Compute eigenvalues and eigenvectors
    n_eig = min(max_domains + 1, N)
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, n_eig - 1])

    # Auto-estimate number of domains if not specified
    if n_domains is None:
        n_domains = estimate_num_domains(eigenvalues, max_domains)

    if n_domains <= 1:
        return np.ones(N, dtype=int), 1

    # Use first n_domains eigenvectors for clustering
    features = eigenvectors[:, :n_domains]

    # Normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    features = features / norms

    # K-means clustering
    kmeans = KMeans(n_clusters=n_domains, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    # Convert to 1-indexed
    labels = labels + 1

    return labels, n_domains


def enforce_contiguity(labels, min_segment_len=10):
    """Post-process labels to enforce spatial contiguity.

    Small isolated segments are merged into their neighbors.

    Args:
        labels: np.array [N] domain labels
        min_segment_len: minimum segment length

    Returns:
        cleaned_labels: np.array [N]
    """
    N = len(labels)
    if N == 0:
        return labels

    cleaned = labels.copy()

    # Find segments
    segments = []
    start = 0
    for i in range(1, N):
        if cleaned[i] != cleaned[i - 1]:
            segments.append((start, i - 1, cleaned[start]))
            start = i
    segments.append((start, N - 1, cleaned[start]))

    # Merge small segments into neighbors
    for seg_start, seg_end, seg_label in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len < min_segment_len:
            # Find nearest larger neighbor
            if seg_start > 0:
                left_label = cleaned[seg_start - 1]
            else:
                left_label = seg_label
            if seg_end < N - 1:
                right_label = cleaned[seg_end + 1]
            else:
                right_label = seg_label

            # Prefer the neighbor with more residues
            if left_label != seg_label:
                cleaned[seg_start:seg_end + 1] = left_label
            elif right_label != seg_label:
                cleaned[seg_start:seg_end + 1] = right_label

    return cleaned


def sword_predict(pdb_path, contact_threshold=8.0, n_domains=None,
                  max_domains=10, min_segment_len=10):
    """Run SWORD domain prediction on a single PDB file.

    Args:
        pdb_path: path to PDB file
        contact_threshold: distance cutoff for contacts (Angstroms)
        n_domains: number of domains (None = auto-detect)
        max_domains: max domains for auto-detection
        min_segment_len: minimum segment length for contiguity enforcement

    Returns:
        labels: np.array [N] domain labels (1-indexed)
        resi: np.array [N] residue indices
        n_domains: int
        or (None, None, 0) on failure
    """
    coords, resi = extract_ca_coords(pdb_path)
    if coords is None or len(coords) < 3:
        return None, None, 0

    contact_map, _ = build_contact_map(coords, contact_threshold)
    labels, n_dom = spectral_clustering_domains(
        contact_map, n_domains=n_domains, max_domains=max_domains
    )

    labels = enforce_contiguity(labels, min_segment_len)

    return labels, resi, n_dom
