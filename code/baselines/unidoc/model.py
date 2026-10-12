"""
UniDoc: A unified approach to protein domain parsing with inter-residue distance matrix.

Reference: Zhu et al., Bioinformatics, 2023 (btad070)

Algorithm:
1. Compute inter-residue Cb-Cb distance matrix from structure
2. Transform distances to contact probabilities via sigmoid
3. Top-down splitting: iteratively split fragments to minimize inter-domain interaction
   - Continuous split (single cut) and discontinuous split (two cuts)
   - Accept split only if DIS_inter < 0.5 * DIS_intra
   - Constraints: fragment size > 30, cut not in regular secondary structure
4. Bottom-up merging: merge fragments with strong inter-domain interaction
   - Merge pair (I,J) if S(I,J) = DIS_inter(I,J) - min(DIS_intra(I), DIS_intra(J)) > 0
   - Post-processing: merge fragments with DIS_intra < 1.0
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# PDB parsing utilities
# ---------------------------------------------------------------------------

def extract_cb_coords(pdb_path):
    """Extract Cb coordinates (Ca for Glycine) and residue indices from PDB.

    Returns:
        coords: np.array [N, 3]
        resi: np.array [N] residue sequence numbers
    """
    ca_coords = {}
    cb_coords = {}
    res_names = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            atom_name = line[12:16].strip()
            res_seq = int(line[22:26].strip())
            res_name = line[17:20].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            if atom_name == 'CA':
                ca_coords[res_seq] = np.array([x, y, z])
                res_names[res_seq] = res_name
            elif atom_name == 'CB':
                cb_coords[res_seq] = np.array([x, y, z])

    if len(ca_coords) == 0:
        return None, None

    resi_sorted = sorted(ca_coords.keys())
    coords = []
    resi = []
    for r in resi_sorted:
        # Use Cb for non-Glycine, Ca for Glycine
        if r in cb_coords:
            coords.append(cb_coords[r])
        else:
            coords.append(ca_coords[r])
        resi.append(r)

    return np.array(coords), np.array(resi)


def assign_secondary_structure(pdb_path, n_residues, resi):
    """Parse HELIX/SHEET records from PDB to get secondary structure assignment.

    Returns:
        ss: np.array [N] of chars, 'H'=helix, 'E'=sheet, 'C'=coil
    """
    resi_set = set(resi.tolist())
    resi_to_idx = {r: i for i, r in enumerate(resi)}

    ss = np.array(['C'] * n_residues)

    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('HELIX'):
                    start = int(line[21:25].strip())
                    end = int(line[33:37].strip())
                    for r in range(start, end + 1):
                        if r in resi_to_idx:
                            ss[resi_to_idx[r]] = 'H'
                elif line.startswith('SHEET'):
                    start = int(line[22:26].strip())
                    end = int(line[33:37].strip())
                    for r in range(start, end + 1):
                        if r in resi_to_idx:
                            ss[resi_to_idx[r]] = 'E'
    except Exception:
        pass

    # If no HELIX/SHEET records found, use simple distance-based heuristic
    if np.all(ss == 'C'):
        ss = _estimate_ss_from_distances(pdb_path, n_residues, resi)

    return ss


def _estimate_ss_from_distances(pdb_path, n_residues, resi):
    """Fallback: estimate secondary structure from Ca-Ca distances.

    Helix: Ca(i)-Ca(i+3) ~ 5.5 Å
    Sheet: Ca(i)-Ca(i+2) ~ 6.7 Å, extended conformation
    """
    ca_coords = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    res_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ca_coords[res_seq] = np.array([x, y, z])

    resi_to_idx = {r: i for i, r in enumerate(resi)}
    ss = np.array(['C'] * n_residues)

    resi_list = list(resi)
    for idx in range(len(resi_list) - 3):
        r0, r1, r2, r3 = resi_list[idx], resi_list[idx+1], resi_list[idx+2], resi_list[idx+3]
        if r0 in ca_coords and r3 in ca_coords:
            d = np.linalg.norm(ca_coords[r0] - ca_coords[r3])
            if 4.5 < d < 6.5:  # helix-like
                for r in [r0, r1, r2, r3]:
                    if r in resi_to_idx:
                        ss[resi_to_idx[r]] = 'H'

    return ss


# ---------------------------------------------------------------------------
# Contact probability and interaction scores (Equations 1-3 in paper)
# ---------------------------------------------------------------------------

def distance_to_contact_prob(dist_matrix, d0=8.0, d_scale=1.5):
    """Transform distance matrix to contact probability matrix (Eq. 1).

    p_ij = 1 / (1 + exp((d_ij - d0) / d_scale))
    """
    return 1.0 / (1.0 + np.exp((dist_matrix - d0) / d_scale))


def compute_dis_inter(prob_matrix, indices1, indices2, a=0.43):
    """Compute inter-domain interaction score (Eq. 2).

    DIS_inter(D1, D2) = (1 / (l1^a * l2^a)) * sum_{i in D1, j in D2} p_ij
    """
    l1 = len(indices1)
    l2 = len(indices2)
    if l1 == 0 or l2 == 0:
        return 0.0
    sub = prob_matrix[np.ix_(indices1, indices2)]
    return sub.sum() / (l1 ** a * l2 ** a)


def compute_dis_intra(prob_matrix, indices, b=0.95):
    """Compute intra-domain interaction score (Eq. 3).

    DIS_intra(D) = (1 / l^b) * sum_{i,j in D, |i-j|>2} p_ij
    """
    l = len(indices)
    if l <= 2:
        return 0.0
    sub = prob_matrix[np.ix_(indices, indices)]
    total = 0.0
    for ii in range(l):
        for jj in range(l):
            if abs(indices[ii] - indices[jj]) > 2:
                total += sub[ii, jj]
    return total / (l ** b)


def compute_dis_intra_fast(prob_matrix, indices, b=0.95):
    """Fast vectorized version of compute_dis_intra."""
    l = len(indices)
    if l <= 2:
        return 0.0
    idx_arr = np.array(indices)
    sub = prob_matrix[np.ix_(indices, indices)]
    # Build mask for |i-j| > 2 (using original residue index positions)
    diff = np.abs(idx_arr[:, None] - idx_arr[None, :])
    mask = diff > 2
    total = (sub * mask).sum()
    return total / (l ** b)


# ---------------------------------------------------------------------------
# Top-down splitting (Section 2.4)
# ---------------------------------------------------------------------------

def _is_in_secondary_structure(pos, ss):
    """Check if position is in regular secondary structure (helix or sheet)."""
    if 0 <= pos < len(ss):
        return ss[pos] in ('H', 'E')
    return False


def _valid_cut_position(pos, ss, indices):
    """Check if a cut position is valid: not in secondary structure."""
    # pos is an index within the fragment's indices list
    actual_pos = indices[pos]
    return not _is_in_secondary_structure(actual_pos, ss)


def continuous_split(prob_matrix, dist_matrix, indices, ss, min_frag_size=30):
    """Find optimal continuous split of a fragment (Eq. 4).

    Split fragment D into D1=[0..k-1] and D2=[k..l-1].

    Returns:
        best_k: optimal cut position (index within fragment), or -1 if no valid split
        best_score: DIS_inter at best cut
    """
    l = len(indices)
    best_k = -1
    best_score = float('inf')

    for k in range(min_frag_size, l - min_frag_size + 1):
        # Check secondary structure constraint
        if _is_in_secondary_structure(indices[k], ss):
            continue

        d1 = indices[:k]
        d2 = indices[k:]
        score = compute_dis_inter(prob_matrix, d1, d2)
        if score < best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def discontinuous_split(prob_matrix, dist_matrix, indices, ss, min_frag_size=30):
    """Find optimal discontinuous split of a fragment (Eq. 5).

    Split fragment D into:
      D1 (discontinuous) = indices[0:t] + indices[s:l]
      D2 (continuous) = indices[t:s]

    Constraints:
      - s > t + 35 (continuous fragment not too small)
      - d(indices[t], indices[s]) < 8 Å (cut points spatially close)

    Returns:
        best_t, best_s: optimal cut positions, or (-1, -1) if no valid split
        best_score: DIS_inter at best cut
    """
    l = len(indices)
    best_t = -1
    best_s = -1
    best_score = float('inf')

    for t in range(min_frag_size, l - min_frag_size):
        # Check secondary structure constraint at t
        if _is_in_secondary_structure(indices[t], ss):
            continue

        for s in range(t + 35, l - min_frag_size + 1):
            if s >= l:
                break

            # Check secondary structure constraint at s
            if _is_in_secondary_structure(indices[s], ss):
                continue

            # Check spatial proximity constraint: d(t, s) < 8 Å
            if dist_matrix[indices[t], indices[s]] >= 8.0:
                continue

            # D1 (discontinuous): [0:t] + [s:l]
            d1 = list(indices[:t]) + list(indices[s:])
            # D2 (continuous): [t:s]
            d2 = list(indices[t:s])

            if len(d1) < min_frag_size or len(d2) < min_frag_size:
                continue

            score = compute_dis_inter(prob_matrix, d1, d2)
            if score < best_score:
                best_score = score
                best_t = t
                best_s = s

    return best_t, best_s, best_score


def top_down_split(prob_matrix, dist_matrix, ss, min_frag_size=30):
    """Top-down algorithm to iteratively split protein into fragments.

    At each step, select the fragment with the best split (lowest DIS_inter)
    and accept it only if DIS_inter < 0.5 * DIS_intra.

    Returns:
        fragments: list of lists of indices (each fragment is a list of residue indices)
    """
    N = prob_matrix.shape[0]
    # Start with the whole protein as a single fragment
    fragments = [list(range(N))]

    changed = True
    while changed:
        changed = False
        best_frag_idx = -1
        best_split_type = None  # 'continuous' or 'discontinuous'
        best_split_score = float('inf')
        best_split_params = None

        for fi, frag in enumerate(fragments):
            if len(frag) < 2 * min_frag_size:
                continue

            dis_intra = compute_dis_intra_fast(prob_matrix, frag)

            # Try continuous split
            k, c_score = continuous_split(prob_matrix, dist_matrix, frag, ss, min_frag_size)
            if k >= 0 and c_score < best_split_score:
                # Check acceptance: DIS_inter < 0.5 * DIS_intra
                if c_score < 0.5 * dis_intra:
                    best_split_score = c_score
                    best_frag_idx = fi
                    best_split_type = 'continuous'
                    best_split_params = (k,)

            # Try discontinuous split
            t, s, d_score = discontinuous_split(prob_matrix, dist_matrix, frag, ss, min_frag_size)
            if t >= 0 and d_score < best_split_score:
                if d_score < 0.5 * dis_intra:
                    best_split_score = d_score
                    best_frag_idx = fi
                    best_split_type = 'discontinuous'
                    best_split_params = (t, s)

        if best_frag_idx >= 0:
            frag = fragments[best_frag_idx]
            if best_split_type == 'continuous':
                k = best_split_params[0]
                f1 = frag[:k]
                f2 = frag[k:]
                fragments[best_frag_idx] = f1
                fragments.insert(best_frag_idx + 1, f2)
            elif best_split_type == 'discontinuous':
                t, s = best_split_params
                # D1 (discontinuous): [0:t] + [s:l]
                f1 = frag[:t] + frag[s:]
                # D2 (continuous): [t:s]
                f2 = frag[t:s]
                fragments[best_frag_idx] = f1
                fragments.insert(best_frag_idx + 1, f2)
            changed = True

    return fragments


# ---------------------------------------------------------------------------
# Bottom-up merging (Section 2.5)
# ---------------------------------------------------------------------------

def bottom_up_merge(prob_matrix, fragments):
    """Bottom-up algorithm to merge fragments with strong inter-domain interaction.

    Iteratively merge pair (I, J) that maximizes:
      S(i,j) = DIS_inter(Di, Dj) - min(DIS_intra(Di), DIS_intra(Dj))
    Merge if S(I, J) > 0.

    Then post-process: merge fragments with DIS_intra < 1.0.

    Returns:
        merged_fragments: list of lists of indices
    """
    frags = [list(f) for f in fragments]

    # Iterative merging
    changed = True
    while changed and len(frags) > 1:
        changed = False
        best_score = -float('inf')
        best_i = -1
        best_j = -1

        n = len(frags)
        for i in range(n):
            intra_i = compute_dis_intra_fast(prob_matrix, frags[i])
            for j in range(i + 1, n):
                intra_j = compute_dis_intra_fast(prob_matrix, frags[j])
                inter_ij = compute_dis_inter(prob_matrix, frags[i], frags[j])
                s = inter_ij - min(intra_i, intra_j)
                if s > best_score:
                    best_score = s
                    best_i = i
                    best_j = j

        if best_score > 0:
            # Merge fragments best_i and best_j
            merged = frags[best_i] + frags[best_j]
            frags[best_i] = merged
            frags.pop(best_j)
            changed = True

    # Post-processing: merge fragments with DIS_intra < 1.0
    changed = True
    while changed and len(frags) > 1:
        changed = False
        weak_idx = -1
        for i, frag in enumerate(frags):
            intra = compute_dis_intra_fast(prob_matrix, frag)
            if intra < 1.0:
                weak_idx = i
                break

        if weak_idx >= 0:
            # Find the fragment with strongest interaction
            best_partner = -1
            best_inter = -float('inf')
            for j in range(len(frags)):
                if j == weak_idx:
                    continue
                inter = compute_dis_inter(prob_matrix, frags[weak_idx], frags[j])
                if inter > best_inter:
                    best_inter = inter
                    best_partner = j

            if best_partner >= 0:
                merged = frags[weak_idx] + frags[best_partner]
                # Remove the higher index first
                if weak_idx > best_partner:
                    frags.pop(weak_idx)
                    frags[best_partner] = merged
                else:
                    frags.pop(best_partner)
                    frags[weak_idx] = merged
                changed = True

    return frags


# ---------------------------------------------------------------------------
# Main UniDoc prediction function
# ---------------------------------------------------------------------------

def unidoc_predict(pdb_path, min_frag_size=30, d0=8.0, d_scale=1.5):
    """Run UniDoc domain parsing on a single PDB file.

    Args:
        pdb_path: path to PDB file
        min_frag_size: minimum fragment size (paper uses 30)
        d0: distance threshold for contact probability (paper uses 8.0)
        d_scale: scale parameter for contact probability (paper uses 1.5)

    Returns:
        labels: np.array [N] domain labels (1-indexed)
        resi: np.array [N] residue indices
        n_domains: int number of domains
        or (None, None, 0) on failure
    """
    # Step 1: Extract Cb coordinates
    coords, resi = extract_cb_coords(pdb_path)
    if coords is None or len(coords) < min_frag_size:
        return None, None, 0

    N = len(coords)

    # Step 2: Compute distance matrix and contact probability
    dist_matrix = squareform(pdist(coords))
    prob_matrix = distance_to_contact_prob(dist_matrix, d0=d0, d_scale=d_scale)

    # Step 3: Get secondary structure
    ss = assign_secondary_structure(pdb_path, N, resi)

    # Step 4: Top-down splitting
    fragments = top_down_split(prob_matrix, dist_matrix, ss, min_frag_size=min_frag_size)

    # Step 5: Bottom-up merging
    fragments = bottom_up_merge(prob_matrix, fragments)

    # Step 6: Convert fragments to per-residue labels
    labels = np.zeros(N, dtype=np.int64)
    for dom_id, frag in enumerate(fragments, start=1):
        for idx in frag:
            labels[idx] = dom_id

    # Handle any unassigned residues (shouldn't happen, but safety)
    if np.any(labels == 0):
        labels[labels == 0] = 1

    n_domains = len(fragments)
    return labels, resi, n_domains
