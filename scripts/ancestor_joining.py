import json
import sys
import numpy as np
import pandas as pd
import networkx as nx
from networkx.utils import UnionFind
import heapq
from serial_id_generator import SerialIDGenerator
import time

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------

directory = sys.argv[1]  # folder containing all input / output files

# ---- Inputs ------------------------------------------------------------------
similarity_file   = f"{directory}/similarity_matrix.csv"
editdistance_file = f"{directory}/editdistance_matrix.csv"
mutation_json     = f"{directory}/mutation.json"
daydiff_json      = f"{directory}/datetime.json"

# ---- Outputs -----------------------------------------------------------------
adj_matrix_file   = f"{directory}/adj_matrix.npy"
updated_json      = f"{directory}/updated_mutation.json"
updated_edit      = f"{directory}/updated_editdistance_matrix.npy"
updated_sim       = f"{directory}/updated_similarity_matrix.npy"

# -----------------------------------------------------------------------------
# Helper loaders ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_mutations(path):
    with open(path) as fh:
        raw = json.load(fh)
    mut_dict, mutset2id = {}, {}
    for node_id, lst in raw.items():
        fs = frozenset(tuple(x) for x in lst)
        mut_dict[node_id] = fs
        mutset2id[fs] = node_id
    return mut_dict, mutset2id


def load_daydiff(path):
    with open(path) as fh:
        raw = json.load(fh)
    return {k: (int(v) if v is not None else None) for k, v in raw.items()}

# -----------------------------------------------------------------------------
# Core graph operations --------------------------------------------------------
# -----------------------------------------------------------------------------

def new_inferred(gen):
    return gen.generate()


def connect_direct(G, src, dst, d_src, d_dst):
    if d_src is not None and d_dst is not None and d_src < d_dst:
        G.add_edge(src, dst)
    else:
        G.add_edge(dst, src)


def find_relation(G, u, v, mut_dict, rev_map, day_dict, gen):
    """Return list[new inferred IDs]."""
    new_ids = []
    mu, mv = mut_dict[u], mut_dict[v]
    if mu <= mv or mv <= mu:
        connect_direct(G, u, v, day_dict.get(u), day_dict.get(v))
        return new_ids
    intersect = mu & mv
    pid = rev_map.get(intersect)
    if pid is None:
        pid = new_inferred(gen)
        G.add_node(pid, inferred=True)
        mut_dict[pid] = intersect
        rev_map[intersect] = pid
        new_ids.append(pid)
    G.add_edge(pid, u)
    G.add_edge(pid, v)
    return new_ids

# -----------------------------------------------------------------------------
# Neighbour selection ----------------------------------------------------------
# -----------------------------------------------------------------------------

def dense_group(G, E, S, hdr, mut_dict, rev, day, gen):
    N = len(hdr)
    E2 = E.astype(float).copy(); np.fill_diagonal(E2, np.inf)
    rmin = E2.min(1, keepdims=True)
    mask = (E2 == rmin)
    best = np.where(mask, S, -1).max(1)
    tie = mask & (S == best[:, None])
    new_ids = []
    for i in range(N):
        cols = np.flatnonzero(tie[i])
        if cols.size == 0:
            continue
        base = hdr[i]
        neigh = [hdr[c] for c in cols]
        common = mut_dict[base]
        for n in neigh:
            common &= mut_dict[n]
            if not common:
                break
        new_ids += connect_group(G, base, neigh, common, mut_dict, rev, gen)
    return new_ids


def knn_group(G, E, S, hdr, mut_dict, rev, day, gen, k=5):
    kth = np.partition(S, -k, 1)[:, -k]
    mask = S >= kth[:, None]
    np.fill_diagonal(mask, False)
    empty = ~mask.any(1)
    if empty.any():
        tmp = E.astype(float).copy(); np.fill_diagonal(tmp, np.inf)
        rmin = tmp.min(1, keepdims=True)
        mmax = S.max(1, keepdims=True)
        mask[empty] |= (E == rmin) & (S == mmax)
    INF = np.iinfo(E.dtype).max
    Em = np.where(mask, E, INF)
    rmin = Em.min(1, keepdims=True)
    cand = Em == rmin
    best = np.where(cand, S, -1).max(1)
    tie = cand & (S == best[:, None])
    new_ids = []
    for i in range(len(hdr)):
        cols = np.flatnonzero(tie[i]);
        if cols.size == 0:
            continue
        base = hdr[i]
        neigh = [hdr[c] for c in cols]
        common = mut_dict[base]
        for n in neigh:
            common &= mut_dict[n]
            if not common:
                break
        new_ids += connect_group(G, base, neigh, common, mut_dict, rev, gen)
    return new_ids


def connect_group(G, base, neigh, common, mut_dict, rev_map, gen):
    new_ids = []
    if not common:
        return new_ids
    grp = [base] + neigh
    parent = next((n for n in grp if mut_dict[n] == common), None)
    if parent is None:
        parent = new_inferred(gen)
        G.add_node(parent, inferred=True)
        mut_dict[parent] = common
        rev_map[common] = parent
        new_ids.append(parent)
    for n in grp:
        if n != parent:
            G.add_edge(parent, n)
    return new_ids

# -----------------------------------------------------------------------------
# Matrix expansion ------------------------------------------------------------
# -----------------------------------------------------------------------------

def expand(new_ids, mut_dict, E, S, hdr, idx_map):
    if not new_ids:
        return E, S, hdr, idx_map
    Nold = E.shape[0]
    Nnew = Nold + len(new_ids)
    dtype = E.dtype
    e = np.zeros((Nnew, Nnew), dtype=dtype)
    s = np.zeros_like(e)
    e[:Nold, :Nold], s[:Nold, :Nold] = E, S
    for off, nid in enumerate(new_ids):
        idx = Nold + off
        hdr.append(nid); idx_map[nid] = idx
        mnew = mut_dict[nid]
        for j, eid in enumerate(hdr[:idx]):
            dist = len(mnew ^ mut_dict[eid])
            sim = len(mnew & mut_dict[eid])
            e[idx, j] = e[j, idx] = dist
            s[idx, j] = s[j, idx] = sim
    return e, s, hdr, idx_map

# -----------------------------------------------------------------------------
# Bridging with union‑find + heap ---------------------------------------------
# -----------------------------------------------------------------------------

def bridge_components(G, E, S, hdr, idx_map, mut_dict, rev, day, gen):
    """Connect components using a global max‑heap (one edge per merge)."""
    N = S.shape[0]
    uf = UnionFind(range(N))
    components = N  # one per node initially

    # build max‑heap of all off‑diagonal similarities once
    heap = [(-int(S[i, j]), i, j)
            for i in range(N - 1)
            for j in range(i + 1, N)
            if S[i, j] > 0]
    heapq.heapify(heap)

    new_ids_all = []
    while components > 1 and heap:
        _, i, j = heapq.heappop(heap)
        if uf[i] == uf[j]:
            continue  # already in same component
        u, v = hdr[i], hdr[j]
        new_ids = find_relation(G, u, v, mut_dict, rev, day, gen)
        new_ids_all.extend(new_ids)
        uf.union(i, j)
        components -= 1
    return new_ids_all

# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    start_time = time.perf_counter()
    # ------------------ load data -----------------------------------
    mut_dict, rev = load_mutations(mutation_json)
    day          = load_daydiff(daydiff_json)
    hdr          = list(mut_dict.keys())
    idx_map      = {h: i for i, h in enumerate(hdr)}

    G = nx.DiGraph(); G.add_nodes_from(hdr)

    S = pd.read_csv(similarity_file, index_col=0).to_numpy()
    E = pd.read_csv(editdistance_file, index_col=0).to_numpy()

    gen = SerialIDGenerator()

    # ------------------ Phase A: local pass --------------------------
    if len(hdr) < 500:
        new_ids = dense_group(G, E, S, hdr, mut_dict, rev, day, gen)
    else:
        new_ids = knn_group(G, E, S, hdr, mut_dict, rev, day, gen, k=5)
    
    E, S, hdr, idx_map = expand(new_ids, mut_dict, E, S, hdr, idx_map)
    
    print ("-----------------------Phase A--------------------------------")
    print("Total components :", nx.number_weakly_connected_components(G))
    print("Total nodes            :", len(G))
    print("Total edges            :", G.number_of_edges())

    # ------------------ Phase B: component bridging ------------------
    new_ids = bridge_components(G, E, S, hdr, idx_map, mut_dict, rev, day, gen)
    E, S, hdr, idx_map = expand(new_ids, mut_dict, E, S, hdr, idx_map)

    # ------------------ outputs -------------------------------------
    with open(updated_json, "w") as fh:
        json.dump({k: sorted(list(v)) for k, v in mut_dict.items()}, fh)

    np.save(updated_edit, E)
    np.save(updated_sim , S)

    adj = nx.to_numpy_array(G, nodelist=hdr, dtype=int)
    np.save(adj_matrix_file, adj)
    print ("-----------------------Phase B--------------------------------")
    print("Graph weakly connected :", nx.is_weakly_connected(G))
    print("Total nodes            :", len(G))
    print("Total edges            :", G.number_of_edges())
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print (f"Elapsed time: {elapsed_time:.4f}seconds")


if __name__ == "__main__":
    main()



'''
import json
import sys
import numpy as np
import pandas as pd
import networkx as nx
from serial_id_generator import SerialIDGenerator
import time

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------

directory = sys.argv[1]  # folder containing all input / output files

# ---- Inputs ------------------------------------------------------------------
similarity_file   = f"{directory}/similarity_matrix.csv"
editdistance_file = f"{directory}/editdistance_matrix.csv"
mutation_json     = f"{directory}/mutation.json"          # {id: [[pos, base], ...]}
daydiff_json      = f"{directory}/datetime.json"          # {id: day_difference}

# ---- Outputs -----------------------------------------------------------------
adj_matrix_file   = f"{directory}/adj_matrix.npy"
updated_json      = f"{directory}/updated_mutation.json"
updated_edit      = f"{directory}/updated_editdistance_matrix.npy"
updated_sim       = f"{directory}/updated_similarity_matrix.npy"

# -----------------------------------------------------------------------------
# Helper loaders ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_mutations(path):
    """Load mutation lists and convert to frozensets of (pos, base) tuples."""
    with open(path) as fh:
        raw = json.load(fh)
    mut_dict, mutset2id = {}, {}
    for node_id, lst in raw.items():
        fset = frozenset(tuple(item) for item in lst)
        mut_dict[node_id] = fset
        mutset2id[fset]   = node_id
    return mut_dict, mutset2id


def load_daydiff(path):
    with open(path) as fh:
        raw = json.load(fh)
    return {k: (int(v) if v is not None else None) for k, v in raw.items()}

# -----------------------------------------------------------------------------
# Domain utilities -------------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_hypothetical_node(var1_fset, var2_fset, new_node_gen):
    """Wrapper around the user's SerialIDGenerator."""
    return new_node_gen.generate()


def find_relation(G, v1, v2, mutation_dict, mutset2id, daydiff_dict, new_node_gen):
    """Add edge or hypothetical node; return list of newly created IDs."""
    new_ids = []
    m1, m2 = mutation_dict[v1], mutation_dict[v2]
    d1, d2 = daydiff_dict.get(v1), daydiff_dict.get(v2)

    # direct‑ancestor logic ---------------------------------------------
    if m1 <= m2:
        src, dst = (v2, v1) if d2 is not None and d1 is not None and d2 < d1 else (v1, v2)
        G.add_edge(src, dst)
        return new_ids
    if m2 <= m1:
        src, dst = (v1, v2) if d1 is not None and d2 is not None and d1 < d2 else (v2, v1)
        G.add_edge(src, dst)
        return new_ids

    # divergent: need ancestor ------------------------------------------
    hyp_mut = m1 & m2                      # intersection
    hyp_id  = mutset2id.get(hyp_mut)
    if hyp_id is None:
        hyp_id = generate_hypothetical_node(m1, m2, new_node_gen)
        G.add_node(hyp_id, is_hypo=True)
        mutation_dict[hyp_id] = hyp_mut
        mutset2id[hyp_mut]   = hyp_id
        new_ids.append(hyp_id)

    G.add_edge(hyp_id, v1)
    G.add_edge(hyp_id, v2)
    return new_ids

# -----------------------------------------------------------------------------
# Exhaustive neighbour selection (grouped) ------------------------------------
# -----------------------------------------------------------------------------
def add_best_edges_dense_group(
    G,
    edit_mat,
    sim_mat,
    headers,
    mutation_dict,
    mutset2id,
    daydiff_dict,
    new_node_gen,
):
    """Full scan, group neighbours sharing optimum criteria (exclude self).

    * Excludes self‐comparison when finding minima.
    * Groups all columns that tie on min‑distance & max‑similarity.
    * Computes intersection across (base + all neighbours).
    """
    N = sim_mat.shape[0]

    # --- ignore self in distance search --------------------------------
    edit_no_diag = edit_mat.copy().astype(float)
    np.fill_diagonal(edit_no_diag, np.inf)

    row_min = edit_no_diag.min(axis=1, keepdims=True)        # (N,1) min non‑self
    cand    = edit_no_diag == row_min                        # minima mask (no self)
    best_sim = np.where(cand, sim_mat, -1).max(axis=1)
    tie_mask = cand & (sim_mat == best_sim[:, None])

    hdr = np.asarray(headers)
    new_total = []
    for i in range(N):
        cols = np.flatnonzero(tie_mask[i])
        if cols.size == 0:
            continue
        base  = hdr[i]
        neigh = [hdr[c] for c in cols]

        # intersection across base + neighbours
        common = mutation_dict[base]
        for nid in neigh:
            common &= mutation_dict[nid]
            if not common:
                break

        new_total += find_relation_group(G, base, neigh, common,
                                         mutation_dict, mutset2id,
                                         new_node_gen)
    return new_total
        
def add_best_edges_kNN(G, edit_mat, sim_mat, headers,
                       mutation_dict, mutset2id, daydiff_dict, gen, k=5):
    """k‑NN mask *but* group ties exactly like dense version.

    Steps:
      1. keep k highest‑similarity columns per row (mask).
      2. within the mask, take min edit distance → max similarity tie.
      3. gather *all* neighbours that tie for that score per row.
      4. compute the intersection across (base + neighbours).
      5. call find_relation_group once for the whole set.
    """
    N = sim_mat.shape[0]

    # --- k‑NN similarity mask -----------------------------------------
    kth = np.partition(sim_mat, -k, axis=1)[:, -k]
    mask = sim_mat >= kth[:, None]
    np.fill_diagonal(mask, False)

    # --- ensure at least one candidate per row -----------------------
    empty = ~mask.any(axis=1)
    if empty.any():
        INF = np.iinfo(edit_mat.dtype).max
        tmp = edit_mat.astype(float).copy()
        np.fill_diagonal(tmp, INF)
        row_min = tmp.min(axis=1, keepdims=True)
        max_sim_row = sim_mat.max(axis=1, keepdims=True)
        mask[empty] |= (edit_mat == row_min) & (sim_mat == max_sim_row)

    # --- choose neighbours per row -----------------------------------
    INF = np.iinfo(edit_mat.dtype).max
    edit_m = np.where(mask, edit_mat, INF)
    row_min = edit_m.min(axis=1, keepdims=True)
    cand    = edit_m == row_min
    best_sim = np.where(cand, sim_mat, -1).max(axis=1)
    tie_mask = cand & (sim_mat == best_sim[:, None])

    hdr = np.asarray(headers)
    new_total = []

    for i in range(N):
        cols = np.flatnonzero(tie_mask[i])
        if cols.size == 0:
            continue
        base  = hdr[i]
        neigh = [hdr[c] for c in cols]

        common = mutation_dict[base]
        for nid in neigh:
            common &= mutation_dict[nid]
            if not common:
                break

        new_total += find_relation_group(G, base, neigh, common,
                                         mutation_dict, mutset2id, gen)

    return new_total

def find_relation_group(G, base_id, neighbour_ids, common_mut_set, mutation_dict, mutset2id, new_node_gen):
    """Connect group via common ancestor (or existing node)."""
    new_ids = []
    if not common_mut_set:
        return new_ids  # nothing to do

    group = [base_id] + neighbour_ids
    parent = None
    for n in group:
        if mutation_dict[n] == common_mut_set:
            parent = n
            break
    if parent is None:
        parent = generate_hypothetical_node(None, None, new_node_gen)
        G.add_node(parent)
        mutation_dict[parent] = common_mut_set
        mutset2id[common_mut_set] = parent
        new_ids.append(parent)
    #print (parent)
    for n in group:
        if n != parent:
            G.add_edge(parent, n)
    return new_ids

# -----------------------------------------------------------------------------
# Matrix expansion ------------------------------------------------------------
# -----------------------------------------------------------------------------

def expand_matrices(new_ids, mutation_dict, edit_mat, sim_mat, headers):
    if not new_ids:
        return edit_mat, sim_mat, headers
    N_old, m = edit_mat.shape[0], len(new_ids)
    N_new = N_old + m
    dtype = edit_mat.dtype
    e = np.zeros((N_new, N_new), dtype=dtype)
    s = np.zeros_like(e)
    e[:N_old, :N_old] = edit_mat
    s[:N_old, :N_old] = sim_mat
    for off, nid in enumerate(new_ids):
        idx = N_old + off
        headers.append(nid)
        m_new = mutation_dict[nid]
        for j, eid in enumerate(headers[:idx]):
            m_exist = mutation_dict[eid]
            dist = len(m_new ^ m_exist)
            sim  = len(m_new & m_exist)
            e[idx, j] = e[j, idx] = dist
            s[idx, j] = s[j, idx] = sim
    return e, s, headers

# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    start_time = time.perf_counter()
    mutation_dict, mutset2id = load_mutations(mutation_json)
    daydiff_dict             = load_daydiff(daydiff_json)

    headers = list(mutation_dict.keys())
    G = nx.DiGraph()
    G.add_nodes_from(headers)

    sim_mat  = pd.read_csv(similarity_file, index_col=0).to_numpy()
    edit_mat = pd.read_csv(editdistance_file, index_col=0).to_numpy()

    gen = SerialIDGenerator()

    # Phase A
    #new_ids = add_best_edges_dense_group(G, edit_mat, sim_mat, headers, mutation_dict, mutset2id, daydiff_dict, new_gen)
    #edit_mat, sim_mat, headers = expand_matrices(new_ids, mutation_dict, edit_mat, sim_mat, headers)
    
    # -------- Phase A --------------------------------------------------
    
    if len(headers) < 2000:
        new_ids = add_best_edges_dense_group(G, edit_mat, sim_mat, headers, mutation_dict, mutset2id, daydiff_dict, gen)
    else:
        new_ids = add_best_edges_kNN(G, edit_mat, sim_mat, headers, mutation_dict, mutset2id, daydiff_dict, gen, k=8)
    
    new_ids = add_best_edges_kNN(G, edit_mat, sim_mat, headers, mutation_dict, mutset2id, daydiff_dict, gen, k=5)
    edit_mat, sim_mat, headers = expand_matrices(new_ids, mutation_dict, edit_mat, sim_mat, headers)

    print ("Weakly connected components:")
    print (nx.number_weakly_connected_components(G))
    print("Total nodes   :", len(G))
    print("Total edges   :", G.number_of_edges())
    # ------------------------------------------------------------------    # ------------------------------------------------------------------
    # Phase B – component bridging (ONE edge per component pair)
    # ------------------------------------------------------------------
    while nx.number_weakly_connected_components(G) > 1:
        comps = [np.fromiter((headers.index(n) for n in comp), dtype=np.int32)
                 for comp in nx.weakly_connected_components(G)]

        new_hypos = []
        for i in range(len(comps) - 1):
            idx_i = comps[i]
            for j in range(i + 1, len(comps)):
                idx_j = comps[j]

                sub = sim_mat[np.ix_(idx_i, idx_j)]
                if sub.size == 0:
                    continue
                max_sim = sub.max()
                if max_sim == 0:
                    continue

                r, c = np.unravel_index(sub.argmax(), sub.shape)
                u, v = headers[idx_i[r]], headers[idx_j[c]]

                # --------------- ONE call = ONE edge -------------------
                new_hypos += find_relation(G, u, v, mutation_dict, mutset2id, daydiff_dict, gen)

                # stop once connected
                if nx.number_weakly_connected_components(G) == 1:
                    break
            else:
                continue  # inner loop didn't break
            break         # inner loop broke (graph connected)

        if new_hypos:
            edit_mat, sim_mat, headers = expand_matrices(new_hypos, mutation_dict, edit_mat, sim_mat, headers)

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    with open(updated_json, "w") as fh:
        json.dump({k: sorted(list(v)) for k, v in mutation_dict.items()}, fh)

    np.save(updated_edit, edit_mat)
    np.save(updated_sim, sim_mat)

    adj = nx.to_numpy_array(G, nodelist=headers, dtype=int)
    np.save(adj_matrix_file, adj)

    print("Graph connected :", nx.is_weakly_connected(G))
    print("Total nodes     :", len(G))
    print("Total edges     :", G.number_of_edges())
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print (f"Elapsed time: {elapsed_time:.4f}seconds")


if __name__ == "__main__":
    main()
'''