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
