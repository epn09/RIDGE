# cython: language_level=3, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from cython cimport integral
from libcpp.queue cimport queue

cimport numpy as np
import numpy as np


ctypedef fused real:
    integral
    float
    double


cdef int get_group(int[:] group, integral idx) noexcept:
    if group[idx] < 0:
        return idx
    group[idx] = get_group(group, group[idx])
    return group[idx]


cdef bint join_group(int[:] group, integral idx1, integral idx2) noexcept:
    cdef int g1 = get_group(group, idx1)
    cdef int g2 = get_group(group, idx2)
    if g1 == g2:
        return False

    # Swap if g1 is the larger group. Note that size is negated.
    if group[g1] < group[g2]:
        g1, g2 = g2, g1

    group[g2] += group[g1]
    group[g1] = g2

    return True


cpdef spanning_tree(integral[:] vertices, real[:] values, int ny, int nx):
    cdef int nv = vertices.size
    cdef Py_ssize_t[:] order = np.argsort(values)[::-1]

    cdef int[:] group = np.full(nv, -1, dtype=np.intc)
    cdef integral idx1, idx2
    cdef int y1, x1, y2, x2,  dx, dy

    cdef int[:, :] edges = np.empty((nv - 1, 2), dtype=np.intc)
    cdef int ne = 0

    cdef Py_ssize_t i
    cdef int g1, g2
    for i in range(nv):
        g1 = order[i]
        idx1 = vertices[g1]
        y1 = idx1 // nx
        x1 = idx1 % nx

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0:
                    continue

                y2 = y1 + dy
                x2 = x1 + dx
                idx2 = y2 * nx + x2
                if (
                    y2 < 0
                    or y2 >= ny
                    or x2 < 0
                    or x2 >= nx
                    or (g2 := binary_search(vertices, idx2)) == -1
                ):
                    continue

                if not join_group(group, g1, g2):
                    continue

                edges[ne, 0] = idx1
                edges[ne, 1] = idx2
                ne += 1

    return np.asarray(edges[:ne])


def spanning_tree_array(real[:, :] arr):
    cdef int[:] idx = np.flatnonzero(np.isfinite(arr)).astype(np.intc)
    cdef real[:] values = np.ravel(arr)[idx]
    return spanning_tree(idx, values, arr.shape[0], arr.shape[1])


cdef Py_ssize_t binary_search(integral[:] arr, integral value) noexcept:
    cdef Py_ssize_t low = 0, high = arr.shape[0], mid

    while low < high:
        mid = (low + high) // 2
        if arr[mid] < value:
            low = mid + 1
        elif arr[mid] > value:
            high = mid - 1
        else:
            return mid

    if 0 <= low < arr.shape[0] and arr[low] == value:
        return low
    return -1


cpdef compress(integral[:, :] graph):
    cdef integral[:] vertices = np.unique(graph)
    nv = vertices.shape[0]
    cdef int[:, :] res = np.empty_like(graph, dtype=np.intc)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            res[i, j] = binary_search(vertices, graph[i, j])
    return np.asarray(vertices), np.asarray(res)


def decompress(edges, vertices):
    return vertices[edges]


cdef integral get_next(integral u, integral[:] next, char[:] is_valid, bint allow_present) noexcept:
    if u < 0:
        return -1
    if allow_present and is_valid[u]:
        return u
    next[u] = get_next(next[u], next, is_valid, True)
    return next[u]


cdef int bfs_ll(
    int start,
    integral[:, :] edges,
    int[:] first_edge,
    int[:] next_edge,
    char[:] is_valid_edge,
    char[:] last_visit,
    int[:] trace,
    int visit
) noexcept:
    """BFS for linked list format."""

    cdef queue[int] q

    last_visit[start] = visit
    trace[start] = -1
    q.push(start)

    cdef int u = start, v, e
    cdef bint allow_present

    while q.size() > 0:
        u = q.front()
        q.pop()

        e = first_edge[u]
        allow_present = True
        while (e := get_next(e, next_edge, is_valid_edge, allow_present)) >= 0:
            allow_present = False
            v = edges[e >> 1, 0] ^ edges[e >> 1, 1] ^ u
            if last_visit[v] != visit:
                last_visit[v] = visit
                trace[v] = e
                q.push(v)

    return u


cpdef decompose(integral[:, :] edges):
    cdef int nv = np.max(edges) + 1
    cdef int ne = edges.shape[0]
    cdef int[:] next_edge = np.full(2 * ne, -1, dtype=np.intc)
    cdef int[:] first_edge = np.full(nv, -1, dtype=np.intc)
    cdef char[:] is_valid_edge = np.full(2 * ne, 1, dtype=np.int8)

    cdef int i, u, v, e
    for i in range(ne):
        u = edges[i, 0]
        v = edges[i, 1]
        e = i * 2
        next_edge[e] = first_edge[u]
        first_edge[u] = e
        next_edge[e + 1] = first_edge[v]
        first_edge[v] = e + 1

    cdef char[:] is_valid_vertex = np.full(nv, 1, dtype=np.int8)
    cdef int[:] next_vertex = np.arange(1, nv + 1, dtype=np.intc)
    next_vertex[nv - 1] = -1

    cdef char[:] last_visit = np.full(nv, -1, dtype=np.int8)
    cdef int[:] trace = np.full(nv, -1, dtype=np.intc)

    cdef int n_valid_edge = ne
    cdef int end_1, end_2
    cdef bint allow_present

    cdef int[:] res_vertices = np.empty(ne * 2, dtype=np.intc)
    cdef int[:] res_path_length = np.zeros(ne, dtype=np.intc)
    cdef int res_vertices_count = 0
    cdef int res_path_count = 0

    while n_valid_edge > 0:
        u = 0
        allow_present = True
        while (u := get_next(u, next_vertex, is_valid_vertex, allow_present)) >= 0:
            allow_present = False
            # Check if u has any edges
            if get_next(first_edge[u], next_edge, is_valid_edge, True) < 0:
                # Invalidate as it has no edge
                is_valid_vertex[u] = 0
                continue

            # Start finding a longest path for the tree containing u
            end_1 = bfs_ll(u, edges, first_edge, next_edge, is_valid_edge, last_visit, trace, 0)
            end_2 = bfs_ll(end_1, edges, first_edge, next_edge, is_valid_edge, last_visit, trace, 1)

            while end_2 != end_1:
                # Remove the edge
                e = trace[end_2]
                is_valid_edge[e] = is_valid_edge[e ^ 1] = 0
                n_valid_edge -= 1

                # Add the vertices to the result
                res_vertices[res_vertices_count] = end_2
                res_vertices_count += 1
                res_path_length[res_path_count] += 1

                # Move up the tree
                end_2 = edges[e >> 1, 0] ^ edges[e >> 1, 1] ^ end_2

            # Add the last vertex
            res_vertices[res_vertices_count] = end_1
            res_vertices_count += 1
            res_path_length[res_path_count] += 1
            res_path_count += 1

    return np.asarray(res_vertices[:res_vertices_count]), np.asarray(res_path_length[:res_path_count])
