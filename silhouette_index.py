from datetime import datetime

from numpy import average
from joblib import Parallel, delayed
from joblib._memmapping_reducer import has_shareable_memory
from dunn_index import find_distance


def calc_sil_index(points, u):
    d, intra, inter = calc_d_matrix(points, u)
    sil = []
    for point in range(len(points)):
        sil.append(calc_sil_for_point(point, d, intra, inter, u))
    return average(sil)


def calc_sil_for_point(point_index, d, intra, inter, u):
    bk = calc_inter_dist_for_point(point_index, d, inter, u)
    ak = calc_intra_dist_for_point(point_index, d, intra, u)
    print(str(datetime.now()) + " Done calc_intra_dist_for_point!")
    return (bk - ak) / max(bk, ak)


def calc_intra_dist_for_point(j, d, intra, u):
    w_distances = []
    for c in range(len(u)):
        intra_dist_sum = 0
        intra_dist_sum_w = 0
        for k in range(len(d)):
            if k == j:
                continue
            intra_dist_sum_w += intra[c][j][k] * d[j][k]
            intra_dist_sum += intra[c][j][k]
        if intra_dist_sum > 0:
            w_distances.append(intra_dist_sum_w / intra_dist_sum)

    return average(w_distances)


def calc_inter_dist_for_point(j, d, inter, u):
    w_distances = []
    for r in range(len(u)):
        dist_r_s = []
        for s in range(len(u)):
            inter_dist_sum = 0
            inter_dist_sum_w = 0
            for k in range(len(d)):
                if k == j:
                    continue
                inter_dist_sum_w += inter[r][s][j][k] * d[j][k]
                inter_dist_sum += inter[r][s][j][k]
            if inter_dist_sum > 0:
                dist_r_s.append(inter_dist_sum_w / inter_dist_sum)
        w_distances.append(average(dist_r_s))
    return min(w_distances)


def calc_d_matrix(points, u):
    d = []
    intra_dist = []
    inter_dist_arr = []

    def calculate(i):
        intra_dist_i = []
        inter_dist = []

        def calculate(s):
            inter_dist_i = []

            def calculate(j, k):
                if j == k:
                    djk = 0
                else:
                    djk = find_distance(points[j], points[k])
                if len(d) <= j:
                    d.append([djk])
                else:
                    d[j].append(djk)
                if i == s:
                    if len(intra_dist_i) <= j:
                        intra_dist_i.append([u[i][j] * u[i][k]])
                    else:
                        intra_dist_i[j].append(u[i][j] * u[i][k])
                if len(inter_dist_i) <= j:
                    inter_dist_i.append([(u[i][j] * u[s][k]) + (u[s][j] * u[i][k])])
                else:
                    inter_dist_i[j].append((u[i][j] * u[s][k]) + (u[s][j] * u[i][k]))

            Parallel(n_jobs=4, require='sharedmem')(
                delayed(has_shareable_memory)(calculate(j, k)) for k in range(len(points)) for j in range(len(points)))
            inter_dist.append(inter_dist_i)

        Parallel(n_jobs=1, require='sharedmem')(delayed(has_shareable_memory)(calculate(s)) for s in range(len(u)))

        if len(intra_dist_i) > 0:
            intra_dist.append(intra_dist_i)
        inter_dist_arr.append(inter_dist)

    Parallel(n_jobs=1, require='sharedmem')(delayed(has_shareable_memory)(calculate(i)) for i in range(len(u)))
    print(str(datetime.now()) + " Done calc_d_matrix!")
    return d, intra_dist, inter_dist_arr
