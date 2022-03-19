import math


def calc_bezdek_index(probability_array, a):
    v_pc = 0
    v_pe = 0
    c = len(probability_array)
    n = len(probability_array[0])
    for i in range(c):
        for k in range(n):
            v_pc += probability_array[i][k]**2
            if a is not None:
                v_pe += probability_array[i][k] * math.log(probability_array[i][k], a)
            else:
                v_pe += probability_array[i][k] * math.log(probability_array[i][k])
    v_pc /= n
    v_pe /= -n

    return v_pc, v_pe
