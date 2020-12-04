import math

def ci95(acc_1, acc_2, size):
    f = acc_1 - acc_2
    v = acc_1* (1- acc_1)/size + acc_2*(1-acc_2)/size
    std = math.sqrt(v)
    coe = 1.96
    bound_1 = f + coe * std
    bound_2 = f - coe * std
    result = [bound_1, bound_2]
    result.sort()
    # print(v)
    result = tuple(result)
    print(f'{f} +- {coe} * {std} = {result}')


# ci95(0.123, 0.0829, 10000)
# ci95(0.10, 0.098, 10000)
# ci95(0.0829, 0.0041, 10000)



ci95(0.912, 0.904, 500)
ci95(0.904, 0.618, 500)

ci95(0.012, 0.004, 500)
ci95(0.28, 0.012, 500)



