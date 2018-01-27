import numpy as np
import matplotlib.pyplot as plt
import math

# Z_SHORT = 0.25  # Weight for short reading
# Z_MAX = 0.25    # Weight for max reading
# Z_RAND = 0.25   # Weight for random reading
# SIGMA_HIT = 0.2 # Noise value for hit reading
# Z_HIT = 0.25    # Weight for hit reading

Z_HIT = 4
Z_SHORT = 3
Z_MAX = 0.1
Z_RAND = 2

Z_NORM = float(Z_HIT + Z_SHORT + Z_MAX + Z_RAND)
Z_HIT /= Z_NORM
Z_SHORT /= Z_NORM
Z_MAX /= Z_NORM
Z_RAND /= Z_NORM


THETA_DISCRETIZATION = 112 # Discretization of scanning angle
SQUASH_FACTOR_N = 1.5
INV_SQUASH_FACTOR = 1 / SQUASH_FACTOR_N    # Factor for helping the weight distribution to be less peaked

# Z_SHORT = 0.25  # Weight for short reading, something in the way
# Z_MAX = 0.25    # Weight for max reading, out of range
# Z_RAND = 0.25   # Weight for random reading, random noise
# Z_HIT = 0.25    # Weight for hit reading, hit obj properly
# Z_SHORT = 0.1  # Weight for short reading, something in the way
# Z_MAX = 0.1    # Weight for max reading, out of range
# Z_RAND = 0.1   # Weight for random reading, random noise
# Z_HIT = 0.7    # Weight for hit reMAX_RANGE_METERSading, hit obj properly
SIGMA_HIT = 20 # Noise value for hit reading

LAMBDA_SHORT = 0.01

def precompute_sensor_model(max_range_px):
    table_width = int(max_range_px) + 1
    sensor_model_table = np.zeros((table_width,table_width))

    # ch 6.2 - have distribution for what a range would be
    # given map and location.

    # Populate sensor model table as specified
    # Note that the row corresponds to the observed measurement and the column corresponds to the expected measurement
    # YOUR CODE HERE

    assert abs((Z_HIT + Z_SHORT + Z_MAX + Z_RAND) - 1.0) < 1E-4

    def compute_eng(ztk, ztkstar, varhit):
        denom1 = math.sqrt(2.0 * np.pi * varhit)
        numer2 = math.pow(ztk - ztkstar, 2.0)
        denom2 = varhit
        return (1.0 / denom1) * np.exp((-0.5) * numer2 / denom2)


    def compute_p_short_unnormalized(ztk, ztkstar):
        if 0.0 <= ztk <= ztkstar:
            return LAMBDA_SHORT * np.exp(-LAMBDA_SHORT * ztk)
        else:
            return 0.0

    engs = np.zeros((table_width, table_width))
    p_short_unnormalized = np.zeros((table_width, table_width))
    for i in range(table_width):
        for j in range(table_width):
            engs[i][j] = compute_eng(float(i), float(j), SIGMA_HIT * SIGMA_HIT)
            p_short_unnormalized[i][j] = compute_p_short_unnormalized(float(i), float(j))

    engs = engs / engs.sum(axis=0, dtype=float)
    p_short_normalized = p_short_unnormalized / p_short_unnormalized.sum(axis=0, dtype=float)

    def compute_p_hit(i, j, max_range_px):
        return engs[i][j] if 0.0 <= i <= max_range_px else 0.0

    def compute_p_short(i, j):
        return p_short_normalized[i][j]

    def compute_p_max(i, max_range_px):
        return 1 if i == max_range_px else 0.0

    def compute_p_rand(i, max_range_px):
        return 1.0 / max_range_px if 0 <= i < max_range_px else 0.0

    for i in range(table_width): # observed
        for j in range(table_width): # expected
            p_hit = compute_p_hit(i, j, max_range_px)
            p_short = compute_p_short(i, j)
            p_max = compute_p_max(i, max_range_px)
            p_rand = compute_p_rand(i, max_range_px)

            #print(i, j, max_range_px, LAMBDA_SHORT, p_hit, p_short, p_max, p_rand)

            assert 0 <= p_hit <= 1
            assert 0 <= p_rand <= 1
            assert 0 <= p_short <= 1
            assert 0 <= p_max <= 1

            sensor_model_table[i][j] = Z_HIT*p_hit + Z_SHORT*p_short + Z_MAX*p_max + Z_RAND*p_rand

    column_sums = sensor_model_table.sum(axis=0)
    row_sums = sensor_model_table.sum(axis=1)
    #print("Column Sums: ", column_sums)

    print("SMT:")
    print(sensor_model_table)
    for j in range(table_width):
        assert abs(column_sums[j] - 1.0) <= 0.02

    return sensor_model_table

sm = precompute_sensor_model(280)
print("Sum on columns: ", sm.sum(axis=0))
model_size = sm.shape[0]
xs = range(0, model_size)
ys = sm[xs, int(model_size * 3) / 5]
#print(ys)
plt.plot(xs, ys, '.')
plt.show()
