import numpy as np
import tensorflow as tf
from math import log

A = [4.65128586073990045278E-5,7.31589045238094711071E-3,1.33847639578309018650E-1,8.79691311754530315341E-1,2.71149851196553469920E0,4.25697156008121755724E0,3.29771340985225106936E0,1.00000000000000000126E0]
B = [6.90990488912553276999E-4,2.54043763932544379113E-2,2.82974860602568089943E-1,1.41172597751831069617E0,3.63800533345137075418E0,5.03278880143316990390E0,3.54771340985225096217E0,9.99999999999999998740E-1]
PIFS = 1.64493406684822643647
def polevlf(x, coef):
    ans = 0
    for i in range(len(coef)):
        ans = ans * x + coef[i]
    return ans
def spence(xx):
    x, w, y, z = 0, 0, 0, 0
    flag = 0
    x = xx

    if x == 1.0:
        return 0.0

    if x == 0.0:
        return PIFS

    if x > 2.0:
        x = 1.0/x
        flag |= 2

    if x > 1.5:
        w = 1.0/x - 1.0
        flag |= 2
    elif x < 0.5:
        w = -x
        flag |= 1
    else:
        w = x - 1.0

    y = -w * polevlf( w, A ) / polevlf( w, B )

    if flag & 1:
        y = PIFS - log(x) * log(1.0-x) - y

    if flag & 2:
        z = log(x)
        y = -0.5 * z * z  -  y
    return y

input_x = np.random.uniform(0.5, 1.5, [1024]).astype(np.float32)
golden = tf.math.special.spence(input_x.astype(np.float32)).numpy().astype(np.float32)
print(input_x)
print("std:", golden)
result = input_x
for i in range(len(result)):
    result[i] = spence(float(result[i]))
print("res:", result)
assert (abs(golden - result) < 1e-5).all()