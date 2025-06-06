import os
import sys
import numpy as np


def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.int8) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.bool_) # 从bin文件读取预期运算结果
    print("=" * 50, real_result[:64], golden[:64], "=" * 50, sep='\n', end='\n', file=sys.stderr)
    if (real_result != golden).any():
        print("\033[1;31m[ERROR] result error!!!!!\033[0m")
        return False
    #for i in range(len(real_result)):
    #    if real_result[i] != golden[i]:
    #        print("[ERROR] result error out {} expect {} but {}".format(i,golden[i],real_result[i]))
    #        return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
