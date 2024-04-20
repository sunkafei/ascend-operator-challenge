import os
import sys
import numpy

loss = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):
    with open("output/meta", "r") as fp:
        dtype_str = fp.readline().strip()
        dtype = eval(dtype_str)
    real_result = numpy.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = numpy.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    print("=" * 50, real_result[:5], golden[:5], "=" * 50, sep='\n', end='\n', file=sys.stderr)
    result = numpy.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = numpy.maximum(numpy.abs(real_result), numpy.abs(golden))  # 获取最大值并组成新数组
    result_atol = numpy.less_equal(result, loss) # 计算绝对误差
    result_rtol = numpy.less_equal(result / numpy.add(deno, minimum), loss) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if numpy.sum(result_rtol == False) > real_result.size * loss and numpy.sum(result_atol == False) > real_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
