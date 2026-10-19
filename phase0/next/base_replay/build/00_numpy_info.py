import numpy, sys
print("numpy", numpy.__version__, numpy.__file__)
try:
    from numpy._core._multiarray_umath import __cpu_features__ as F, __cpu_baseline__ as BL, __cpu_dispatch__ as DI
    print("baseline", BL); print("dispatch", DI)
    print("AVX512_SKX", F.get("AVX512_SKX"), "AVX512_ICL", F.get("AVX512_ICL"), "AVX512_SPR", F.get("AVX512_SPR"), "AVX2", F.get("AVX2"))
except Exception as e:
    print("cpu features unavailable", e)
