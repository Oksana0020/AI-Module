# sheet2_ex1_memory_explanation.py
# Exercise Sheet 2 — Exercise 1
#
# Exercise 1:
# Why does using True/False save memory?
# If we wished to use 0's and 1's how would these be stored?

import numpy as np

def explain_memory():
    """
    Explanation (short + practical):

    1) In NumPy:
       - A boolean array uses dtype=bool, typically 1 byte per element.
       - An integer array often uses int64 by default, typically 8 bytes per element.
       So a bool array can be ~8x smaller than int64.

    2) In Python lists:
       - A list of 0/1 stores pointers to full Python int objects.
       - Each Python int object has large overhead.
       This is far bigger than either NumPy bool or NumPy int arrays.

    If we used 0/1 in NumPy, they'd be stored as an integer dtype (int8/int32/int64)
    depending on how you create/cast the array.
    """
    n = 1000

    bool_array = np.random.choice([True, False], size=n)
    int_array_default = np.random.choice([0, 1], size=n)          # often int64
    int_array_int8 = int_array_default.astype(np.int8)            # force int8

    print("Bool dtype:", bool_array.dtype, "bytes:", bool_array.nbytes)
    print("Int default dtype:", int_array_default.dtype, "bytes:", int_array_default.nbytes)
    print("Int8 dtype:", int_array_int8.dtype, "bytes:", int_array_int8.nbytes)

if __name__ == "__main__":
    explain_memory()
