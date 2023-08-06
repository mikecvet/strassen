# Strassen Multiplication Algorithm

This contains sample implementation of the Strassen Algorithm for matrix multiplication, written in C++ and Rust.

See [Wikipedia](http://en.wikipedia.org/wiki/Strassen_algorithm) or section [28.2 of CLRS](https://sites.math.rutgers.edu/~ajl213/CLRS/Ch28.pdf) for more information on the algorithm.

# C++

Most of the functionality is implemented in templated header files, under `cpp/src/strassen`. There is a test source file at `src/test/test_strassen_matrix.cpp` demonstrating how to use the matrix wrapper classes and the matrix multipliers.

A `matrix<T>` object uses a `matrix_multiplier<T>` object to perform its matrix multiplication. This defaults to the `strassen_matrix_multiplier<T>`, but can be customized by passing a different type to the matrix constructor. For example,

```
strassen::matrix<int> A  (123, 456); // Default strassen_matrix_multiplier<int>

// Uses specific matrix multiplier
strassen::matrix<int> B (123, 456, new strassen::parallel_strassen_matrix_multiplier<int> ()); 
A.mult (B) // A now equals A * B
```

# Rust

This implementation is discussed in [Better-than-Cubic Complexity for Matrix Multiplication in Rust](https://medium.com/@mikecvet/better-than-cubic-complexity-for-matrix-multiplication-in-rust-cf8dfb6299f6). The Rust implementation is under `rust/src` primarily located in two files - `matrix.rs` contains the `Matrix` implementation of matrix structure and convenience functions. These are hardcoded to use `f64` as values. `mult.rs` contains the acutal multiplication logic. Matrices are multiplied by passing a multiplication function pointer into `Matrix.mult()`, for example

```
let a = Matrix::with_array(v1, x, y);
let b = Matrix::with_array(v2, y, x);

// mult takes fn(&Matrix, &Matrix) -> Matrix as an argument
let c = a.mult (&b, mult_strassen);
```

Multipliers for naive algorithms, naive-but-transposed (for better cache performance), and strassen are provided.

```
~/code/strassen/rust ~>> cargo test

running 19 tests
test matrix::tests::test_at ... ok
test matrix::tests::test_add ... ok
test matrix::tests::test_is_square ... ok
test matrix::tests::test_eq ... ok
test matrix::tests::test_pad_cols ... ok
test matrix::tests::test_pad_none ... ok
test matrix::tests::test_pad_rows ... ok
test matrix::tests::test_reduce ... ok
test matrix::tests::test_sub ... ok
test matrix::tests::test_transpose_1 ... ok
test matrix::tests::test_transpose_2 ... ok
test mult::tests::test_mult_naive ... ok
test mult::tests::test_mult_strassen ... ok
test mult::tests::test_reconstitute ... ok
test mult::tests::test_submatrix_add ... ok
test mult::tests::test_submatrix_cpy ... ok
test mult::tests::test_mult_transpose ... ok
test mult::tests::test_submatrix_sub ... ok
test mult::tests::test_mult_aggregate ... ok

test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.88s

~/code/strassen/rust ~>> cargo build --release

[...]

~/code/strassen ~>> ./strassen --lower 75 --upper 100 --factor 50 --trials 2

running 50 groups of 2 trials with bounds between [75->3750, 100->5000]

x y nxn naive transpose strassen par_strassen
75 100 7500 0.00 0.00 1.00 0.00
150 200 30000 6.00 4.00 4.50 2.00
225 300 67500 13.00 9.00 8.50 5.00
300 400 120000 26.00 22.00 17.00 7.50
375 500 187500 53.00 46.00 40.00 13.50
450 600 270000 95.00 84.50 58.00 18.00
525 700 367500 153.50 138.00 86.50 25.00
600 800 480000 231.00 212.00 126.50 36.00
675 900 607500 334.00 310.00 212.50 60.50
750 1000 750000 465.50 429.50 296.00 78.00
825 1100 907500 632.00 580.00 378.50 103.50
900 1200 1080000 820.50 756.50 421.50 111.50
975 1300 1267500 1055.00 971.00 585.00 152.00
1050 1400 1470000 1318.50 1223.50 634.50 169.00
1125 1500 1687500 1638.50 1515.50 841.50 215.00
1200 1600 1920000 1993.00 1886.00 907.00 228.50
1275 1700 2167500 2402.00 2266.00 1211.50 313.00
1350 1800 2430000 2935.50 2716.00 1495.50 367.00
1425 1900 2707500 3526.00 3210.00 1605.00 399.00
1500 2000 3000000 4371.50 3745.50 2073.50 480.50
1575 2100 3307500 5461.50 4353.50 2294.00 556.00
1650 2200 3630000 6746.50 5018.50 2704.00 651.00
1725 2300 3967500 8645.00 5743.00 2453.00 604.00
1800 2400 4320000 7932.50 6540.00 2991.00 708.50
1875 2500 4687500 11744.50 7406.50 3603.00 903.00
1950 2600 5070000 12125.50 8354.00 4153.00 1012.50
2025 2700 5467500 15691.00 9375.50 4931.00 1140.00
2100 2800 5880000 14017.50 10456.00 4511.50 1078.50
2175 2900 6307500 19534.00 11641.50 5542.00 1287.00
2250 3000 6750000 30264.50 12907.00 6012.00 1398.50
2325 3100 7207500 36518.00 14251.50 6993.50 1619.00
2400 3200 7680000 29808.00 15752.50 6412.00 1502.50
2475 3300 8167500 41150.50 17257.00 7444.00 1765.00
2550 3400 8670000 49037.00 18872.50 8540.50 2020.50
2625 3500 9187500 112791.00 38312.50 9337.00 2227.50
2700 3600 9720000 59724.00 22591.00 10862.00 2512.00
2775 3700 10267500 62605.50 24913.00 10879.50 2394.00
2850 3800 10830000 69389.50 26443.50 11284.50 2662.50
2925 3900 11407500 75160.50 28781.50 13324.00 3064.00
3000 4000 12000000 81200.50 31076.00 14907.50 3358.50
3075 4100 12607500 63695.50 33272.00 17651.00 4531.50
3150 4200 13230000 92095.50 35832.50 16601.00 4171.50
3225 4300 13867500 99023.00 38462.50 15927.00 4508.00
3300 4400 14520000 104669.00 41279.00 19213.50 5071.00
3375 4500 15187500 112204.50 44136.50 19303.50 4957.00
3450 4600 15870000 119428.50 47308.50 17871.00 4596.00
3525 4700 16567500 126734.00 50397.50 21686.00 5782.50
3600 4800 17280000 131445.00 53683.50 21210.50 5660.00
3675 4900 18007500 141419.00 58530.00 28291.50 6811.00
3750 5000 18750000 154941.00 60990.00 26132.00 6613.00
```