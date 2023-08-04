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

The Rust implementation is under `rust/src` primarily located in two files - `matrix.rs` contains the `Matrix` implementation of matrix structure and convenience functions. These are hardcoded to use `f64` as values. `mult.rs` contains the acutal multiplication logic. Matrices are multiplied by passing a multiplication function pointer into `Matrix.mult()`, for example

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

~/code/strassen/rust ~>> ./target/release/strassen --lower 120 --upper 140 --factor 12 --trials 10
running 12 groups of 10 trials with bounds between [120->1440, 140->1680]
x y nxn naive transpose strassen
120 140 16800 1.50 1.40 1.40
240 280 67200 11.20 8.10 7.00
360 420 151200 41.50 34.30 25.80
480 560 268800 104.80 90.40 51.20
600 700 420000 206.20 186.10 92.90
720 840 604800 361.50 333.20 184.90
840 980 823200 580.60 540.70 278.60
960 1120 1075200 879.70 827.40 377.80
1080 1260 1360800 1249.70 1186.50 520.30
1200 1400 1680000 1777.60 1676.80 693.50
1320 1540 2032800 2361.80 2232.30 1140.60
1440 1680 2419200 3150.70 2884.90 1316.70
```