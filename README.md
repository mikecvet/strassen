# Strassen Multiplication Algorithm

This contains sample implementation of the Strassen Algorithm for matrix multiplication, written in C++ and Rust.

See [Wikipedia](http://en.wikipedia.org/wiki/Strassen_algorithm) or section [28.2 of CLRS](https://sites.math.rutgers.edu/~ajl213/CLRS/Ch28.pdf) for more information on the algorithm.

# C++

Most of the functionality is implemented in templated header files, under `cpp/src/strassen`. There is a test source file at `src/test/test_strassen_matrix.cpp` demonstrating how to use the matrix wrapper classes and the matrixm multipliers.

A `matrix<T>` object uses a `matrix_multiplier<T>` object to perform its matrix multiplication. This defaults to the `strassen_matrix_multiplier<T>`, but can be customized by passing a different type to the matrix constructor. For example,

```
strassen::matrix<int> A  (123, 456); // Default strassen_matrix_multiplier<int>
strassen::matrix<int> B (123, 456, new strassen::parallel_strassen_matrix_multiplier<int> ()); // Uses custom matrix multiplier
A.mult (B) // A now equals A * B
```

# Rust

The Rust implementation is under `rust/src` primarily located in two files - `matrix.rs` contains the `Matrix` implementation of matrix structure and convenience functions. `mult.rs` contains the acutal multiplication logic. Matrices are multiplied by passing a multiplication function pointer into `Matrix.mult`, for example

```
let a = Matrix::with_array(v1, x, y);
let b = Matrix::with_array(v2, y, x);

// mult takes fn(&Matrix, &Matrix) -> Matrix as an argument
let c = a.mult (&b, mult_strassen);
```

Multipliers for naive algorithms, naive-but-transposed (for better cache performance), and strassen are provided.