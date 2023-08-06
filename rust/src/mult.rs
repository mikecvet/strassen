pub use crate::matrix::Matrix;
use rand::thread_rng;
use std::io;

static mut TEST_STATE:bool = false;

/**
 * Naive multiplication algorithm. O(n^3).
 * Panics if matrices `a` and `b` are of incompatbile dimensions.
 */
pub fn 
mult_naive (a: &Matrix, b: &Matrix) -> Matrix {
    if a.rows == b.cols {
        let m = a.rows;
        let n = a.cols;
        let mut c: Vec<f64> = Vec::with_capacity(m * m);

        for i in 0..m {
            for j in 0..m {
                let mut sum:f64 = 0.0;
                for k in 0..n {
                    sum += a.at(i, k) * b.at(k, j);
                }

                c.push(sum);
            }
        }

        return Matrix::with_vector(c, m, m);
    } else {
        panic!("Matrix sizes do not match");
    }
}

/**
 * Variant of the naive multiplication algorithm, which uses the transpose of `b`, resuting in better memory locality performance characteristics. Still O(n^3).
 * Panics if matrices `a` and `b` are of incompatbile dimensions.
 */
pub fn 
mult_transpose (a: &Matrix, b: &Matrix) -> Matrix {
    if a.rows == b.cols {
        let m = a.rows;
        let n = a.cols;
        let t = b.transpose();
        let mut c: Vec<f64> = Vec::with_capacity(m * m);

        for i in 0..m {
            for j in 0..m {
                let mut sum:f64 = 0.0;
                for k in 0..n {
                    sum += a.at(i, k) * t.at(j, k);
                }

                c.push(sum);
            }
        }

        return Matrix::with_vector(c, m, m);
    } else {
        panic!("Matrix sizes do not match");
    }
}

/**
 * Strassen algorithm. See https://en.wikipedia.org/wiki/Strassen_algorithm
 * Breaks the provided matrices down into 7 smaller submatrices for multiplication, which results in 
 * smaller asymptotic complexity of around O(n^2.8), at the expense of a higher scalar constant due to the extra work required.
 * Falls back to the transpose naive multiplication method if row and column dimensions are 64 or less.
 * Recurses as input matrices are broken down and this algorithm is run further on those submatrices.
 * Panics if matrices `a` and `b` are of incompatbile dimensions.
 */
pub fn
mult_strassen (a: &Matrix, b: &Matrix) -> Matrix {

    if a.rows != b.cols {
        panic!("Matrix sizes do not match");
    }

    let mut max_dimension = a.rows;

    // Find the largest row or column dimension across `a` and `b`
    if a.cols >= a.rows && a.cols >= b.cols {
        max_dimension = a.cols;
    } else if b.rows >= b.cols && b.rows >= a.rows {
        max_dimension = b.rows;
    } else if b.cols >= b.rows && b.cols >= a.cols {
        max_dimension = b.cols;
    }

    // If the largest dimension is odd, we'll add one and then pad the matrix to make it
    // an even number of rows and columns
    if max_dimension % 2 == 1 {
        max_dimension += 1;
    }

    if a.is_square() && b.is_square() && a.rows == max_dimension {
        // The matrices are square; proceed
        return _mult_strassen(&a, &b);
    } else {
        // Pad `a` and `b` to `max_dimension` and pass to underlying function `_mult_strassen`. Strip out
        // extra padded rows and columns after that operation is complete.
        return _mult_strassen(
            &a.pad(max_dimension), 
            &b.pad(max_dimension)
        ).reduce(a.rows, a.rows);
    }
}

/**
 * Inner Strassen algorithm logic. 
 */
fn
_mult_strassen (a: &Matrix, b: &Matrix) -> Matrix {

    unsafe {
        // Ugly hack for enabling recursion testing in unit tests.
        // If not test state, fall back to transpose matrix multiplication if 
        // input Matrix rows and columns are 64 or less.
        if (!TEST_STATE && a.rows <= 64) || (TEST_STATE && a.rows <= 2) {
            return mult_transpose(a, b);
        }
    }

    /* This will be the row and column size of the submatrices */
    let m = a.rows / 2;
     
    /* Top left submatrix */
    let tl_row_start = 0;
    let tl_col_start = 0;

    /* Top right submatrix */
    let tr_row_start = 0;
    let tr_col_start = m;

    /* Bottom left submatrix */
    let bl_row_start = m;
    let bl_col_start = 0;

    /* Bottom right submatrix */
    let br_row_start = m;
    let br_col_start = m;

    /* Vectors for 7 submatrices of `a` */
    let mut aa1 = Vec::with_capacity(m * m);
    let mut aa2 = Vec::with_capacity(m * m);
    let mut aa3 = Vec::with_capacity(m * m);
    let mut aa4 = Vec::with_capacity(m * m);
    let mut aa5 = Vec::with_capacity(m * m);
    let mut aa6 = Vec::with_capacity(m * m);
    let mut aa7 = Vec::with_capacity(m * m);

    /* Vectors for 7 submatrices of `b` */
    let mut bb1 = Vec::with_capacity(m * m);
    let mut bb2 = Vec::with_capacity(m * m);
    let mut bb3 = Vec::with_capacity(m * m);
    let mut bb4 = Vec::with_capacity(m * m);
    let mut bb5 = Vec::with_capacity(m * m);
    let mut bb6 = Vec::with_capacity(m * m);
    let mut bb7 = Vec::with_capacity(m * m);

   /*
    * The output matrix C is expressed in terms of the block matrices M1..M7
    *
    * C1,1 = M1 + M4 - M5 + M7
    * C1,2 = M3 + M5
    * C2,1 = M2 + M4
    * C2,2 = M1 - M2 + M3 + M6
    * 
    * Each of the block matrices M1..M7 is composed of products of quadrants from A and B as follows:
    * 
    * M1 = AA[0] * BB[0] = (A1,1 + A2,2)(B1,1 + B2,2)
    * M2 = AA[1] * BB[1] = (A2,1 + A2,2)(B1,1)
    * M3 = AA[2] * BB[2] = (A1,1)(B1,2 - B2,2)
    * M4 = AA[3] * BB[3] = (A2,2)(B2,1 - B1,1)
    * M5 = AA[4] * BB[4] = (A1,1 + A1,2)(B2,2)
    * M6 = AA[5] * BB[5] = (A2,1 - A1,1)(B1,1 + B1,2)
    * M7 = AA[6] * BB[6] = (A1,2 - A2,2)(B2,1 + B2,2)
    */

    /* Initializes submatrices of `a` based on its quadrants, the manner described below */

    /* AA1 = (A1,1 + A2,2) */
    submatrix_add (&mut aa1, a, tl_row_start, tl_col_start, br_row_start, br_col_start, m);
    /* AA2 = (A2,1 + A2,2) */
    submatrix_add (&mut aa2, a, bl_row_start, bl_col_start, br_row_start, br_col_start, m);
    /* AA3 = (A1,1) */
    submatrix_cpy (&mut aa3, a, tl_row_start, tl_col_start, m);
    /* AA4 = (A2,2) */
    submatrix_cpy (&mut aa4, a, br_row_start, br_col_start, m);
    /* AA5 = (A1,1 + A1,2) */
    submatrix_add (&mut aa5, a, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m);
    /* AA6 = (A2,1 - A1,1) */
    submatrix_sub (&mut aa6, a, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m);
    /* AA7 = (A1,2 - A2,2) */
    submatrix_sub (&mut aa7, a, tr_row_start, tr_col_start, br_row_start, br_col_start, m);

    /* Initializes submatrices of `b` based on its quadrants, the manner described below */

    /* BB1 = (B1,1 + B2,2) */
    submatrix_add (&mut bb1, b, tl_row_start, tl_col_start, br_row_start, br_col_start, m);
    /* BB2 = (B1,1) */
    submatrix_cpy (&mut bb2, b, tl_row_start, tl_col_start, m);
    /* BB3 = (B1,2 - B2,2) */
    submatrix_sub (&mut bb3, b, tr_row_start, tr_col_start, br_row_start, br_col_start, m);
    /* BB4 = (B2,1 - B1,1) */
    submatrix_sub (&mut bb4, b, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m);
    /* BB5 = (B2,2) */
    submatrix_cpy (&mut bb5, b, br_row_start, br_col_start, m);
    /* BB6 = (B1,1 + B1,2) */
    submatrix_add (&mut bb6, b, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m); 
    /* BB7 = (B2,1 + B2,2) */
    submatrix_add (&mut bb7, b, bl_row_start, bl_col_start, br_row_start, br_col_start, m);

    /*
     * Build the intermediate matrices M1..M7
     * 
     * The following operations each recurse further, passing their respective submatrices into the 
     * main `mult_strassen` function above.
     */
     
    let mut m1 = mult_strassen(
        &mut Matrix::with_vector(aa1, m, m),
        &mut Matrix::with_vector(bb1, m, m)
    );

    let m2 = mult_strassen(
        &mut Matrix::with_vector(aa2, m, m),
        &mut Matrix::with_vector(bb2, m, m)
    );

    let m3 = mult_strassen(
        &mut Matrix::with_vector(aa3, m, m),
        &mut Matrix::with_vector(bb3, m, m)
    );

    let mut m4 = mult_strassen(
        &mut Matrix::with_vector(aa4, m, m),
        &mut Matrix::with_vector(bb4, m, m)
    );

    let mut m5 = mult_strassen(
        &mut Matrix::with_vector(aa5, m, m),
        &mut Matrix::with_vector(bb5, m, m)
    );

    let m6 = mult_strassen(
        &mut Matrix::with_vector(aa6, m, m),
        &mut Matrix::with_vector(bb6, m, m)
    );

    let mut m7 = mult_strassen(
        &mut Matrix::with_vector(aa7, m, m),
        &mut Matrix::with_vector(bb7, m, m)
    );

    /* C1,1 = M1 + M4 - M5 + M7 */
    let m11 = m7.sub(&m5).add(&m4).add(&m1);

    /* C1,2 = M3 + M5 */
    let m12 = m5.add(&m3);

    /* C2,1 = M2 + M4 */
    let m21 = m4.add(&m2);

    /* C2,2 = M1 - M2 + M3 + M6 */
    let m22 = m1.sub(&m2).add(&m3).add(&m6);

    /* Return a single matrix composing each of these four matrices as quadrants */
    return reconstitute(&m11, &m12, &m21, &m22, m, a.rows);
}

/**
 * Adds the two specified submatrices of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Similarly for the `b` values.
 * Quadrants have `m` rows and columns.
 */
pub fn 
submatrix_add (c: &mut Vec<f64>, a: &Matrix, 
                a_row_start: usize, a_col_start: usize, 
                b_row_start: usize, b_col_start: usize,
                m: usize) {

    for i in 0..m {
        for j in 0..m {
            c.push(
                a.at(a_row_start + i, a_col_start + j) + 
                a.at(b_row_start + i, b_col_start + j)
            )
        }
    }
}

/**
 * Subtracts the two specified submatrices of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Similarly for the `b` values.
 * Quadrants have `m` rows and columns.
 */
pub fn 
submatrix_sub (c: &mut Vec<f64>, a: &Matrix, 
                a_row_start: usize, a_col_start: usize, 
                b_row_start: usize, b_col_start: usize,
                m: usize) {
    
    for i in 0..m {
        for j in 0..m {
            c.push(
                a.at(a_row_start + i, a_col_start + j) - 
                a.at(b_row_start + i, b_col_start + j)
            )
        }
    }
}

/**
 * Copies the specified submatrix of `a` into `c`, using the provided quadrant offsets.
 * `a_row_start` and `a_col_start` refer to row and column offsets into `a`, used to represent a matrix quadrant.
 * Quadrants have `m` rows and columns.
 */
pub fn 
submatrix_cpy (c: &mut Vec<f64>, a: &Matrix, 
                a_row_start: usize, a_col_start: usize, 
                m: usize) {
    
    for i in 0..m {
        let indx = ((a_row_start + i) * a.cols) + a_col_start;
        c.extend_from_slice(&a.elements[indx..(indx + m)]);
    }
}

/**
 * Reconstitutes a large matrix composed of the four provided matrices, composing 
 * them as quadrants in a larger matrix.
 * `m11` refers to `M(1,1)` for example.
 */
pub fn 
reconstitute (m11: &Matrix, m12: &Matrix, 
               m21: &Matrix, m22: &Matrix, 
               m: usize, n: usize) -> Matrix {
    
    let mut v:Vec<f64> = Vec::with_capacity(n * n);
    let mut indx: usize;

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m11.elements[indx..(indx + m)]);
        v.extend_from_slice(&m12.elements[indx..(indx + m)]);
    }

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m21.elements[indx..(indx + m)]);
        v.extend_from_slice(&m22.elements[indx..(indx + m)]);
    }

    return Matrix::with_vector(v, n, n);
}

#[cfg(test)]
mod tests {

    use rand::Rng;

    use super::*;

    fn test_multiplication_outputs(multipler: fn(&Matrix, &Matrix) -> Matrix) {
        let v1: Vec<f64> = vec![12.0, 8.0, 4.0, 3.0, 17.0, 14.0, 9.0, 8.0, 10.0];
        let v2: Vec<f64> = vec![5.0, 19.0, 3.0, 6.0, 15.0, 9.0, 7.0, 8.0, 16.0];
        let v3: Vec<f64> = vec![136.0, 380.0, 172.0, 215.0, 424.0, 386.0, 163.0, 371.0, 259.0];

        let a: Matrix = Matrix::with_vector(v1, 3, 3);
        let b: Matrix = Matrix::with_vector(v2, 3, 3);
        let c: Matrix = Matrix::with_vector(v3, 3, 3);

        assert!(a.mult(&b, multipler).eq(&c));

        let v4: Vec<f64> = vec![7.0, 14.0, 15.0, 6.0, 4.0, 8.0, 12.0, 3.0, 14.0, 21.0, 6.0, 9.0, 13.0, 7.0, 6.0, 4.0];
        let v5: Vec<f64> = vec![5.0, 7.0, 14.0, 2.0, 8.0, 16.0, 4.0, 9.0, 13.0, 6.0, 8.0, 4.0, 6.0, 3.0, 2.0, 4.0];
        let v6: Vec<f64> = vec![378.0, 381.0, 286.0, 224.0, 258.0, 237.0, 190.0, 140.0, 370.0, 497.0, 346.0, 277.0, 223.0, 251.0, 266.0, 129.0];

        let d: Matrix = Matrix::with_vector(v4, 4, 4);
        let e: Matrix = Matrix::with_vector(v5, 4, 4);
        let f: Matrix = Matrix::with_vector(v6, 4, 4);

        assert!(d.mult(&e, multipler).eq(&f));
    }

    #[test]
    fn test_mult_naive () {
        test_multiplication_outputs(mult_naive);
    }

    #[test]
    fn test_mult_transpose () {
        test_multiplication_outputs(mult_transpose);
    }

    #[test]
    fn test_mult_strassen () {
        unsafe {
            TEST_STATE = true;
        }

        test_multiplication_outputs(mult_strassen);
    }

    #[test]
    fn test_mult_aggregate () {
        unsafe {
            TEST_STATE = true;
        }

        let cols = 123;
        let rows = 219;
        let n = rows * cols;
        let mut v1: Vec<f64> = Vec::with_capacity(n);
        let mut v2: Vec<f64> = Vec::with_capacity(n);

        let mut rng = thread_rng();

        for _ in 0..n {
            v1.push(rng.gen::<f64>() % 1000000.0);
            v2.push(rng.gen::<f64>() % 1000000.0);
        }

        let a: Matrix = Matrix::with_vector(v1, rows, cols);
        let b: Matrix = Matrix::with_vector(v2, cols, rows);

        let naive_result = a.mult(&b, mult_naive);
        let transpose_result = a.mult(&b, mult_transpose);
        let strassen_result = a.mult(&b, mult_strassen);

        assert!(naive_result.eq(&transpose_result));
        assert!(transpose_result.eq(&naive_result));
        assert!(naive_result.eq(&strassen_result));
        assert!(strassen_result.eq(&naive_result));
        assert!(transpose_result.eq(&strassen_result));
        assert!(strassen_result.eq(&transpose_result));
    }

    #[test]
    fn test_submatrix_add () {
        /*
         * Given an input matrix
         * -          -
         * | 1 1 2  2 |
         * | 1 1 2  2 |
         * | 3 3 10 10|
         * | 3 3 10 10|
         * -          -
         * 
         * Should correctly add the top-left 2x2 and bottom-right 2x2 subarrays
         */
        let v1: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 3.0, 3.0, 10.0, 10.0];
        let mut v2: Vec<f64> = Vec::with_capacity(4);
        let v3: Vec<f64> = vec![11.0, 11.0, 11.0, 11.0];
        let a: Matrix = Matrix::with_vector(v1, 4, 4);

        submatrix_add (&mut v2, &a, 0, 0, 2, 2, 2);

        assert!(v2.eq(&v3));
    }

    #[test]
    fn test_submatrix_sub () {
        /*
         * Given an input matrix
         * -          -
         * | 1 1 2  2 |
         * | 1 1 2  2 |
         * | 3 3 10 10|
         * | 3 3 10 10|
         * -          -
         * 
         * Should correctly add the top-left 2x2 and bottom-right 2x2 subarrays
         */
        let v1: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 3.0, 3.0, 10.0, 10.0];
        let mut v2: Vec<f64> = Vec::with_capacity(4);
        let v3: Vec<f64> = vec![9.0, 9.0, 9.0, 9.0];
        let a: Matrix = Matrix::with_vector(v1, 4, 4);

        submatrix_sub (&mut v2, &a, 2, 2, 0, 0, 2);

        assert!(v2.eq(&v3));
    }

    #[test]
    fn test_submatrix_cpy () {
        /*
         * Given an input matrix
         * -          -
         * | 1 1 2  2 |
         * | 1 1 2  2 |
         * | 3 3 10 10|
         * | 3 3 10 10|
         * -          -
         * 
         * Should correctly copy the top-left and bottom-right 2x2 subarrays into new vecs
         */
        let v1: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 3.0, 3.0, 10.0, 10.0];
        let mut v2: Vec<f64> = Vec::with_capacity(4);
        let mut v3: Vec<f64> = Vec::with_capacity(4);
        let v4: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
        let v5: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0];
        let a: Matrix = Matrix::with_vector(v1, 4, 4);

        submatrix_cpy(&mut v2, &a, 0, 0, 2);
        submatrix_cpy(&mut v3, &a, 2, 2, 2);

        assert!(v2.eq(&v4));
        assert!(v3.eq(&v5));
    }

    #[test]
    fn test_reconstitute () {
        /*
         * Given 3 input submatrices representing M1,1 | M1,2 | M2,1 | M2,2
         * -     - -      -
         * | 1 1 | | 2  2 |
         * | 1 1 | | 2  2 |
         * -     - -      -
         * -     - -       -
         * | 3 3 | | 10 10 |
         * | 3 3 | | 10 10 |
         * -     - -       -
         * 
         * Should correctly reconstitute a 4x4 matrix as such:
         * -          -
         * | 1 1 2  2 |
         * | 1 1 2  2 |
         * | 3 3 10 10|
         * | 3 3 10 10|
         * -          -
         */
        let m11 = Matrix::with_vector(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        let m12 = Matrix::with_vector(vec![2.0, 2.0, 2.0, 2.0], 2, 2);
        let m21 = Matrix::with_vector(vec![3.0, 3.0, 3.0, 3.0], 2, 2);
        let m22 = Matrix::with_vector(vec![10.0, 10.0, 10.0, 10.0], 2, 2);
        let a = Matrix::with_vector(vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 3.0, 3.0, 10.0, 10.0], 4, 4);

        let b = reconstitute (&m11, &m12, &m21, &m22, 2, 4);

        assert!(a.eq(&b));
    }
}