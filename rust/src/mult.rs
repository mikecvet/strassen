pub use crate::matrix::Matrix;
use rand::thread_rng;
use std::io;

static mut TEST_STATE:bool = false;

pub fn 
mult_naive (a: &Matrix, b: &Matrix) -> Matrix {
    if a.rows == b.cols {
        let m = a.rows;
        let n = a.cols;

        let mut c: Vec<i64> = Vec::with_capacity(m * m);

        for i in 0..m {
            for j in 0..m {

                let mut sum:i64 = 0;
                for k in 0..n {
                    sum += a.at(i, k) * b.at(k, j);
                }

                c.push(sum);
            }
        }

        return Matrix::with_array(c, m, m);
    } else {
        panic!("Matrix sizes do not match");
    }
}

pub fn 
mult_transpose (a: &Matrix, b: &Matrix) -> Matrix {
    // println!("t: input dimensions: a[{}, {}] b[{}, {}]", a.rows, a.cols, b.rows, b.cols);

    if a.rows == b.cols {
        let m = a.rows;
        let n = a.cols;

        let t = b.transpose();
        let mut c: Vec<i64> = Vec::with_capacity(m * m);

        for i in 0..m {
            for j in 0..m {

                let mut sum:i64 = 0;
                for k in 0..n {
                    sum += a.at(i, k) * t.at(j, k);
                }

                c.push(sum);
            }
        }

        return Matrix::with_array(c, m, m);
    } else {
        panic!("Matrix sizes do not match");
    }
}

pub fn
mult_strassen (a: &Matrix, b: &Matrix) -> Matrix {

    //println!("s: input dimensions: a[{}, {}] b[{}, {}]", a.rows, a.cols, b.rows, b.cols);
    let mut max_dimension = a.cols;

    if a.rows >= a.cols && a.rows >= b.rows {
        max_dimension = a.rows;
    } else if a.cols >= a.rows && a.cols >= b.cols {
        max_dimension = a.cols;
    } else if b.rows >= b.cols && b.rows >= a.rows {
        max_dimension = b.rows;
    } else if b.cols >= b.rows && b.cols >= a.cols {
        max_dimension = b.cols;
    }

    /* Find the nearest power of 2 greater than the largest dimension of these matrices */
    // let log2 = (2.0 as f32).log(10.0);
    // let log_max = (max_dimension as f32).log(10.0);
    // let padding_size: usize = (2 as usize).pow((log_max / log2) as u32 + 1);
    
    if max_dimension % 2 == 1 {
        max_dimension += 1;
    }

    // println!("input dimensions: a[{}, {}] b[{}, {}] max: {} padding: {}", 
    // a.rows, a.cols, b.rows, b.cols, max_dimension, padding_size);

    return __mult_strassen(
        &a.pad(max_dimension), 
        &b.pad(max_dimension)
    ).reduce(a.rows, a.rows);
}

fn
__mult_strassen (a: &Matrix, b: &Matrix) -> Matrix {

    unsafe {
        if (!TEST_STATE && a.rows <= 64) || (TEST_STATE && a.rows <= 2) {
            return mult_transpose(a, b);
        }
    }

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

    let mut a_submatrices: [Vec<i64>; 7] = [
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
    ];

    let mut b_submatrices: [Vec<i64>; 7] = [
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
        Vec::with_capacity(m * m),
    ];

    let mut intermediate:Vec<Matrix> = Vec::with_capacity(7);

    /*
     * The output matrix C is expressed in terms of the block matrices M1..M7
     *
     * C1,1 = M1 + M4 - M5 + M7
     * C1,2 = M3 + M5
     * C2,1 = M2 + M4
     * C2,2 = M1 - M2 + M3 + M6
     * 
     * Each of the block matrices M1..M7 is composed of quadrants from A and B as follows:
     * 
     * M1 = AA[0] * BB[0] = (A1,1 + A2,2)(B1,1 + B2,2)
     * M2 = AA[1] * BB[1] = (A2,1 + A2,2)(B1,1)
     * M3 = AA[2] * BB[2] = (A1,1)(B1,2 - B2,2)
     * M4 = AA[3] * BB[3] = (A2,2)(B2,1 - B1,1)
     * M5 = AA[4] * BB[4] = (A1,1 + A1,2)(B2,2)
     * M6 = AA[5] * BB[5] = (A2,1 - A1,1)(B1,1 + B1,2)
     * M7 = AA[6] * BB[6] = (A1,2 - A2,2)(B2,1 + B2,2)
     */

    /* AA[0] = (A1,1 + A2,2) */
    __submatrix_add (&mut a_submatrices[0], a, tl_row_start, tl_col_start, br_row_start, br_col_start, m);
    /* AA[1] = (A2,1 + A2,2) */
    __submatrix_add (&mut a_submatrices[1], a, bl_row_start, bl_col_start, br_row_start, br_col_start, m);
    /* AA[2] = (A1,1) */
    __submatrix_cpy (&mut a_submatrices[2], a, tl_row_start, tl_col_start, m);
    /* AA[3] = (A2,2) */
    __submatrix_cpy (&mut a_submatrices[3], a, br_row_start, br_col_start, m);
    /* AA[4] = (A1,1 + A1,2) */
    __submatrix_add (&mut a_submatrices[4], a, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m);
    /* AA[5] = (A2,1 - A1,1) */
    __submatrix_sub (&mut a_submatrices[5], a, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m);
    /* AA[6] = (A1,2 - A2,2) */
    __submatrix_sub (&mut a_submatrices[6], a, tr_row_start, tr_col_start, br_row_start, br_col_start, m);

    /* BB[0] = (B1,1 + B2,2) */
    __submatrix_add (&mut b_submatrices[0], b, tl_row_start, tl_col_start, br_row_start, br_col_start, m);
    /* BB[1] = (B1,1) */
    __submatrix_cpy (&mut b_submatrices[1], b, tl_row_start, tl_col_start, m);
    /* BB[2] = (B1,2 - B2,2) */
    __submatrix_sub (&mut b_submatrices[2], b, tr_row_start, tr_col_start, br_row_start, br_col_start, m);
    /* BB[3] = (B2,1 - B1,1) */
    __submatrix_sub (&mut b_submatrices[3], b, bl_row_start, bl_col_start, tl_row_start, tl_col_start, m);
    /* BB[4] = (B2,2) */
    __submatrix_cpy (&mut b_submatrices[4], b, br_row_start, br_col_start, m);
    /* BB[5] = (B1,1 + B1,2) */
    __submatrix_add (&mut b_submatrices[5], b, tl_row_start, tl_col_start, tr_row_start, tr_col_start, m); 
    /* BB[6] = (B2,1 + B2,2) */
    __submatrix_add (&mut b_submatrices[6], b, bl_row_start, bl_col_start, br_row_start, br_col_start, m);
     
     for i in 0..7 {
        intermediate.push(
            mult_transpose(
                &mut Matrix::with_array(a_submatrices[i].to_vec(), m, m),
                &mut Matrix::with_array(b_submatrices[i].to_vec(), m, m)
            )
        );
    }

    /* C1,1 = M1 + M4 - M5 + M7 */
    let mut m11 = intermediate[0].copy();
    m11.add(&intermediate[3])
         .sub(&intermediate[4])
         .add(&intermediate[6]);

    /* C1,2 = M3 + M5 */
    let mut m12 = intermediate[2].copy();
    m12.add(&intermediate[4]);

    /* C2,1 = M2 + M4 */
    let mut m21 = intermediate[1].copy();
    m21.add(&intermediate[3]);

    /* C2,2 = M1 - M2 + M3 + M6 */
    let mut m22 = intermediate[0].copy();
    m22.sub(&intermediate[1])
         .add(&intermediate[2])
         .add(&intermediate[5]);

    return __reconstitute(&m11, &m12, &m21, &m22, m, a.rows);
}

fn 
__submatrix_add (c: &mut Vec<i64>, a: &Matrix, 
                a_row_start: usize, a_col_start: usize, 
                b_row_start: usize, b_col_start: usize,
                m: usize) {
    
    // println!("A: {}", a);
    // println!("__submatrix_add: a_row: {} a_col: {} b_row: {} b_col: {} m: {}", a_row_start, a_col_start, b_row_start, b_col_start, m);
    for i in 0..m {
        for j in 0..m {
            c.push(
                a.at(a_row_start + i, a_col_start + j) + 
                a.at(b_row_start + i, b_col_start + j)
            )
        }
    }
}

fn 
__submatrix_sub (c: &mut Vec<i64>, a: &Matrix, 
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

fn 
__submatrix_cpy (c: &mut Vec<i64>, a: &Matrix, 
                a_row_start: usize, a_col_start: usize, 
                m: usize) {
    
    for i in 0..m {
        let indx = ((a_row_start + i) * a.cols) + a_col_start;
        c.extend_from_slice(&a.array[indx..(indx + m)]);
    }
}

fn 
__reconstitute (m11: &Matrix, m12: &Matrix, 
                m21: &Matrix, m22: &Matrix, 
                m: usize, n: usize) -> Matrix {
    
    let mut v:Vec<i64> = Vec::with_capacity(n * n);
    let mut indx: usize;

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m11.array[indx..(indx + m)]);
        v.extend_from_slice(&m12.array[indx..(indx + m)]);
    }

    for i in 0..m {
        indx = i * m;
        v.extend_from_slice(&m21.array[indx..(indx + m)]);
        v.extend_from_slice(&m22.array[indx..(indx + m)]);
    }

    return Matrix::with_array(v, n, n);
}

#[cfg(test)]
mod tests {

    use rand::Rng;

    use super::*;

    fn test_multiplication_outputs(multipler: fn(&Matrix, &Matrix) -> Matrix) {
        let v1: Vec<i64> = vec![12, 8, 4, 3, 17, 14, 9, 8, 10];
        let v2: Vec<i64> = vec![5, 19, 3, 6, 15, 9, 7, 8, 16];
        let v3: Vec<i64> = vec![136, 380, 172, 215, 424, 386, 163, 371, 259];

        let a: Matrix = Matrix::with_array(v1, 3, 3);
        let b: Matrix = Matrix::with_array(v2, 3, 3);
        let c: Matrix = Matrix::with_array(v3, 3, 3);

        assert!(a.mult(&b, multipler).eq(&c));

        let v4: Vec<i64> = vec![7, 14, 15, 6, 4, 8, 12, 3, 14, 21, 6, 9, 13, 7, 6, 4];
        let v5: Vec<i64> = vec![5, 7, 14, 2, 8, 16, 4, 9, 13, 6, 8, 4, 6, 3, 2, 4];
        let v6: Vec<i64> = vec![378, 381, 286, 224, 258, 237, 190, 140, 370, 497, 346, 277, 223, 251, 266, 129];

        let d: Matrix = Matrix::with_array(v4, 4, 4);
        let e: Matrix = Matrix::with_array(v5, 4, 4);
        let f: Matrix = Matrix::with_array(v6, 4, 4);

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
        let mut v1: Vec<i64> = Vec::with_capacity(n);
        let mut v2: Vec<i64> = Vec::with_capacity(n);

        let mut rng = thread_rng();

        for _ in 0..n {
            v1.push(rng.gen::<i64>() % 1000000);
            v2.push(rng.gen::<i64>() % 1000000);
        }

        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, cols, rows);

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
        let v1: Vec<i64> = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 10, 10, 3, 3, 10, 10];
        let mut v2: Vec<i64> = Vec::with_capacity(4);
        let v3: Vec<i64> = vec![11, 11, 11, 11];
        let a: Matrix = Matrix::with_array(v1, 4, 4);

        __submatrix_add (&mut v2, &a, 0, 0, 2, 2, 2);

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
        let v1: Vec<i64> = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 10, 10, 3, 3, 10, 10];
        let mut v2: Vec<i64> = Vec::with_capacity(4);
        let v3: Vec<i64> = vec![9, 9, 9, 9];
        let a: Matrix = Matrix::with_array(v1, 4, 4);

        __submatrix_sub (&mut v2, &a, 2, 2, 0, 0, 2);

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
        let v1: Vec<i64> = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 10, 10, 3, 3, 10, 10];
        let mut v2: Vec<i64> = Vec::with_capacity(4);
        let mut v3: Vec<i64> = Vec::with_capacity(4);
        let v4: Vec<i64> = vec![1, 1, 1, 1];
        let v5: Vec<i64> = vec![10, 10, 10, 10];
        let a: Matrix = Matrix::with_array(v1, 4, 4);

        __submatrix_cpy(&mut v2, &a, 0, 0, 2);
        __submatrix_cpy(&mut v3, &a, 2, 2, 2);

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
        let m11 = Matrix::with_array(vec![1, 1, 1, 1], 2, 2);
        let m12 = Matrix::with_array(vec![2, 2, 2, 2], 2, 2);
        let m21 = Matrix::with_array(vec![3, 3, 3, 3], 2, 2);
        let m22 = Matrix::with_array(vec![10, 10, 10, 10], 2, 2);
        let a = Matrix::with_array(vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 10, 10, 3, 3, 10, 10], 4, 4);

        let b = __reconstitute (&m11, &m12, &m21, &m22, 2, 4);

        assert!(a.eq(&b));
    }
}