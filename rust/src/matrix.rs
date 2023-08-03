use std::fmt;

/* Floating-point comparison precision */
const EPSILON:f64 = 0.000001;

/**
 * Basic matrix implementation. Metadata with a vector of f64 floats; matrix
 * elements are indexed into the single vector.
 */
pub struct Matrix {
    // Number of rows
    pub rows: usize,

    // Number of columns
    pub cols: usize,

    // Number of elements; rows * cols
    pub n: usize,

    // Underlying matrix data
    pub array: Vec<f64>
}

impl Matrix {

    /**
     * Initializes a new Matrix, wrapping the provided vector. Rows and cols must be nonzero.
     */
    pub fn with_array (array: Vec<f64>, rows: usize, cols: usize) -> Matrix {
        if rows == 0 || cols == 0 {
            panic!("cannot create a null matrix");
        }

        Matrix {
            rows: rows,
            cols: cols,
            n: rows * cols,
            array: array
        }
    }

    /**
     * Creates a new empty matrix of the specified dimensions; underlying vector is initialized to zeroes.
     * Rows and cols must be nonzero.
     */
    pub fn new (rows: usize, cols: usize) -> Matrix {
        if rows == 0 || cols == 0 {
            panic!("cannot create a null matrix");
        }

        return Matrix::with_array(vec![0.0; rows * cols], rows, cols);
    }

    /**
     * Returns a deep copy of this Matrix; clones underlying vector data as well.
     */
    pub fn copy (&self) -> Matrix {
        return Matrix::with_array(self.array.to_vec(), self.rows, self.cols);
    }

    /**
     * Returns the element at (i, j).
     * Unsafe.
     */
    #[inline]
    pub fn at (&self, i: usize, j: usize) -> f64 {
    //    return self.array[i * self.cols + j];
         unsafe {
            return *self.array.get_unchecked(i * self.cols + j);
        }
    }

    /**
     * Returns true if the number of rows in this Matrix equals the number of columns.
     */
    #[inline]
    pub fn is_square (&self) -> bool {
        return self.rows == self.cols;
    }

    /**
     * Adds the contents of `b` to this Matrix, and returns self. Panics if `b` is not the same size as this Matrix.
     */
    pub fn add (&mut self, b: &Matrix) -> &mut Matrix {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                self.array[i] += b.array[i];
            }
        } else {
            panic!("matrices are not the same size! A: [{}, {}], B: [{}, {}]", self.rows, self.cols, b.rows, b.cols);
        }

        return self;
    }

    /**
     * Subtracts the contents of `b` from this Matrix, and returns self. Panics if `b` is not the same size as this Matrix.
     */
    pub fn sub (&mut self, b: &Matrix) -> &mut Matrix {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                self.array[i] -= b.array[i];
            }
        } else {
            panic!("matrices are not the same size! A: [{}, {}], B: [{}, {}]", self.rows, self.cols, b.rows, b.cols);
        }

        return self;
    }

    /**
     * Returns true if all elements of `b` are equal to all elements of this Matrix, subject to a difference of value of `EPSILON` or greater. 
     * Panics if `b` is not the same size as this Matrix.
     */
    pub fn eq (&self, b: &Matrix) -> bool {
        if self.rows == b.rows && self.cols == b.cols {
            for i in 0..self.n {
                let delta = (self.array[i] - b.array[i]).abs();
                if delta > EPSILON {
                    return false;
                }
            }

            return true;
        } else {
            panic!("matrices are not the same size! A: [{}, {}], B: [{}, {}]", self.rows, self.cols, b.rows, b.cols);
        }
    }

    /**
     * Returns a new Matrix with the contents of this Matrix copied into it. If the provided parameter `n` is
     * smaller or equal to the existing number of rows and columns of this matrix, a copy of this matrix
     * is returned.
     * 
     * Otherwise, new a square n x n Matrix is returned, with any added rows or columns filled with zeroes.
     */
    pub fn pad (&self, n: usize) -> Matrix {
        if n <= self.rows && n <= self.cols {
            return self.copy();
        } else {
            // Initialize new vector with expected capacity
            let mut v:Vec<f64> = Vec::with_capacity(n * n);

            for i in 0..self.rows {
                for j in 0..self.cols {
                    v.push(self.at(i, j));
                }

                // These are additional, padded columns
                for _ in self.cols..n {
                    v.push(0.0);
                }
            }

            // These are additional, padded rows
            for _ in self.rows..n {
                for _ in 0..n {
                    v.push(0.0);
                }
            }

            return Matrix::with_array(v, n, n);
        }
    }

    /**
     * Returns a new Matrix which contains a copy of this matrix, reduced to the given
     * number of rows and columns, starting from index [0, 0]. Typically used to remove
     * padding applied during the beginning of Strassen multiplication, to return a matrix
     * back to its original dimensions.
     * Panics if the specified number of rows or columns are larger than this Matrix's number of rows or columns.
     */
    pub fn reduce (&self, rows: usize, cols: usize) -> Matrix {

        if rows > self.rows || cols > self.cols {
            panic!("Tried to reduce self to larger dimensions");
        }

        let mut v:Vec<f64> = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            let indx = i * self.cols;
            v.extend_from_slice(&self.array[indx..(indx + cols)])
        }
    
        return Matrix::with_array(v, rows, cols);
    }

    /**
     * Returns a new Matrix containing the transpose of this Matrix.
     */
    pub fn transpose (&self) -> Matrix {
        let mut v:Vec<f64> = Vec::with_capacity(self.n);

        for i in 0..self.cols {
            for j in 0..self.rows {
                v.push(self.at(j, i));
            }
        }

        return Matrix::with_array(v, self.cols, self.rows)
    }

    /**
     * Multiplies this Matrix by `b`, using the provided `multiplier` function.
     */
    pub fn mult (&self, b: &Matrix, multipler: fn(&Matrix, &Matrix) -> Matrix) -> Matrix {
        return multipler(self, b);
    }

}

/**
 * Pretty-printing for the Matrix struct.
 */
impl fmt::Display for Matrix {
    fn fmt (&self, f: &mut fmt::Formatter) -> fmt::Result {
        let array_string = self.array.iter().map(|i| i.to_string()).collect::<Vec<String>>().join(", ");
        write!(f, "rows: {} cols: {} n: {} array: [{}]", self.rows, self.cols, self.n, array_string)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_eq () {
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let v2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let a: Matrix = Matrix::with_array(v1, 2, 2);
        let b: Matrix = Matrix::with_array(v2, 2, 2);

        assert!(a.eq(&b));
    }
    
    #[test]
    fn test_add () {
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let v2: Vec<f64> = vec![9.0, 8.0, 7.0, 6.0];
        let v3: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0];

        let mut a: Matrix = Matrix::with_array(v1, 2, 2);
        let b: Matrix = Matrix::with_array(v2, 2, 2);

        a.add(&b);

        assert!(a.array.eq(&v3));
        assert!(a.eq(&Matrix::with_array(v3, 2, 2)));
    }

    #[test]
    fn test_sub () {
        let v1: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0];
        let v2: Vec<f64> = vec![6.0, 7.0, 8.0, 9.0];
        let v3: Vec<f64> = vec![4.0, 3.0, 2.0, 1.0];

        let mut a: Matrix = Matrix::with_array(v1, 2, 2);
        let b: Matrix = Matrix::with_array(v2, 2, 2);

        a.sub(&b);

        assert!(a.array.eq(&v3));
        assert!(a.eq(&Matrix::with_array(v3, 2, 2)));
    }

    #[test]
    fn test_at () {
        let rows = 2;
        let cols = 6;
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a: Matrix = Matrix::with_array(v, rows, cols);
        let mut indx = 1;

        for i in 0..rows {
            for j in 0..cols {
                let delta = (a.at(i, j) - indx as f64).abs();
                assert!(delta < 0.000001);
                indx += 1;
            }
        }
    }

    #[test]
    fn test_is_square () {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let a: Matrix = Matrix::with_array(v.to_vec(), 2, 8);
        let b: Matrix = Matrix::with_array(v.to_vec(), 4, 4);
        
        assert!(!a.is_square());
        assert!(b.is_square());
    }

    #[test]
    fn test_transpose_1 () {
        let rows = 2;
        let cols = 6;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let v2: Vec<f64> = vec![1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, cols, rows);

        let c = a.transpose();

        assert!(c.eq(&b));
    }

    #[test]
    fn test_transpose_2 () {
        let rows = 3;
        let cols = 3;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v2: Vec<f64> = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, cols, rows);

        let c = a.transpose();

        assert!(c.eq(&b));
    }

    #[test]
    fn test_pad_cols () {
        let rows = 5;
        let cols = 2;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let v2: Vec<f64> = vec![1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0 ,0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 0.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, 5, 5);

        let c = a.pad(rows);

        assert!(c.eq(&b));
    }

    #[test]
    fn test_pad_rows () {
        let rows = 2;
        let cols = 5;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let v2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, 5, 5);

        let c = a.pad(cols);

        assert!(c.eq(&b));
    }

    #[test]
    fn test_pad_none () {
        let rows = 2;
        let cols = 5;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);

        let c = a.pad(1);

        assert!(c.eq(&a));
    }

    #[test]
    fn test_reduce () {
        let rows = 5;
        let cols = 5;
        let v1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0];
        let a: Matrix = Matrix::with_array(v1, rows, cols);
        let b: Matrix = Matrix::with_array(v2, 2, 4);

        let c = a.reduce(2, 4);

        println!("reduced C: {}", c);

        assert!(c.eq(&b));
    }
}