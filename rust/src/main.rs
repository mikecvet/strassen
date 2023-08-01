use strassen::matrix::Matrix;

fn 
main () {
    let mut a:Matrix = Matrix::new(2, 2);
    let mut b:Matrix = Matrix::new(2, 2);

    a.random(Some(10));
    b.random(Some(20));
    
    a.mult(&b, strassen::mult::mult_naive);
}
