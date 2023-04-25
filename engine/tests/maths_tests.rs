#[cfg(test)]
mod maths_tests {
    use bricks::maths::matrix::Matrix;

    #[test]
    fn test_matrix_add() {
        let mat1 : Matrix = Matrix::reshape(vec![1.0,2.0,3.0,4.0], 2, 2).unwrap();
        let mat2 : Matrix = Matrix::reshape(vec![1.0,2.0,3.0,4.0], 2, 2).unwrap();

        let res = (&mat1 + &mat2).unwrap();

        assert_eq!(2.0, res.get(0).unwrap());
        assert_eq!(4.0, res.get(1).unwrap());
        assert_eq!(6.0, res.get(2).unwrap());
        assert_eq!(8.0, res.get(3).unwrap());
    }
}