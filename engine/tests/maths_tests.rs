#[cfg(test)]
mod maths_tests {
    use bricks::maths::Matrix;

    #[test]
    fn test_matrix_add() {
        let mat1: Matrix = Matrix::reshape(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mat2: Matrix = Matrix::reshape(vec![1.0, 2.0, 3.0, 4.0], 2, 2);

        let res = (&mat1 + &mat2);

        assert_eq!(2.0, res.get(0));
        assert_eq!(4.0, res.get(1));
        assert_eq!(6.0, res.get(2));
        assert_eq!(8.0, res.get(3));
    }
}
