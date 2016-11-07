use super::Matrix;
use ::vector::{Vector, ArrayLen};
#[cfg(test)] use typenum::consts::*;
use num;

use std::ops::{Add, Sub, Mul, Div, Neg,
    AddAssign, SubAssign, MulAssign, DivAssign};

macro_rules! impl_matrix_arith {
    (T T : $op_trait:ident, $op_fn: ident) => {
        impl<T, Row, Col> $op_trait<Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait,
                Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as $op_trait>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<T as $op_trait>::Output>,
        {
            type Output = Matrix<<T as $op_trait>::Output, Row, Col>;

            #[inline]
            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                Matrix($op_trait::$op_fn(self.0, rhs.0))
            }
        }
    };

    (&T T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<Matrix<T, Row, Col>> for &'a Matrix<T, Row, Col>
            where
                &'a T: $op_trait<T>,
                Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<&'a T as $op_trait<T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<&'a T as $op_trait<T>>::Output>,
        {
            type Output = Matrix<<&'a T as $op_trait<T>>::Output, Row, Col>;

            #[inline]
            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                Matrix($op_trait::$op_fn(&self.0, rhs.0))
            }
        }
    };

    (T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait<&'a T>,
                Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as $op_trait<&'a T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<T as $op_trait<&'a T>>::Output>,
        {
            type Output = Matrix<<T as $op_trait<&'a T>>::Output, Row, Col>;

            #[inline]
            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                Matrix($op_trait::$op_fn(self.0, &rhs.0))
            }
        }
    };

    (&T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, 'b, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for &'b Matrix<T, Row, Col>
            where
                &'b T: $op_trait<&'a T>,
                Row: ArrayLen<Vector<T, Col>> +
                     ArrayLen<Vector<<&'b T as $op_trait<&'a T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<&'b T as $op_trait<&'a T>>::Output>,
        {
            type Output = Matrix<<&'b T as $op_trait<&'a T>>::Output, Row, Col>;

            #[inline]
            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                Matrix($op_trait::$op_fn(&self.0, &rhs.0))
            }
        }
    };
}

impl_matrix_arith!(T T: Add, add);
impl_matrix_arith!(T T: Sub, sub);
impl_matrix_arith!(&T T: Add, add);
impl_matrix_arith!(&T T: Sub, sub);
impl_matrix_arith!(T &T: Add, add);
impl_matrix_arith!(T &T: Sub, sub);
impl_matrix_arith!(&T &T: Add, add);
impl_matrix_arith!(&T &T: Sub, sub);

impl<T, U, Row, Col> AddAssign<Matrix<U, Row, Col>> for Matrix<T, Row, Col>
    where
        T: AddAssign<U>,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<U, Col>>,
        Col: ArrayLen<T> + ArrayLen<U>,
{
    fn add_assign(&mut self, rhs: Matrix<U, Row, Col>) {
        for (lrow, rrow) in self.0.iter_mut().zip(rhs.0) {
            for (l, r) in lrow.iter_mut().zip(rrow) {
                *l += r;
            }
        }
    }
}

impl<T, U, Row, Col> SubAssign<Matrix<U, Row, Col>> for Matrix<T, Row, Col>
    where
        T: SubAssign<U>,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<U, Col>>,
        Col: ArrayLen<T> + ArrayLen<U>,
{
    fn sub_assign(&mut self, rhs: Matrix<U, Row, Col>) {
        for (lrow, rrow) in self.0.iter_mut().zip(rhs.0) {
            for (l, r) in lrow.iter_mut().zip(rrow) {
                *l -= r;
            }
        }
    }
}

impl<T, Row, Col> Neg for Matrix<T, Row, Col>
    where
        T: Neg,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as Neg>::Output, Col>>,
        Col: ArrayLen<T> + ArrayLen<<T as Neg>::Output>,
{
    type Output = Matrix<<T as Neg>::Output, Row, Col>;

    fn neg(self) -> Self::Output {
        Matrix(self.0.into_iter().map(|v| -v).collect())
    }
}

// matrix * x

impl<T, Row, Col> Mul<T> for Matrix<T, Row, Col>
    where
        T: Mul<Output = T> + Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Output = Matrix<T, Row, Col>;

    fn mul(self, rhs: T) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| row * rhs.clone()).collect())
    }
}

impl<'a, T, Row, Col> Mul<&'a T> for Matrix<T, Row, Col>
    where
        T: Mul<&'a T, Output = T>,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Output = Matrix<T, Row, Col>;

    fn mul(self, rhs: &'a T) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| row * rhs).collect())
    }
}

impl<T, Row, Col> MulAssign<T> for Matrix<T, Row, Col>
    where
        T: MulAssign + Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        for row in self.0.iter_mut() {
            for a in row.iter_mut() {
                *a *= rhs.clone();
            }
        }
    }
}

// matrix * matrix

impl<T, N, LRow, RCol> Mul<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: Clone + num::Zero + Add<T, Output = T> + Mul<T, Output = T>,
        N: ArrayLen<Vector<T, RCol>> +
           ArrayLen<T>,
        RCol: ArrayLen<T>,
        LRow: ArrayLen<Vector<T, RCol>> + ArrayLen<Vector<T, N>>,
{
    type Output = Matrix<T, LRow, RCol>;

    fn mul(self, rhs: Matrix<T, N, RCol>) -> Self::Output {
        Matrix(self.0.into_iter().map(|lrow| {
            rhs.cols_iter().map(|rcol| {
                lrow.iter().cloned()
                    .zip(rcol).map(|(a, b)| a * b).fold(T::zero(), Add::add)
            }).collect()
        }).collect())
    }
}

impl<'a, T, N, LRow, RCol> Mul<&'a Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: Clone + num::Zero + Add<T, Output = T> + Mul<&'a T, Output = T>,
        N: ArrayLen<Vector<T, RCol>> +
           ArrayLen<T>,
        RCol: ArrayLen<T>,
        LRow: ArrayLen<Vector<T, RCol>> + ArrayLen<Vector<T, N>>,
{
    type Output = Matrix<T, LRow, RCol>;

    fn mul(self, rhs: &'a Matrix<T, N, RCol>) -> Self::Output {
        Matrix(self.0.into_iter().map(|lrow| {
            rhs.cols_iter_ref().map(|rcol| {
                lrow.iter().cloned()
                    .zip(rcol).map(|(a, b)| a * b).fold(T::zero(), Add::add)
            }).collect()
        }).collect())
    }
}

impl<T, N> MulAssign<Matrix<T, N, N>> for Matrix<T, N, N>
    where
        T: Clone + num::Zero + Add<T, Output = T> + Mul<T, Output = T>,
        N: ArrayLen<Vector<T, N>> + ArrayLen<T>,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, N>) {
        *self = Matrix(self.0.iter().map(|lrow| {
            rhs.cols_iter().map(|rcol| {
                lrow.iter().cloned()
                    .zip(rcol).map(|(a, b)| a * b).fold(T::zero(), Add::add)
            }).collect()
        }).collect())
    }
}

#[test]
fn test_mul_assign_mat() {
    let mut m = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    m *= Matrix::<i32, U2, U2>::new([[5, 6], [7, 8]]);
    assert_eq!(m, Matrix::new([[19, 22], [43, 50]]));
}

// matrix * vector

impl<T, Row, Col> Mul<Vector<T, Col>> for Matrix<T, Row, Col>
    where
        T: Clone + Mul<T, Output = T> + Add<T, Output = T> + num::Zero,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    type Output = Vector<T, Row>;

    fn mul(self, rhs: Vector<T, Col>) -> Self::Output {
        self.0.into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(rhs.iter().cloned())
                    .map(|(a, b)| a * b).fold(T::zero(), Add::add)
            })
            .collect()
    }
}

impl<'a, T, Row, Col> Mul<&'a Vector<T, Col>> for Matrix<T, Row, Col>
    where
        T: Mul<&'a T, Output = T> + Add<T, Output = T> + num::Zero,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    type Output = Vector<T, Row>;

    fn mul(self, rhs: &'a Vector<T, Col>) -> Self::Output {
        self.0.into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(rhs.iter())
                    .map(|(a, b)| a * b).fold(T::zero(), Add::add)
            })
            .collect()
    }
}

// matrix / x

impl<T, Row, Col> Div<T> for Matrix<T, Row, Col>
    where
        T: Clone + Div<T, Output = T>,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Output = Matrix<T, Row, Col>;

    fn div(self, rhs: T) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| {
            row / rhs.clone()
        }).collect())
    }
}

impl<'a, T, Row, Col> Div<&'a T> for Matrix<T, Row, Col>
    where
        T: Div<&'a T, Output = T>,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Output = Matrix<T, Row, Col>;

    fn div(self, rhs: &'a T) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| {
            row / &rhs
        }).collect())
    }
}

impl<T, Row, Col> DivAssign<T> for Matrix<T, Row, Col>
    where
        T: DivAssign + Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn div_assign(&mut self, rhs: T) {
        for row in self.0.iter_mut() {
            for a in row.iter_mut() {
                *a /= rhs.clone();
            }
        }
    }
}


#[test]
fn test_matrix_arith() {
    let mut m1 = Matrix::<i32, U2, U2>::default();
    m1[(1,1)] = 4;
    m1[(0,0)] = 1;

    let m2 = Matrix::new([[0, 0], [0, 1]]);
    assert_eq!(m2[(1, 1)], 1);

    let m3 = m1 + m2;
    assert_eq!(m3, Matrix::new([[1, 0], [0, 5]]));
    assert_eq!(m3[(0,0)], 1);
    assert_eq!(m3[(1,1)], 5);
    assert_eq!(-m3, Matrix::new([[-1, 0], [0, -5]]));

    assert_eq!(&m1 + m2, Matrix::new([[1, 0], [0, 5]]));
    assert_eq!(m1 + &m2, Matrix::new([[1, 0], [0, 5]]));
    assert_eq!(&m1 + &m2, Matrix::new([[1, 0], [0, 5]]));

    let m4 = m1 - m2;
    assert_eq!(m4, Matrix::new([[1, 0], [0, 3]]));
    assert_eq!(m4[(0,0)], 1);
    assert_eq!(m4[(1,1)], 3);

    assert_eq!(&m1 - m2, Matrix::new([[1, 0], [0, 3]]));
    assert_eq!(m1 - &m2, Matrix::new([[1, 0], [0, 3]]));
    assert_eq!(&m1 - &m2, Matrix::new([[1, 0], [0, 3]]));

    let id = Matrix::identity();
    assert_eq!(m1, m1 * id);
    assert_eq!(m1 * num::pow::pow(id, 20), m1);

    let m5 = Matrix::<i32, U2, U1>::new([[1], [3]]);
    assert_eq!(m5[(1,0)], 3);
    assert_eq!((m5 + m5 - m5)[(1,0)], 3);
    assert_eq!(m4 * m5 * 2, Matrix::new([[2], [18]]));
    assert_eq!(m4 * &m5 * 2 / 2, Matrix::new([[1], [9]]));
    assert_eq!(m5 * &Vector::<i32, U1>::new([2]), Vector::new([2, 6]));

    let mut m = m4;
    m *= 2;
    m = m * &2;
    m /= 2;
    m = m / &2;
    assert_eq!(m, m4);

    let mb = Matrix::<num::BigInt, U2, U2>::new([[1.into(), 2.into()], [3.into(), 4.into()]]);
    assert_eq!(mb.clone() * mb, Matrix::new([[7.into(), 10.into()], [15.into(), 22.into()]]));
}


