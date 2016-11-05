use super::Matrix;
use ::vector::{Vector, ArrayLen};

use typenum;
use typenum::consts::*;
use typenum::uint::UInt;
use typenum::operator_aliases::Diff;
use num;
use arrayvec::ArrayVec;

use std::ops::{Add, Mul, Sub, Div, Neg};

pub trait Determinant {
    type Output;

    /// Computes determinant of the matrix.
    fn determinant(&self) -> Self::Output;
}

pub trait Cofactor {
    type Output;

    /// Computes cofactor of the matrix.
    fn cofactor(&self, i: usize, j: usize) -> Self::Output;
}

impl<T> Determinant for Matrix<T, U1, U1> where T: Clone {
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        self[(0, 0)].clone()
    }
}

impl<T> Determinant for Matrix<T, U2, U2>
    where T: num::Signed + Clone
{
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        self[(0, 0)].clone() * self[(1,1)].clone()
            - self[(1, 0)].clone() * self[(0, 1)].clone()
    }
}

impl<T> Determinant for Matrix<T, U3, U3>
    where T: num::Signed + Clone
{
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        self[(0, 0)].clone() * self[(1, 1)].clone() * self[(2, 2)].clone() +
            self[(0, 1)].clone() * self[(1, 2)].clone() * self[(2, 0)].clone() +
            self[(0, 2)].clone() * self[(1, 0)].clone() * self[(2, 1)].clone() -
            self[(0, 2)].clone() * self[(1, 1)].clone() * self[(2, 0)].clone() -
            self[(0, 1)].clone() * self[(1, 0)].clone() * self[(2, 2)].clone() -
            self[(0, 0)].clone() * self[(1, 2)].clone() * self[(2, 1)].clone()
    }
}

// Determinant for Matrix<T, N, N> where N > U3
// Note: U3 = UInt<UInt<UTerm, B1>, B1>
impl<T, U, Ba, Bb, Bc> Determinant for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,
        T: num::Signed + Clone,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>:
            ArrayLen<T> + ArrayLen<Vector<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>> +
            Sub<U1> + for<'a> ArrayLen<&'a T>,
        Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>: Cofactor<Output = T>,
{
    type Output = T;

    fn determinant(&self) -> T {
        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();
        (0 .. n)
            .map(|i| self[(i, 0)].clone() * self.cofactor(i, 0))
            .fold(T::zero(), Add::add)
    }
}

impl<T> Cofactor for Matrix<T, U2, U2> where T: num::Signed + Clone
{
    type Output = T;

    #[inline]
    fn cofactor(&self, i: usize, j: usize) -> T {
        assert!(i < 2);
        assert!(j < 2);

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn * self[(1 - i, 1 - j)].clone()
    }
}

impl<T> Cofactor for Matrix<T, U3, U3>
    where
        U3: Mul + for<'a> ArrayLen<&'a T>,
        T: num::Signed + Clone,
{
    type Output = T;

    fn cofactor(&self, i: usize, j: usize) -> T {
        assert!(i < 3 && j < 3);

        let mut arr = ArrayVec::new();

        for (ii, row) in self.rows_iter().enumerate() {
            if ii == i { continue; }

            for (jj, a) in row.into_iter().enumerate() {
                if jj == j { continue; }
                arr.push(a.clone());
            }
        }

        debug_assert!(arr.is_full());

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn *
            Matrix::<T, U2, U2>::from(Vector::new(arr
                                      .into_inner()
                                      .unwrap_or_else(|_| unreachable!())))
                .determinant()
    }
}

// Cofactor for Matrix<T, N, N> where N > U3
impl<T, U, Ba, Bb, Bc> Cofactor for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,

        T: num::Signed + Clone,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>:
            ArrayLen<T> + ArrayLen<Vector<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>> +
            Sub<U1> + for<'a> ArrayLen<&'a T>,

        Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>: ArrayLen<T> +
            ArrayLen<Vector<T, Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>>> +
            ArrayLen<<Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1> as ArrayLen<T>>::Array>,
        Matrix<T,
               Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>,
               Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>>: Determinant<Output = T>,
{
    type Output = T;

    fn cofactor(&self, i: usize, j: usize) -> T {
        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();
        assert!(i < n && j < n);

        let mut arr = ArrayVec::new();

        for (ii, row) in self.rows_iter().enumerate() {
            if ii == i { continue; }

            let mut subrow = ArrayVec::new();

            for (jj, a) in row.into_iter().enumerate() {
                if jj == j { continue; }
                subrow.push(a);
            }

            arr.push(subrow.into_inner().unwrap_or_else(|_| unreachable!()));
        }

        debug_assert!(arr.is_full());

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn *
            Matrix::<T,
                     Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>,
                     Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>>::new(arr
                                                                     .into_inner()
                                                                     .unwrap_or_else(|_| unreachable!()))
            .determinant()
    }
}

#[test]
fn test_det_cof_impl() {
    let m = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(m.cofactor(0, 0), 4);
    assert_eq!(m.determinant(), -2);
    let m = Matrix::<i32, U3, U3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    assert_eq!(m.determinant(), 0);
    assert_eq!(m.cofactor(1, 1), -12);
    let m = Matrix::<i32, U4, U4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    assert_eq!(m.determinant(), 0);
    let m = Matrix::<i32,  U6,  U6>::new([[1, 2, 3, 4, 5, 6],
                                         [7, 8, 9, 10, 11, 12],
                                         [13, 14, 15, 16, 17, 18],
                                         [19, 20, 21, 22, 23, 24],
                                         [25, 26, 27, 28, 29, 30],
                                         [31, 32, 33, 34, 35, 36]]);
    assert_eq!(m.determinant(), 0);
    let m = Matrix::<i32,  U6,  U6>::new([[1, 1, 1, 1, 1, 63],
                                         [7, 8, 9, 10, 11, 12],
                                         [13, 14, 1, 18, 17, 18],
                                         [19, 80, 81, 22, 23, 20],
                                         [25, 26, 27, 28, 29, 30],
                                         [31, 32, 34, 34, 35, 44]]);
    assert_eq!(m.determinant(), -535680);

    // TODO:
    // let m = m.into_flat_iter().map(|a| BigInt::from(a)).collect::<Matrix::<_, U3, U3>>();
    // assert_eq!(m.determinant(), 0);
}

pub trait Inverse {
    type Output;

    fn inverse(&self) -> Self::Output;
}

impl<T> Inverse for Matrix<T, U1, U1> where T: Clone + num::One + Div<T> {
    type Output = Matrix<<T as Div>::Output, U1, U1>;

    #[inline]
    fn inverse(&self) -> Self::Output {
        Matrix::new([[T::one() / self[(0, 0)].clone()]])
    }
}

impl<T> Inverse for Matrix<T, U2, U2>
    where
        Matrix<T, U2, U2>: Determinant<Output = T>,
        T: Div<T, Output = T> + Neg<Output = T> + Clone,
{
    type Output = Matrix<T, U2, U2>;

    /// Returns inverse of the matrix.
    #[inline]
    fn inverse(&self) -> Self::Output {
        let det = self.determinant();
        Matrix::<T, U2, U2>::new([[self[(1,1)].clone(), -self[(0,1)].clone()], [-self[(1,0)].clone(), self[(0,0)].clone()]]) / det
    }
}

impl<T> Inverse for Matrix<T, U3, U3>
    where
        Matrix<T, U3, U3>: Determinant<Output = T>,
        T: Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Matrix<T, U3, U3>;

    #[inline]
    fn inverse(&self) -> Self::Output {
        use std::marker::PhantomData;

        let det = self.determinant();

        macro_rules! ms {
            (@i $mat:ident : ($i:ident, $j:ident)) => {
                $mat[(PhantomData::<$i>, PhantomData::<$j>)].clone()
            };
            ($mat:ident : $( ($i1:ident, $j1:ident) * ($i2:ident, $j2:ident) - ($i3:ident, $j3:ident) * ($i4:ident, $j4:ident) ),*) => {
                [$(
                    ms!(@i $mat : ($i1, $j1)) * ms!(@i $mat : ($i2, $j2)) - ms!(@i $mat : ($i3, $j3)) * ms!(@i $mat : ($i4, $j4)),
                )*]
            };
        }

        let arr = [
            ms![self : (U1,U1)*(U2,U2)-(U1,U2)*(U2,U1), (U0,U2)*(U2,U1)-(U0,U1)*(U2,U2), (U0,U1)*(U1,U2)-(U0,U2)*(U1,U1)],
            ms![self : (U1,U2)*(U2,U0)-(U1,U0)*(U2,U2), (U0,U0)*(U2,U2)-(U0,U2)*(U2,U0), (U0,U2)*(U1,U0)-(U0,U0)*(U1,U2)],
            ms![self : (U1,U0)*(U2,U1)-(U1,U1)*(U2,U0), (U0,U1)*(U2,U0)-(U0,U0)*(U2,U1), (U0,U0)*(U1,U1)-(U0,U1)*(U1,U0)],
        ];

        Matrix::<T, U3, U3>::new(arr) / det
    }
}

#[test]
fn test_hard_coded_inverse() {
    use num::rational::Ratio;

    let m = Matrix::<i32, U2, U2>::new([[1, 5], [2, 4]]).map(Ratio::from_integer);
    let mi = m.inverse();
    assert_eq!((mi * Ratio::from_integer(6)).map(|a| a.to_integer()), Matrix::new([[-4, 5], [2, -1]]));

    let m = Matrix::<i32, U3, U3>::new([[1, 5, 3], [2, 4, 1], [2, 2, 2]]).map(Ratio::from_integer);
    let mi = m.inverse();
    assert_eq!((mi * Ratio::from_integer(16)).map(|a| a.to_integer()),
               Matrix::new([[-6, 4, 7], [2, 4, -5], [4, -8, 6]]));
}

// {Row, Col} > 3
impl<T, U, Ba, Bb, Bc> Inverse for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,

        T: num::Signed + Clone + ,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>:
            typenum::Unsigned + ArrayLen<T> + ArrayLen<Vector<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>>,
{
    type Output = Self;

    fn inverse(&self) -> Self {
        let mut mat = self.clone();
        let mut inv: Self::Output = Matrix::identity();

        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();

        for i in 0..n {
            let rcp = T::one() / unsafe { mat.get_unchecked((i, i)).clone() };

            for j in 0..n {
                unsafe {
                    *mat.get_unchecked_mut((i, j)) = mat.get_unchecked((i, j)).clone() * rcp.clone();
                    *inv.get_unchecked_mut((i, j)) = inv.get_unchecked((i, j)).clone() * rcp.clone();
                }
            }

            for j in 0..n {
                if i == j { continue; }

                let a = unsafe { mat.get_unchecked((j, i)).clone() };
                for k in 0..n {
                    unsafe {
                        *mat.get_unchecked_mut((j, k)) =
                            mat.get_unchecked((j, k)).clone() -
                            mat.get_unchecked((i, k)).clone() * a.clone();
                        *inv.get_unchecked_mut((j, k)) =
                            inv.get_unchecked((j, k)).clone() -
                            inv.get_unchecked((i, k)).clone() * a.clone();
                    }
                }
            }
        }
        inv
    }
}

#[test]
fn test_inv() {
    use num::rational::Ratio;

    let m = Matrix::<i32, U4, U4>::new([
                                       [1, 5, 3, 1],
                                       [2, 4, 1, 2],
                                       [2, 2, 2, 1],
                                       [1, 5, 1, 1]]).map(Ratio::from_integer);
    let mi = m.inverse();
    assert_eq!((mi * m.determinant()).map(|a| a.to_integer()),
        Matrix::new([
                    [-9, -6, 12, 9],
                    [-1, -2, 0, 5],
                    [6, 0, 0, -6],
                    [8, 16, -12, -16]]));
}
