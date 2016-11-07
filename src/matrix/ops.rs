use super::Matrix;
use ::vector::{Vector, ArrayLen};

use typenum;
use typenum::consts::*;
use typenum::uint::UInt;
use typenum::operator_aliases::Diff;
use num;

use std::ops::{Add, Mul, Sub, Div, Neg};

macro_rules! idx {
    ($mat:ident ( $i:ident, $j:ident )) => {
        $mat[($i::new(), $j::new())].clone()
    }
}

/// Trait for computing determinants of square matrices.
pub trait Determinant {
    type Output;

    /// Computes determinant of the matrix.
    fn determinant(&self) -> Self::Output;
}

/// Trait for computing cofactors of square matrices.
pub trait Cofactor {
    type Output;

    /// Computes cofactor of the matrix.
    fn cofactor(&self, i: usize, j: usize) -> Self::Output;
}

impl<T> Determinant for Matrix<T, U1, U1> where T: Clone {
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        idx!(self(U0, U0))
    }
}

impl<T> Determinant for Matrix<T, U2, U2>
    where
        T: Clone + Sub<T, Output = T> + Mul<T, Output = T>,
{
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        idx!(self(U0,U0)) * idx!(self(U1,U1))
            - idx!(self(U1, U0)) * idx!(self(U0, U1))
    }
}

impl<T> Determinant for Matrix<T, U3, U3>
    where
        T: Clone + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
{
    type Output = T;

    #[inline]
    fn determinant(&self) -> T {
        idx!(self(U0,U0)) * idx!(self(U1,U1)) * idx!(self(U2,U2)) +
            idx!(self(U0,U1)) * idx!(self(U1,U2)) * idx!(self(U2,U0)) +
            idx!(self(U0,U2)) * idx!(self(U1,U0)) * idx!(self(U2,U1)) -
            idx!(self(U0,U2)) * idx!(self(U1,U1)) * idx!(self(U2,U0)) -
            idx!(self(U0,U1)) * idx!(self(U1,U0)) * idx!(self(U2,U2)) -
            idx!(self(U0,U0)) * idx!(self(U1,U2)) * idx!(self(U2,U1))
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
        T: num::Float + Clone + ::std::fmt::Display,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>:
            ArrayLen<T> + ArrayLen<Vector<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>> +
            Sub<U1> + for<'a> ArrayLen<&'a T> + ArrayLen<usize> + ArrayLen<Vector<(String, usize), UInt<UInt<UInt<U, Ba>, Bb>, Bc>>> +
            ArrayLen<(String, usize)>,
        Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>: Cofactor<Output = T>,
{
    type Output = T;

    fn determinant(&self) -> T {
        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();

        let mut mat = self.clone();

        if (0..n).any(|i| mat[(i, i)].is_zero()) {
            // cannot create triangular matrix, falling back
            return (0 .. n)
                .map(|i| {
                    self[(i, 0)].clone() * self.cofactor(i, 0)
                })
                .fold(T::zero(), Add::add);
        }

        // http://thira.plavox.info/blog/2008/06/_c.html
        for i in 0..n {
            for j in (i+1)..n {
                let d = mat[(j, i)].clone() / mat[(i, i)].clone();

                for k in 0..n {
                    let sub = mat[(j, k)].clone() -
                        mat[(i, k)].clone() * d.clone();
                    mat[(j, k)] = sub;
                }
            }
        }

        (0..n).map(|i| mat[(i, i)].clone()).fold(T::one(), Mul::mul)
    }
}

impl<T> Cofactor for Matrix<T, U2, U2> where T: num::Signed + Clone
{
    type Output = T;

    #[inline]
    fn cofactor(&self, i: usize, j: usize) -> T {
        let sgn = if (i + j) % 2 == 0 { T::one() } else { T::zero() };

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

        let sgn = if (i + j) % 2 == 0 { T::one() } else { T::zero() };

        let arr = self.rows_iter_ref().enumerate().filter(|&(ii, _)| ii != i).map(|(_, row)| {
            row.into_iter().enumerate()
                .filter(|&(jj, _)| jj != j).map(|(_, a)| a.clone()).collect()
        }).collect();

        sgn * Matrix::<T, U2, U2>(arr).determinant()
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

        let arr = self.rows_iter_ref().enumerate().filter(|&(ii, _)| ii != i).map(|(_, row)| {
                row.into_iter()
                    .enumerate()
                    .filter(|&(jj, _)| jj != j)
                    .map(|(_, a)| a.clone())
                    .collect::<Vector<T, _>>()
            }).collect();

        let sgn = if (i + j) % 2 == 0 { T::one() } else { T::zero() };

        sgn *
            Matrix::<T,
                     Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>,
                     Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>>(arr)
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
    let m = Matrix::<f32, U4, U4>::new([[1.0, 2.0, 3.0, 4.0],
                                       [5.0, 6.0, 7.0, 8.0],
                                       [9.0, 10.0, 11.0, 12.0],
                                       [13.0, 14.0, 15.0, 16.0]]);
    relative_eq!(m.determinant(), 0.0);
    let m = Matrix::<f32,  U6,  U6>::new([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                                         [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                                         [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                                         [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                                         [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]]);
    relative_eq!(m.determinant(), 0.0);
    let m = Matrix::<f32,  U6,  U6>::new([[1.0, 1.0, 1.0, 1.0, 1.0, 63.0],
                                         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                                         [13.0, 14.0, 1.0, 18.0, 17.0, 18.0],
                                         [19.0, 80.0, 81.0, 22.0, 23.0, 20.0],
                                         [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                                         [31.0, 32.0, 34.0, 34.0, 35.0, 44.0]]);
    relative_eq!(m.determinant(), -535680.0);
}

pub trait Inverse {
    type Output;

    /// Returns the inverse of this matrix.
    fn inverse(&self) -> Option<Self::Output>;
}

impl<T> Inverse for Matrix<T, U1, U1> where T: Clone + num::Zero + num::One + Div<T, Output = T> {
    type Output = Matrix<T, U1, U1>;

    #[inline]
    fn inverse(&self) -> Option<Self::Output> {
        let d = idx!(self(U0,U0));
        if d.is_zero() {
            None
        } else {
            Some(Matrix::new([[T::one() / d]]))
        }
    }
}

impl<T> Inverse for Matrix<T, U2, U2>
    where
        Matrix<T, U2, U2>: Determinant<Output = T>,
        T: num::Zero + Div<T, Output = T> + Neg<Output = T> + Clone,
{
    type Output = Matrix<T, U2, U2>;

    #[inline]
    fn inverse(&self) -> Option<Self::Output> {
        let det = self.determinant();
        if det.is_zero() {
            None
        } else {
            Some(Matrix::<T, U2, U2>::new([[idx!(self(U1,U1)), -idx!(self(U0,U1))],
                                     [-idx!(self(U1,U0)), idx!(self(U0,U0))]]) / det)
        }
    }
}

impl<T> Inverse for Matrix<T, U3, U3>
    where
        Matrix<T, U3, U3>: Determinant<Output = T>,
        T: num::Zero + Mul<T, Output = T> + Div<T, Output = T> + Sub<T, Output = T> + Clone,
{
    type Output = Matrix<T, U3, U3>;

    #[inline]
    fn inverse(&self) -> Option<Self::Output> {
        let det = self.determinant();

        if det.is_zero() {
            None
        } else {
            macro_rules! ms {
                ($mat:ident : $( ($i1:ident, $j1:ident) * ($i2:ident, $j2:ident) -
                                 ($i3:ident, $j3:ident) * ($i4:ident, $j4:ident) ),*) => {
                    [$(
                        idx!($mat($i1, $j1)) * idx!($mat($i2, $j2)) -
                        idx!($mat($i3, $j3)) * idx!($mat($i4, $j4)),
                    )*]
                };
            }

            let arr = [
                ms![self : (U1,U1)*(U2,U2)-(U1,U2)*(U2,U1), (U0,U2)*(U2,U1)-(U0,U1)*(U2,U2),
                    (U0,U1)*(U1,U2)-(U0,U2)*(U1,U1)],
                ms![self : (U1,U2)*(U2,U0)-(U1,U0)*(U2,U2), (U0,U0)*(U2,U2)-(U0,U2)*(U2,U0),
                    (U0,U2)*(U1,U0)-(U0,U0)*(U1,U2)],
                ms![self : (U1,U0)*(U2,U1)-(U1,U1)*(U2,U0), (U0,U1)*(U2,U0)-(U0,U0)*(U2,U1),
                    (U0,U0)*(U1,U1)-(U0,U1)*(U1,U0)],
            ];

            Some(Matrix::<T, U3, U3>::new(arr) / det)
        }
    }
}

#[test]
fn test_hard_coded_inverse() {
    use num::rational::Ratio;
    use num::{BigInt, ToPrimitive};

    let m = Matrix::<i32, U2, U2>::new([[1, 5], [2, 4]]).map(|x| Ratio::from_integer(BigInt::from(x)));
    let mi = m.inverse().unwrap();
    assert_eq!((mi * Ratio::from_integer(BigInt::from(6))).map(|a| a.to_integer().to_i32().unwrap()),
        Matrix::new([[-4, 5], [2, -1]]));

    let m = Matrix::<i32, U3, U3>::new([[1, 5, 3], [2, 4, 1], [2, 2, 2]]).map(Ratio::from_integer);
    let mi = m.inverse().unwrap();
    assert_eq!((mi * Ratio::from_integer(16)).map(|a| a.to_integer()),
               Matrix::new([[-6, 4, 7], [2, 4, -5], [4, -8, 6]]));
}

// {Row, Col} > 3
// TODO: specialize for Matrix<T, U4, U4>
impl<T, U, Ba, Bb, Bc> Inverse for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,

        T: Clone + num::Signed,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>:
            typenum::Unsigned + ArrayLen<T> + ArrayLen<Vector<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>>,
{
    type Output = Self;

    fn inverse(&self) -> Option<Self> {
        let mut mat = self.clone();
        let mut inv: Self::Output = Matrix::identity();

        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();

        for i in 0..n {
            let rcp = {
                let d = mat[(i, i)].clone();
                if d.is_zero() {
                    return None;
                }
                T::one() / d
            };

            for j in 0..n {
                mat[(i, j)] = mat[(i, j)].clone() * rcp.clone();
                inv[(i, j)] = inv[(i, j)].clone() * rcp.clone();
            }

            for j in 0..n {
                if i == j { continue; }

                let a = mat[(j, i)].clone();
                for k in 0..n {
                    mat[(j, k)] =
                        mat[(j, k)].clone() -
                        mat[(i, k)].clone() * a.clone();
                    inv[(j, k)] =
                        inv[(j, k)].clone() -
                        inv[(i, k)].clone() * a.clone();
                }
            }
        }

        Some(inv)
    }
}

#[test]
fn test_inverse() {
    use num::rational::Ratio;

    let m = Matrix::<i32, U4, U4>::new([
                                       [1, 5, 3, 1],
                                       [2, 4, 1, 2],
                                       [2, 2, 2, 1],
                                       [1, 5, 1, 1]]).map(Ratio::from_integer);
    let mi = m.inverse().unwrap();
    assert_eq!((mi * Ratio::from_integer(12)).map(|a| a.to_integer()),
        Matrix::new([
                    [-9, -6, 12, 9],
                    [-1, -2, 0, 5],
                    [6, 0, 0, -6],
                    [8, 16, -12, -16]]));
}
