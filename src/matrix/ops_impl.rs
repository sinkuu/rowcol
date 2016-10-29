use super::Matrix;
use ::vector::ArrayLen;

use typenum;
use typenum::consts::*;
use typenum::uint::UInt;
use typenum::operator_aliases::{Diff, Prod};
use num;
use arrayvec::ArrayVec;

use std::ops::{Add, Mul, Sub, Neg};


/// Internal trait used for determinant implementation dispatch.
pub trait DeterminantImpl {
    type Item;

    fn determinant_impl(&self) -> Self::Item;
}

/// Internal trait used for cofactor implementation dispatch.
pub trait CofactorImpl {
    type Item;

    fn cofactor_impl(&self, i: usize, j: usize) -> Self::Item;
}

impl<T> DeterminantImpl for Matrix<T, U1, U1> where T: Clone {
    type Item = T;

    #[inline]
    fn determinant_impl(&self) -> T {
        self[(0, 0)].clone()
    }
}

impl<T> DeterminantImpl for Matrix<T, U2, U2>
    where T: num::Signed + Clone
{
    type Item = T;

    #[inline]
    fn determinant_impl(&self) -> T {
        self[(0, 0)].clone() * self[(1,1)].clone()
            - self[(1, 0)].clone() * self[(0, 1)].clone()
    }
}

impl<T> DeterminantImpl for Matrix<T, U3, U3>
    where T: num::Signed + Clone
{
    type Item = T;

    #[inline]
    fn determinant_impl(&self) -> T {
        self[(0, 0)].clone() * self[(1, 1)].clone() * self[(2, 2)].clone() +
            self[(0, 1)].clone() * self[(1, 2)].clone() * self[(2, 0)].clone() +
            self[(0, 2)].clone() * self[(1, 0)].clone() * self[(2, 1)].clone() -
            self[(0, 2)].clone() * self[(1, 1)].clone() * self[(2, 0)].clone() -
            self[(0, 1)].clone() * self[(1, 0)].clone() * self[(2, 2)].clone() -
            self[(0, 0)].clone() * self[(1, 2)].clone() * self[(2, 1)].clone()
    }
}

// DeterminantImpl for Matrix<T, N, N> where N > U3
// Note: U3 = UInt<UInt<UTerm, B1>, B1>
impl<T, U, Ba, Bb, Bc> DeterminantImpl for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,
        T: num::Signed + Clone,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>: Mul + typenum::Unsigned,
        Prod<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>: ArrayLen<T>,
        Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>: CofactorImpl<Item = T>,
{
    type Item = T;

    fn determinant_impl(&self) -> T {
        (0 .. <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize())
            .map(|i| self[(i, 0)].clone() + self.cofactor_impl(i, 0))
            .fold(T::zero(), Add::add)
    }
}

impl<T> CofactorImpl for Matrix<T, U2, U2> where T: num::Signed + Clone
{
    type Item = T;

    #[inline]
    fn cofactor_impl(&self, i: usize, j: usize) -> T {
        assert!(i < 2);
        assert!(j < 2);

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn * self[(1 - i, 1 - j)].clone()
    }
}

impl<T> CofactorImpl for Matrix<T, U3, U3>
    where
        U3: Mul + typenum::Unsigned + for<'a> ArrayLen<&'a T>,
        T: num::Signed + Clone,
        // Prod<U3, U3>: ArrayLen<T>,
        // Diff<UInt<U, UInt<Ba, Bb>>, U1>: Mul,
        // Prod<Diff<UInt<U, UInt<Ba, Bb>>, U1>, Diff<UInt<U, UInt<Ba, Bb>>, U1>>: ArrayLen<T>,
        // Matrix<T, Diff<UInt<U, UInt<Ba, Bb>>, U1>, Diff<UInt<U, UInt<Ba, Bb>>, U1>>: DeterminantImpl<Item = T>,
{
    type Item = T;

    fn cofactor_impl(&self, i: usize, j: usize) -> T {
        assert!(i < 3 && j < 3);

        let mut arr = ArrayVec::new();

        for (ii, row) in self.rows_ref().enumerate() {
            if ii == i { continue; }

            for (jj, a) in row.into_iter().enumerate() {
                if jj == j { continue; }
                arr.push(a.clone());
            }
        }

        debug_assert!(arr.is_full());

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn *
            Matrix::<T, U2, U2>::from_flat_array(arr
                                                 .into_inner()
                                                 .unwrap_or_else(|_| unreachable!()))
                .determinant_impl()
    }
}

// CofactorImpl for Matrix<T, N, N> where N > U3
impl<T, U, Ba, Bb, Bc> CofactorImpl for Matrix<T, UInt<UInt<UInt<U, Ba>, Bb>, Bc>, UInt<UInt<UInt<U, Ba>, Bb>, Bc>>
    where
        U: typenum::Unsigned,
        Ba: typenum::Bit,
        Bb: typenum::Bit,
        Bc: typenum::Bit,

        T: num::Signed + Clone,
        UInt<UInt<UInt<U, Ba>, Bb>, Bc>: Mul + typenum::Unsigned + Sub<U1> + for<'a> ArrayLen<&'a T>,
        <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as Mul>::Output: ArrayLen<T>,

        Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1>: Mul,
        <Diff<UInt<UInt<UInt<U, Ba>, Bb>, Bc>, U1> as Mul>::Output: ArrayLen<T>,
        Matrix<T, <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as Sub<U1>>::Output, <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as Sub<U1>>::Output>:
            DeterminantImpl<Item = T>,
{
    type Item = T;

    fn cofactor_impl(&self, i: usize, j: usize) -> T {
        let n = <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as typenum::Unsigned>::to_usize();
        assert!(i < n && j < n);

        let mut arr = ArrayVec::new();

        for (ii, row) in self.rows_ref().enumerate() {
            if ii == i { continue; }

            for (jj, a) in row.into_iter().enumerate() {
                if jj == j { continue; }
                arr.push(a.clone());
            }
        }

        debug_assert!(arr.is_full());

        let sgn = num::pow::pow(-T::one(), i + j);

        sgn * Matrix::<T,
                     <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as Sub<U1>>::Output,
                     <UInt<UInt<UInt<U, Ba>, Bb>, Bc> as Sub<U1>>::Output>::from_flat_array(arr
                                                                                  .into_inner()
                                                                                  .unwrap_or_else(|_| unreachable!()))
                .determinant_impl()
    }
}

#[test]
fn test_det_cof_impl() {
    let m = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(m.cofactor_impl(0, 0), 4);
    assert_eq!(m.determinant_impl(), -2);
    let m = Matrix::<i32, U3, U3>::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    assert_eq!(m.determinant_impl(), 0);
    assert_eq!(m.cofactor_impl(1, 1), -12);
    let m = Matrix::<i32, U4, U4>::new([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    assert_eq!(m.determinant_impl(), 0);

    // TODO:
    // let m = m.into_flat_iter().map(|a| BigInt::from(a)).collect::<Matrix::<_, U3, U3>>();
    // assert_eq!(m.determinant_impl(), 0);
}

