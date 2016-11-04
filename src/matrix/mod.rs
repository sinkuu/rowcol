mod ops_impl;
pub use self::ops_impl::{Cofactor, Determinant};

use typenum::{self, Prod, Same, Mod};
use typenum::consts::*;

use arrayvec::ArrayVec;

use num;

use std::ops::{Add, Sub, Mul, Rem, Index, IndexMut};
use std::marker::PhantomData;
use std::fmt::{Debug, Formatter};
use std::fmt::Result as FmtResult;

use vector::{Vector, ArrayLen};

/// A fixed-size matrix allocated on the stack.
pub struct Matrix<T, Row, Col>(Vector<Vector<T, Col>, Row>)
    where
        Col: ArrayLen<T>,
        Row: ArrayLen<Vector<T, Col>>;

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: ArrayLen<<Col as ArrayLen<T>>::Array> + ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    /// Creates a new matrix.
    ///
    /// ```rust
    /// # use rowcol::typenum::consts::*;
    /// # use rowcol::Matrix;
    /// let mat = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    /// ```
    pub fn new(rows: <Row as ArrayLen<<Col as ArrayLen<T>>::Array>>::Array)
        -> Matrix<T, Row, Col>
    {
        let mut arr = ArrayVec::new();
        for row in Vector::<<Col as ArrayLen<T>>::Array, Row>::new(rows) {
            arr.push(Vector::<T, Col>::new(row));
        }
        debug_assert!(arr.is_full());
        Matrix(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}

impl<T, Row, Col> From<Vector<T, Prod<Row, Col>>> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + Mul<Col>,
        Col: ArrayLen<T>,
        Prod<Row, Col>: ArrayLen<T> + Rem<Col>,
        Mod<Prod<Row, Col>, Col>: Same<U0>,
{
    #[inline]
    fn from(v: Vector<T, Prod<Row, Col>>) -> Self {
        let mut res = ArrayVec::new();
        for row in v.into_chunks::<Col>() {
            res.push(row);
        }
        debug_assert!(res.is_full());
        Matrix(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (Row::to_usize(), Col::to_usize())
    }

    #[inline]
    pub fn rows(&self) -> usize {
        Row::to_usize()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        Col::to_usize()
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T> + ArrayLen<Vector<T, Row>>,
{
    pub fn transposed(&self) -> Matrix<T, Col, Row> {
        let arr: ArrayVec<_> = self.cols_iter().collect();
        Matrix(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}

#[test]
fn test_transposed() {
    let mat = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(mat.transposed(), Matrix::new([[1, 3], [2, 4]]));
}

impl<T, Row, Col> Default for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        <Col as ArrayLen<T>>::Array: Default,
        <Row as ArrayLen<Vector<T, Col>>>::Array: Default,
{
    #[inline]
    fn default() -> Self {
        Matrix(Default::default())
    }
}

impl<T, Row, Col> Clone for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        Vector<Vector<T, Col>, Row>: Clone
{
    fn clone(&self) -> Self {
        Matrix::<T, Row, Col>(self.0.clone())
    }
}

impl<T, Row, Col> Copy for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        T: Copy,
        <Row as ArrayLen<Vector<T, Col>>>::Array: Copy,
        <Col as ArrayLen<T>>::Array: Copy
{
}

impl<T, Row, Col> PartialEq for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        T: PartialEq,
{
    fn eq(&self, rhs: &Matrix<T, Row, Col>) -> bool {
        self.0 == rhs.0
    }
}

impl<T, Row, Col> Eq for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        T: PartialEq,
{
}

impl<T, Row, Col> Debug for Matrix<T, Row, Col>
    where
        T: Debug,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        try!(write!(fmt, "Matrix["));

        let last = self.0.len() - 1;
        for (i, row) in self.0.iter().enumerate() {
            try!(row.as_slice().fmt(fmt));
            if i != last { try!(write!(fmt, ", ")); }
        }

        write!(fmt, "]")
    }
}

#[test]
fn test_debug_matrix() {
    let m = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(format!("{:?}", m), "Matrix[[1, 2], [3, 4]]");
}

impl<T, Row, Col> Index<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.0[i][j]
    }
}

impl<T, Row, Col, IRow, ICol> Index<(PhantomData<IRow>, PhantomData<ICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        IRow: typenum::Unsigned + typenum::Cmp<Row>,
        ICol: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<IRow, Row>: Same<typenum::Less>,
        typenum::Compare<ICol, Col>: Same<typenum::Less>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: (PhantomData<IRow>, PhantomData<ICol>)) -> &T {
        unsafe {
            self.0
                .get_unchecked(IRow::to_usize())
                .get_unchecked(ICol::to_usize())
        }
    }
}

impl<T, Row, Col> IndexMut<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.0.as_slice_mut()[i].as_slice_mut()[j]
    }
}

impl<T, Row, Col, IRow, ICol> IndexMut<(PhantomData<IRow>, PhantomData<ICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        IRow: typenum::Unsigned + typenum::Cmp<Row>,
        ICol: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<IRow, Row>: Same<typenum::Less>,
        typenum::Compare<ICol, Col>: Same<typenum::Less>,
{
    #[inline]
    fn index_mut(&mut self, _: (PhantomData<IRow>, PhantomData<ICol>)) -> &mut T {
        unsafe {
            self.0.get_unchecked_mut(IRow::to_usize())
                .get_unchecked_mut(ICol::to_usize())
        }
    }
}

macro_rules! impl_matrix_arith {
    (T T : $op_trait:ident, $op_fn: ident) => {
        impl<T, Row, Col> $op_trait<Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait,
                Row: Mul<Col> + ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as $op_trait>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<T as $op_trait>::Output>,
                Prod<Row, Col>: ArrayLen<<T as $op_trait>::Output> + Rem<Col>,
                Mod<Prod<Row, Col>, Col>: Same<U0>,
        {
            type Output = Matrix<<T as $op_trait>::Output, Row, Col>;

            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (xr, yr) in self.0.into_iter().zip(rhs.0.into_iter()) {
                    for (x, y) in xr.into_iter().zip(yr) {
                        res.push($op_trait::$op_fn(x, y));
                    }
                }

                Matrix::from(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
            }
        }
    };

    (&T T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<Matrix<T, Row, Col>> for &'a Matrix<T, Row, Col>
            where
                &'a T: $op_trait<T>,
                Row: Mul<Col> +
                    ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<&'a T as $op_trait<T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<&'a T as $op_trait<T>>::Output>,
                Prod<Row, Col>: ArrayLen<<&'a T as $op_trait<T>>::Output> + Rem<Col>,
                Mod<Prod<Row, Col>, Col>: Same<U0>,
        {
            type Output = Matrix<<&'a T as $op_trait<T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (xr, yr) in self.0.as_slice().iter().zip(rhs.0.into_iter()) {
                    for (x, y) in xr.iter().zip(yr.into_iter()) {
                        res.push($op_trait::$op_fn(x, y));
                    }
                }

                Matrix::from(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
            }
        }
    };

    (T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait<&'a T>,
                Row: Mul<Col> +
                    ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as $op_trait<&'a T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<T as $op_trait<&'a T>>::Output>,
                Prod<Row, Col>: ArrayLen<<T as $op_trait<&'a T>>::Output> + Rem<Col>,
                Mod<Prod<Row, Col>, Col>: Same<U0>,
        {
            type Output = Matrix<<T as $op_trait<&'a T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (xr, yr) in self.0.into_iter().zip(rhs.0.as_slice()) {
                    for (x, y) in xr.into_iter().zip(yr.iter()) {
                        res.push($op_trait::$op_fn(x, y));
                    }
                }

                Matrix::from(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
            }
        }
    };

    (&T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, 'b, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for &'b Matrix<T, Row, Col>
            where
                &'b T: $op_trait<&'a T>,
                Row: Mul<Col> + ArrayLen<Vector<T, Col>> +
                     ArrayLen<Vector<<&'b T as $op_trait<&'a T>>::Output, Col>>,
                Col: ArrayLen<T> + ArrayLen<<&'b T as $op_trait<&'a T>>::Output>,
                Prod<Row, Col>: ArrayLen<<&'b T as $op_trait<&'a T>>::Output> + Rem<Col>,
                Mod<Prod<Row, Col>, Col>: Same<U0>,
        {
            type Output = Matrix<<&'b T as $op_trait<&'a T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (xr, yr) in self.0.as_slice().iter().zip(rhs.0.as_slice()) {
                    for (x, y) in xr.iter().zip(yr.iter()) {
                        res.push($op_trait::$op_fn(x, y));
                    }
                }

                Matrix::from(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
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

/*
impl<T, U, Row, Col> Mul<U> for Matrix<T, Row, Col>
    where
        T: Mul<U>,
        U: Clone,
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
{
    type Output = Matrix<<T as Mul<U>>::Output, Row, Col>;

    fn mul(self, rhs: U) -> Self::Output {
        let mut arr = ArrayVec::new();

        for x in self.0 {
            arr.push(x * rhs.clone());
        }

        debug_assert!(arr.is_full());
        Matrix::from_flat_array(arr.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}
*/

impl<T, Row, Col> Mul<T> for Matrix<T, Row, Col>
    where
        T: Mul + Clone,
        Row: Mul<Col> + ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as Mul>::Output, Col>>,
        Col: ArrayLen<T> + ArrayLen<<T as Mul>::Output>,
        Prod<Row, Col>: ArrayLen<<T as Mul>::Output> + Rem<Col>,
        Mod<Prod<Row, Col>, Col>: Same<U0>,
{
    type Output = Matrix<<T as Mul>::Output, Row, Col>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut arr = ArrayVec::new();

        for row in self.0 {
            for x in row {
                arr.push(x * rhs.clone());
            }
        }

        debug_assert!(arr.is_full());
        Matrix::from(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}

impl<T, N, LRow, RCol> Mul<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: num::Zero + Add<T, Output = T> + Mul<T, Output = T> + Clone,
        N: Mul<RCol> +
           ArrayLen<Vector<T, RCol>> +
           ArrayLen<T> +
           for<'a> ArrayLen<&'a T>,
        RCol: ArrayLen<T>,
        LRow: ArrayLen<Vector<T, RCol>> + ArrayLen<Vector<T, N>> + Mul<RCol>,
        Prod<LRow, RCol>: ArrayLen<T> + Rem<RCol>,
        Mod<Prod<LRow, RCol>, RCol>: Same<U0>,
{
    type Output = Matrix<Prod<T, T>, LRow, RCol>;

    fn mul(self, rhs: Matrix<T, N, RCol>) -> Self::Output {
        let mut res = ArrayVec::new();

        for lrow in self.rows_iter() {
            for rcol in rhs.cols_iter() {
                let s = lrow.iter().cloned().zip(rcol.into_iter())
                    .map(|(a, b)| a.clone() * b)
                    .fold(T::zero(), Add::add);
                res.push(s);
            }
        }

        debug_assert!(res.is_full());

        Matrix::from(Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}

#[test]
fn test_matrix_add_sub_mul() {
    let mut m1 = Matrix::<i32, U2, U2>::default();
    m1[(1,1)] = 4;
    m1[(0,0)] = 1;

    let m2 = Matrix::new([[0, 0], [0, 1]]);
    assert_eq!(m2[(1, 1)], 1);

    let m3 = m1 + m2;
    assert_eq!(m3, Matrix::new([[1, 0], [0, 5]]));
    assert_eq!(m3[(0,0)], 1);
    assert_eq!(m3[(1,1)], 5);

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

    let id = Matrix::new([[1, 0], [0, 1]]);
    assert_eq!(m1, m1 * id);

    let m5 = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(m5[(1,0)], 3);
    assert_eq!(m4 * m5 * 2, Matrix::new([[2, 4], [18, 24]]));

    let mb = Matrix::<num::BigInt, U2, U2>::new([[1.into(), 2.into()], [3.into(), 4.into()]]);
    assert_eq!(mb.clone() * mb, Matrix::new([[7.into(), 10.into()], [15.into(), 22.into()]]));
}


/*
impl<T, N> Matrix<T, N, N>
    where
        T: One + Div,
        N: Mul<Col>,
        <N as Mul<N>>::Output: ArrayLen<T>
{
    pub fn inverse(&self) -> Option<Matrix<T, N, N>> {
        let inv_det = T::one() / self.determinant();
    }
}
*/

// TODO: `into_rows` and `into_cols`

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    pub fn rows_iter(&self) -> RowsIter<T, Row, Col> {
        RowsIter(&self.0, 0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    pub fn cols_iter(&self) -> ColsIter<T, Row, Col> {
        ColsIter(&self.0, 0)
    }
}

pub struct RowsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize)
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a;

impl<'a, T: 'a, Row, Col> Iterator
    for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Item = Vector<T, Col>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.1 < Row::to_usize() {
            let s = self.1;
            let ret = self.0.as_ref()[s].clone();
            self.1 += 1;
            Some(ret)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = Row::to_usize() - self.1;
        (s, Some(s))
    }

    #[inline]
    fn count(self) -> usize {
        Row::to_usize() - self.1
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.1 += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    // avoiding an assert in the provided method
    #[inline]
    fn len(&self) -> usize {
        Row::to_usize() - self.1
    }
}

pub struct ColsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize)
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a;

impl<'a, T: 'a, Row, Col> Iterator
    for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    type Item = Vector<T, Row>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.1 < Col::to_usize() {
            let s = self.1;
            self.1 += 1;

            let mut arr = ArrayVec::new();
            for x in self.0.iter().map(|row| row.iter().nth(s).unwrap()) { // FIXME
                arr.push(x.clone());
            }
            debug_assert!(arr.is_full());

            Some(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = Col::to_usize() - self.1;
        (s, Some(s))
    }

    #[inline]
    fn count(self) -> usize {
        Col::to_usize() - self.1
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.1 += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    // avoiding an assert in the provided method
    #[inline]
    fn len(&self) -> usize {
        Col::to_usize() - self.1
    }
}

#[test]
fn test_matrix_rows_cols_iter() {
    let mut m: Matrix<i32, U3, U3> = Default::default();
    m[(0,0)] = 1;
    m[(1,1)] = 2;
    m[(2,2)] = 3;
    m[(0,2)] = 4;
    assert_eq!(m.dim(), (3, 3));

    let mut rows = m.rows_iter();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(rows.next().unwrap().iter().eq(&[0, 0, 3]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut cols = m.cols_iter();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(cols.next().unwrap().iter().eq(&[4, 0, 3]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);
}

