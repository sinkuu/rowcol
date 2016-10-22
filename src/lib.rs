pub extern crate typenum;
extern crate arrayvec;
extern crate num;

use typenum::consts::*;
use typenum::operator_aliases::{Prod, Mod};
use typenum::type_operators::Same;
use typenum::marker_traits::Unsigned;

use arrayvec::ArrayVec;

use std::ops::{Deref, DerefMut, Add, Sub, Mul, Rem, Index, IndexMut};
use std::marker::PhantomData;

/// A fixed-size vector.
///
/// ```rust
/// # use static_matrix::typenum::consts::*;
/// # use static_matrix::Vector;
///
/// let arr = Vector::<i32, U5>::new([1, 2, 3, 4, 5]);
/// assert_eq!(*arr, [1, 2, 3, 4, 5]);
/// ```
#[derive(Debug)]
pub struct Vector<T, N: ArrayLen<T>>(pub N::Array);

impl<T, N: ArrayLen<T>> Vector<T, N> {
    #[inline]
    pub fn new(array: N::Array) -> Self {
        Vector(array)
    }

    #[inline]
    pub fn into_inner(self) -> N::Array {
        self.0
    }

    #[inline]
    pub fn into_chunks<I>(self) -> VectorChunks<T, N, I>
        where
            N: Rem<I>,
            Mod<N, I>: Same<U0>
    {
        VectorChunks::new(ArrayVec::from(self.0))
    }
}

impl<T, N> Clone for Vector<T, N>
    where
        N: ArrayLen<T>,
        N::Array: Clone
{
    #[inline]
    fn clone(&self) -> Self {
        Vector(self.0.clone())
    }
}

impl<T, N> Copy for Vector<T, N>
    where
        N: ArrayLen<T>,
        N::Array: Copy
{
}

impl<T, N> Default for Vector<T, N>
    where
        N: ArrayLen<T>,
        N::Array: Default
{
    #[inline]
    fn default() -> Self {
        Vector(Default::default())
    }
}

impl<T, N> Deref for Vector<T, N>
    where
        N: ArrayLen<T>
{
    type Target = N::Array;

    #[inline]
    fn deref(&self) -> &N::Array {
        &self.0
    }
}

impl<T, N> DerefMut for Vector<T, N>
    where N: ArrayLen<T>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut N::Array {
        &mut self.0
    }
}

pub struct VectorChunks<T, N, I> where N: ArrayLen<T> {
    array: ArrayVec<N::Array>,
    _i: PhantomData<I>,
}

impl<T, N, I> VectorChunks<T, N, I> where N: ArrayLen<T> {
    fn new(a: ArrayVec<N::Array>) -> Self {
        VectorChunks {
            array: a,
            _i: PhantomData,
        }
    }
}

impl<T, N, I> Iterator for VectorChunks<T, N, I>
    where
        N: ArrayLen<T> + Rem<I>,
        I: Unsigned + ArrayLen<T>,
        Mod<N, I>: Same<U0>,
{
    type Item = Vector<T, I>;

    fn next(&mut self) -> Option<Self::Item> {
        // FIXME

        if self.array.is_empty() {
            None
        } else {
            let mut res = ArrayVec::new();

            for _ in 0..I::to_usize()  {
                res.push(self.array.pop().unwrap_or_else(|| unreachable!()));
            }
            debug_assert!(res.is_full());

            Some(Vector(res.into_inner().unwrap_or_else(|_| unreachable!())))
        }
    }
}

#[test]
fn test_vector_chunks() {
    let arr: Vector<i32, U6> = Vector::default();
    for a in arr.into_chunks::<U2>() {
        let a: [i32; 2] = a.into_inner();
        assert_eq!(a.len(), 2);
    }
}

pub trait ArrayLen<T> {
    type Array: AsRef<[T]> + AsMut<[T]> + arrayvec::Array<Item = T>;
}

macro_rules! impl_arraylen {
    ($tn:ident, $len:expr) => {
        impl<T> ArrayLen<T> for $tn {
            type Array = [T; $len];
        }
    }
}

impl_arraylen!(U1, 1);
impl_arraylen!(U2, 2);
impl_arraylen!(U3, 3);
impl_arraylen!(U4, 4);
impl_arraylen!(U5, 5);
impl_arraylen!(U6, 6);
impl_arraylen!(U7, 7);
impl_arraylen!(U8, 8);
impl_arraylen!(U9, 9);
impl_arraylen!(U10, 10);
impl_arraylen!(U11, 11);
impl_arraylen!(U12, 12);
impl_arraylen!(U13, 13);
impl_arraylen!(U14, 14);
impl_arraylen!(U15, 15);
impl_arraylen!(U16, 16);
impl_arraylen!(U17, 17);
impl_arraylen!(U18, 18);
impl_arraylen!(U19, 19);
impl_arraylen!(U20, 20);
impl_arraylen!(U21, 21);
impl_arraylen!(U22, 22);
impl_arraylen!(U23, 23);
impl_arraylen!(U24, 24);
impl_arraylen!(U25, 25);
impl_arraylen!(U26, 26);
impl_arraylen!(U27, 27);
impl_arraylen!(U28, 28);
impl_arraylen!(U29, 29);
impl_arraylen!(U30, 30);
impl_arraylen!(U31, 31);
impl_arraylen!(U32, 32);

#[test]
fn test_array() {
    use std::ops::Sub;
    // rustc bug (broken MIR) https://github.com/rust-lang/rust/issues/28828
    // use typenum::Diff;
    // let a: Vector<i32, Diff<U8, U3>> = Default::default();
    let a: Vector<i32, <U8 as Sub<U3>>::Output> = Default::default();
    assert_eq!(a.len(), 5);
    let _: [i32; 5] = a.0;
}

/// A fixed-size matrix.
///
/// ```rust
/// # use static_matrix::typenum::consts::*;
/// # use static_matrix::Matrix;
/// let mut m = Matrix::<i32, U3, U3>::from_array([[0, 0, 0], [0, 1, 0], [0, 2, 0]]);
///
/// assert_eq!(m[(1,1)], 1);
/// assert_eq!(m[(1,2)], 2);
/// assert_eq!(m.rows().nth(1), Some([0, 1, 0]));
/// assert_eq!(m.cols().nth(1), Some([0, 1, 2]));
/// assert_eq!(m + m, Matrix::from_array([[0, 0, 0], [0, 2, 0], [0, 4, 0]]));
/// ```
pub struct Matrix<T, Row, Col>(Vector<T, Prod<Row, Col>>)
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>;

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn new(arr: Vector<T, Prod<Row, Col>>) -> Self {
        Matrix(arr)
    }

    pub fn from_inner(arr: <<Row as Mul<Col>>::Output as ArrayLen<T>>::Array)
        -> Matrix<T, Row, Col>
    {
        Matrix::new(Vector::new(arr))
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn dim() -> (usize, usize) {
        (Row::to_usize(), Col::to_usize())
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned + ArrayLen<<Col as ArrayLen<T>>::Array>,
        Col: Unsigned + ArrayLen<T>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn from_array(rows: <Row as ArrayLen<<Col as ArrayLen<T>>::Array>>::Array)
        -> Matrix<T, Row, Col>
    {
        let mut arr = ArrayVec::new();
        for row in ArrayVec::from(rows) {
            arr.extend(ArrayVec::from(row));
        }
        debug_assert!(arr.is_full());
        Matrix::new(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
    }
}


impl<T, Row, Col> Default for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Default
{
    #[inline]
    fn default() -> Self {
        Matrix(Default::default())
    }
}

impl<T, Row, Col> Clone for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Clone
{
    fn clone(&self) -> Self {
        Matrix(self.0.clone())
    }
}

impl<T, Row, Col> Copy for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Copy
{
}

impl<T, Row, Col> PartialEq for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: PartialEq
{
    fn eq(&self, rhs: &Matrix<T, Row, Col>) -> bool {
        (self.0).0 == (rhs.0).0
    }
}

impl<T, Row, Col> Eq for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Eq
{
}

impl<T, Row, Col> ::std::fmt::Debug for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: ::std::fmt::Debug
{
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        (self.0).0.fmt(fmt)
    }
}

impl<T, Row, Col> Index<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i < Col::to_usize());
        assert!(j < Row::to_usize());

        &self.0.as_ref()[i + j * Col::to_usize()]
    }
}

impl<T, Row, Col> IndexMut<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i < Col::to_usize());
        assert!(j < Row::to_usize());

        &mut self.0.as_mut()[i + j * Col::to_usize()]
    }
}

impl<T, U, Row, Col> Add<Matrix<U, Row, Col>> for Matrix<T, Row, Col>
    where
        T: Add<U>,
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Row as Mul<Col>>::Output: ArrayLen<U>,
        <Row as Mul<Col>>::Output: ArrayLen<<T as Add<U>>::Output>
{
    type Output = Matrix<<T as Add<U>>::Output, Row, Col>;

    fn add(self, rhs: Matrix<U, Row, Col>) -> Self::Output {
        let xs: ArrayVec<_> = self.0.into_inner().into();
        let ys: ArrayVec<_> = rhs.0.into_inner().into();

        let mut res = ArrayVec::new();

        for (x, y) in xs.into_iter().zip(ys) {
            res.push(x + y);
        }

        Matrix::from_inner(res.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

impl<T, U, Row, Col> Sub<Matrix<U, Row, Col>> for Matrix<T, Row, Col>
    where
        T: Sub<U>,
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Row as Mul<Col>>::Output: ArrayLen<U>,
        <Row as Mul<Col>>::Output: ArrayLen<<T as Sub<U>>::Output>
{
    type Output = Matrix<<T as Sub<U>>::Output, Row, Col>;

    fn sub(self, rhs: Matrix<U, Row, Col>) -> Self::Output {
        let xs: ArrayVec<_> = self.0.into_inner().into();
        let ys: ArrayVec<_> = rhs.0.into_inner().into();

        let mut res = ArrayVec::new();

        for (x, y) in xs.into_iter().zip(ys) {
            res.push(x - y);
        }

        Matrix::from_inner(res.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

/*
impl<T, U, Row, Col> Mul<U> for Matrix<T, Row, Col>
    where
        T: Mul<U>,
        U: Clone,
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
{
    type Output = Matrix<<T as Mul<U>>::Output, Row, Col>;

    fn mul(self, rhs: U) -> Self::Output {
        let mut arr = ArrayVec::new();

        for x in self.0 {
            arr.push(x * rhs.clone());
        }

        debug_assert!(arr.is_full());
        Matrix::from_inner(arr.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}
*/

impl<T, Row, Col> Mul<T> for Matrix<T, Row, Col>
    where
        T: Mul + Clone,
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>,
        <Row as Mul<Col>>::Output: ArrayLen<<T as Mul>::Output>,
{
    type Output = Matrix<<T as Mul>::Output, Row, Col>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut arr = ArrayVec::new();

        for x in ArrayVec::from(self.0.into_inner()) {
            arr.push(x * rhs.clone());
        }

        debug_assert!(arr.is_full());
        Matrix::from_inner(arr.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

impl<T, N, LRow, RCol> Mul<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: Mul<T, Output = T> + Clone + ::std::iter::Sum,
        N: Mul<RCol> + Unsigned + ArrayLen<T>,
        <N as ArrayLen<T>>::Array: Clone,
        RCol: Unsigned,
        LRow: Unsigned + Mul<N> + Mul<RCol>,
        <LRow as Mul<N>>::Output: ArrayLen<T>,
        <N as Mul<RCol>>::Output: ArrayLen<T>,
        <LRow as Mul<RCol>>::Output: ArrayLen<<T as Mul>::Output>,
        <LRow as Mul<RCol>>::Output: ArrayLen<T>
{
    type Output = Matrix<<T as Mul>::Output, LRow, RCol>;

    fn mul(self, rhs: Matrix<T, N, RCol>) -> Self::Output {
        let mut res = ArrayVec::new();

        for lrow in self.rows() {
            for rcol in rhs.cols() {
                let s = ArrayVec::from(lrow.clone()).into_iter().zip(ArrayVec::from(rcol))
                    .map(|(a, b)| a * b)
                    .sum();
                res.push(s);
            }
        }

        debug_assert!(res.is_full());

        Matrix::from_inner(res.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

#[test]
fn test_matrix_add_sub_mul() {
    let mut m1 = Matrix::<i32, U2, U2>::default();
    m1[(1,1)] = 4;
    m1[(0,0)] = 1;
    let m2 = Matrix::from_array([[0, 0], [0, 1]]);
    assert_eq!(m2[(1, 1)], 1);
    let m3 = m1 + m2;
    assert_eq!(m3[(0,0)], 1);
    assert_eq!(m3[(1,1)], 5);
    let m4 = m1 - m2;
    assert_eq!(m4[(0,0)], 1);
    assert_eq!(m4[(1,1)], 3);
    let id = Matrix::from_array([[1, 0], [0, 1]]);
    assert_eq!(m1, m1 * id);
    let m5 = Matrix::<i32, U2, U2>::from_array([[1, 2], [3, 4]]);
    assert_eq!(m4 * m5 * 2, Matrix::from_array([[2, 4], [18, 24]]));
}

impl<'a, T: 'a, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned + ArrayLen<&'a T>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn rows_ref(&'a self) -> RowsIter<'a, T, Row, Col> {
        RowsIter(&*self.0, 0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: Mul<Col> + Unsigned,
        Col: Unsigned + ArrayLen<T>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn rows(&self) -> RowsClonedIter<T, Row, Col> {
        RowsClonedIter(&*self.0, 0)
    }
}

impl<'a, T: 'a, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned + ArrayLen<&'a T>,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn cols_ref(&'a self) -> ColsIter<'a, T, Row, Col> {
        ColsIter(&*self.0, 0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: Mul<Col> + Unsigned + ArrayLen<T>,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn cols(&self) -> ColsClonedIter<T, Row, Col> {
        ColsClonedIter(&*self.0, 0)
    }
}

macro_rules! decl_row_iter {
    ($name:ident, $item:ty) => {
        pub struct $name<'a, T: 'a, Row, Col>
            (&'a <Prod<Row, Col> as ArrayLen<T>>::Array, usize)
            where
                Row: Mul<Col>,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a;

        impl<'a, T: 'a, Row, Col> Iterator
            for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + Unsigned,
                Col: Unsigned + ArrayLen<$item>,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a,
        {
            type Item = <Col as ArrayLen<$item>>::Array;

            fn next(&mut self) -> Option<Self::Item> {
                if self.1 < Row::to_usize() {
                    let s = self.1;
                    self.1 += 1;

                    let mut arr = ArrayVec::new();
                    for x in &self.0.as_ref()[s * Col::to_usize() .. (s + 1) * Col::to_usize()] {
                        // no-op for `&T`
                        let x: $item = x.clone();
                        arr.push(x);
                    }
                    debug_assert!(arr.is_full());

                    Some(arr.into_inner().unwrap_or_else(|_| unreachable!()))
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
                assert!(n < Row::to_usize() - self.1);
                self.1 += n;
                self.next()
            }
        }

        impl<'a, T: 'a, Row, Col> ExactSizeIterator for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + Unsigned,
                Col: Unsigned + ArrayLen<$item>,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a
            {}
    }
}

macro_rules! decl_col_iter {
    ($name:ident, $item:ty) => {
        pub struct $name<'a, T: 'a, Row, Col>
            (&'a <Prod<Row, Col> as ArrayLen<T>>::Array, usize)
            where
                Row: Mul<Col>,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a;

        impl<'a, T: 'a, Row, Col> Iterator
            for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + Unsigned + ArrayLen<$item>,
                Col: Unsigned,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a
        {
            type Item = <Row as ArrayLen<$item>>::Array;

            fn next(&mut self) -> Option<Self::Item> {
                if self.1 < Col::to_usize() {
                    let s = self.1;
                    self.1 += 1;

                    let mut arr = ArrayVec::new();
                    for x in (0..Row::to_usize()).map(|i| &self.0.as_ref()[Col::to_usize() * i + s]) {
                        // no-op for `&T`
                        let x: $item = x.clone();
                        arr.push(x);
                    }
                    debug_assert!(arr.is_full());

                    Some(arr.into_inner().unwrap_or_else(|_| unreachable!()))
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
                assert!(self.1 + n < Col::to_usize());
                self.1 += n;
                self.next()
            }
        }

        impl<'a, T: 'a, Row, Col> ExactSizeIterator for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + Unsigned + ArrayLen<$item>,
                Col: Unsigned,
                <Row as Mul<Col>>::Output: ArrayLen<T> + 'a
            {}
    };
}

decl_row_iter!(RowsIter, &'a T);
decl_row_iter!(RowsClonedIter, T);
decl_col_iter!(ColsIter, &'a T);
decl_col_iter!(ColsClonedIter, T);


#[test]
fn test_matrix_rows_cols_iter() {
    let mut m: Matrix<i32, U3, U3> = Default::default();
    m[(0,0)] = 1;
    m[(1,1)] = 2;
    m[(2,2)] = 3;
    m[(0,2)] = 4;

    let mut rows = m.rows_ref();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[&1, &0, &0]));
    assert!(rows.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(rows.next().unwrap().iter().eq(&[&4, &0, &3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols_ref();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[&1, &0, &4]));
    assert!(cols.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(cols.next().unwrap().iter().eq(&[&0, &0, &3]));
    assert_eq!(cols.next(), None);

    let mut rows = m.rows();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 0]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(rows.next().unwrap().iter().eq(&[4, 0, 3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 0, 3]));
    assert_eq!(cols.next(), None);
}

