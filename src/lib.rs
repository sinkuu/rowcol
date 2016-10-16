pub extern crate typenum;

use typenum::consts::*;
use typenum::operator_aliases::Prod;
use typenum::Unsigned;
use std::ops::{Deref, DerefMut, Add, Sub, Mul, Index, IndexMut};

/// A fixed-size array.
///
/// ```rust
/// # use static_array::typenum::consts::*;
/// # use static_array::Array;
///
/// let arr = Array::<i32, U5>::new([1, 2, 3, 4, 5]);
/// assert_eq!(*arr, [1, 2, 3, 4, 5]);
/// ```
#[derive(Debug)]
pub struct Array<T, N: ArrayLen<T>>(pub N::Array);

impl<T, N: ArrayLen<T>> Array<T, N> {
    #[inline]
    pub fn new(array: N::Array) -> Self {
        Array(array)
    }
}

impl<T, N> Default for Array<T, N>
    where
        N: ArrayLen<T>,
        N::Array: Default
{
    #[inline]
    fn default() -> Self {
        Array::<T, N>(Default::default())
    }
}

impl<T, N> Deref for Array<T, N>
    where
        N: ArrayLen<T>
{
    type Target = N::Array;

    #[inline]
    fn deref(&self) -> &N::Array {
        &self.0
    }
}

impl<T, N> DerefMut for Array<T, N>
    where N: ArrayLen<T>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut N::Array {
        &mut self.0
    }
}

pub trait ArrayLen<T> {
    type Array: AsRef<[T]> + AsMut<[T]>;
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
    // let a: Array<i32, Diff<U8, U3>> = Default::default();
    let a: Array<i32, <U8 as Sub<U3>>::Output> = Default::default();
    assert_eq!(a.len(), 5);
    let _: [i32; 5] = a.0;
}

/// A fixed-size matrix.
///
/// ```rust
/// # use static_array::typenum::consts::*;
/// # use static_array::Matrix;
/// let mut m = Matrix::<i32, U3, U3>::default();
///
/// m[(1,1)] = 1;
/// m[(1,2)] = 2;
/// assert_eq!(m.rows_cloned().nth(1), Some([0, 1, 0]));
/// assert_eq!(m.cols_cloned().nth(1), Some([0, 1, 2]));
/// ```
pub struct Matrix<T, Row, Col>(Array<T, Prod<Row, Col>>)
    where
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>;

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

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn new(arr: Array<T, Prod<Row, Col>>) -> Self {
        Matrix(arr)
    }

    #[inline]
    pub fn dim() -> (usize, usize) {
        (Row::to_usize(), Col::to_usize())
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

impl<T, Row, Col> Add for Matrix<T, Row, Col>
    where
        T: Add<Output = T>,
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    type Output = Self;

    fn add(self, rhs: Matrix<T, Row, Col>) -> Self {
        unsafe {
            use std::{mem, ptr};

            let p1 = self.0.as_ref().as_ptr() as *const T;
            let p2 = rhs.0.as_ref().as_ptr() as *const T;

            mem::forget(self);
            mem::forget(rhs);

            let mut arr: <<Row as Mul<Col>>::Output as ArrayLen<T>>::Array = mem::uninitialized();
            for (i, e) in arr.as_mut().into_iter().enumerate() {
                mem::forget(mem::replace(e, ptr::read(p1.offset(i as isize)) + ptr::read(p2.offset(i as isize))));
            }

            Matrix(Array::new(arr))
        }
    }
}

impl<T, Row, Col> Sub for Matrix<T, Row, Col>
    where
        T: Sub<Output = T>,
        Row: Mul<Col>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    type Output = Self;

    fn sub(self, rhs: Matrix<T, Row, Col>) -> Self {
        unsafe {
            use std::{mem, ptr};

            let p1 = self.0.as_ref().as_ptr() as *const T;
            let p2 = rhs.0.as_ref().as_ptr() as *const T;

            mem::forget(self);
            mem::forget(rhs);

            let mut arr: <<Row as Mul<Col>>::Output as ArrayLen<T>>::Array = mem::uninitialized();
            for (i, e) in arr.as_mut().into_iter().enumerate() {
                mem::forget(mem::replace(e, ptr::read(p1.offset(i as isize)) - ptr::read(p2.offset(i as isize))));
            }

            Matrix(Array::new(arr))
        }
    }
}

#[test]
fn test_matrix_add() {
    let mut m1 = Matrix::<i32, U2, U2>::default();
    m1[(1,1)] = 4;
    m1[(0,0)] = 1;
    let mut m2 = Matrix::<i32, U2, U2>::default();
    m2[(1,1)] = 1;
    let m3 = m1 + m2;
    assert_eq!(m3[(0,0)], 1);
    assert_eq!(m3[(1,1)], 5);
}

impl<'a, T: 'a, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + Unsigned,
        Col: Unsigned + ArrayLen<&'a T>,
        <Row as Mul<Col>>::Output: ArrayLen<T>
{
    #[inline]
    pub fn rows(&'a self) -> RowsIter<'a, T, Row, Col> {
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
    pub fn rows_cloned(&self) -> RowsClonedIter<T, Row, Col> {
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
    pub fn cols(&'a self) -> ColsIter<'a, T, Row, Col> {
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
    pub fn cols_cloned(&self) -> ColsClonedIter<T, Row, Col> {
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

                    unsafe {
                        use std::mem;

                        let mut arr: Self::Item = mem::uninitialized();
                        for (a, b) in arr.as_mut().iter_mut().zip(&self.0.as_ref()[s * Col::to_usize() .. (s + 1) * Col::to_usize()]) {
                            let mut b: $item = b.clone();
                            mem::forget(mem::swap(a, &mut b));
                        }
                        Some(arr)
                    }
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

                    unsafe {
                        use std::mem;

                        let mut arr: Self::Item = mem::uninitialized();
                        for (a, b) in arr.as_mut().iter_mut().zip((0..Row::to_usize()).map(|i| &self.0.as_ref()[Col::to_usize() * i + s])) {
                            let mut b: $item = b.clone();
                            mem::forget(mem::swap(a, &mut b));
                        }
                        Some(arr)
                    }
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

    let mut rows = m.rows();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[&1, &0, &0]));
    assert!(rows.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(rows.next().unwrap().iter().eq(&[&4, &0, &3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[&1, &0, &4]));
    assert!(cols.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(cols.next().unwrap().iter().eq(&[&0, &0, &3]));
    assert_eq!(cols.next(), None);

    let mut rows = m.rows_cloned();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 0]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(rows.next().unwrap().iter().eq(&[4, 0, 3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols_cloned();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 0, 3]));
    assert_eq!(cols.next(), None);
}

