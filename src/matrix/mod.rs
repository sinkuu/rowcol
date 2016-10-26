use typenum::{self, Prod, Same};
#[cfg(test)] use typenum::consts::*;

use arrayvec::ArrayVec;

use std::ops::{Add, Sub, Mul, Index, IndexMut};
use std::marker::PhantomData;

use vector::{Vector, ArrayLen};

/// A fixed-size matrix whose elements are allocated on the stack.
///
/// ```rust
/// # use static_matrix::typenum::consts::*;
/// # use static_matrix::{Matrix, Vector};
/// let mut m = Matrix::<i32, U3, U3>::new([[0, 0, 0], [0, 1, 0], [0, 2, 0]]);
///
/// assert_eq!(m[(1,1)], 1);
/// assert_eq!(m[(2,1)], 2);
/// assert_eq!(m.rows().nth(1), Some(Vector::new([0, 1, 0])));
/// assert_eq!(m.cols().nth(1), Some(Vector::new([0, 1, 2])));
/// assert_eq!(m + m, Matrix::new([[0, 0, 0], [0, 2, 0], [0, 4, 0]]));
/// ```
pub struct Matrix<T, Row, Col>(Vector<T, Prod<Row, Col>>)
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>;

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned + ArrayLen<<Col as ArrayLen<T>>::Array>,
        Col: typenum::Unsigned + ArrayLen<T>,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn new(rows: <Row as ArrayLen<<Col as ArrayLen<T>>::Array>>::Array)
        -> Matrix<T, Row, Col>
    {
        let mut arr = ArrayVec::new();
        for row in ArrayVec::from(rows) {
            arr.extend(ArrayVec::from(row));
        }
        debug_assert!(arr.is_full());
        Matrix::from_flat_array(arr.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

impl<T, Row, Col> From<Vector<T, Prod<Row, Col>>> for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
{
    #[inline]
    fn from(arr: Vector<T, Prod<Row, Col>>) -> Self {
        Matrix(arr)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>
{
    /// Creates a matrix from its representation in a flat array.
    ///
    /// ```rust
    /// # use static_matrix::Matrix;
    /// # use static_matrix::typenum::consts::*;
    ///
    /// let mat = Matrix::<i32, U2, U2>::from_flat_array([1, 2, 3, 4]);
    /// assert_eq!(mat, Matrix::new([[1, 2], [3, 4]]));
    /// ```
    #[inline]
    pub fn from_flat_array(arr: <Prod<Row, Col> as ArrayLen<T>>::Array)
        -> Matrix<T, Row, Col>
    {
        Matrix::from(Vector::new(arr))
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (Row::to_usize(), Col::to_usize())
    }
}

impl<T, Row, Col> Default for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
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
        Prod<Row, Col>: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Clone
{
    fn clone(&self) -> Self {
        Matrix(self.0.clone())
    }
}

impl<T, Row, Col> Copy for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: Copy
{
}

impl<T, Row, Col> PartialEq for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
        T: PartialEq,
{
    fn eq(&self, rhs: &Matrix<T, Row, Col>) -> bool {
        self.0 == rhs.0
    }
}

impl<T, Row, Col> Eq for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
        T: PartialEq,
{
}

impl<T, Row, Col> ::std::fmt::Debug for Matrix<T, Row, Col>
    where
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
        <Prod<Row, Col> as ArrayLen<T>>::Array: ::std::fmt::Debug
{
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        use std::fmt::Pointer;
        self.0.as_slice().fmt(fmt)
    }
}

impl<T, Row, Col> Index<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned,
        Prod<Row, Col>: ArrayLen<T>
{
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i < Col::to_usize());
        assert!(j < Row::to_usize());

        &self.0.as_ref()[i * Col::to_usize() + j]
    }
}

impl<T, Row, Col, IRow, ICol> Index<(PhantomData<IRow>, PhantomData<ICol>)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned,
        IRow: typenum::Unsigned + typenum::Cmp<Row>,
        ICol: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<IRow, Row>: Same<typenum::Less>,
        typenum::Compare<ICol, Col>: Same<typenum::Less>,
        Prod<Row, Col>: ArrayLen<T>
{
    type Output = T;

    #[inline]
    fn index(&self, _: (PhantomData<IRow>, PhantomData<ICol>)) -> &T {
        &self.0.as_ref()[IRow::to_usize() * Col::to_usize() + ICol::to_usize()]
    }
}

impl<T, Row, Col> IndexMut<(usize, usize)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i < Col::to_usize());
        assert!(j < Row::to_usize());

        &mut self.0.as_mut()[i * Col::to_usize() + j]
    }
}

impl<T, Row, Col, IRow, ICol> IndexMut<(PhantomData<IRow>, PhantomData<ICol>)> for Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned,
        IRow: typenum::Unsigned + typenum::Cmp<Row>,
        ICol: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<IRow, Row>: Same<::typenum::Less>,
        typenum::Compare<ICol, Col>: Same<::typenum::Less>,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    fn index_mut(&mut self, _: (PhantomData<IRow>, PhantomData<ICol>)) -> &mut T {
        &mut self.0.as_mut()[IRow::to_usize() * Col::to_usize() + ICol::to_usize()]
    }
}

macro_rules! impl_matrix_arith {
    (T T : $op_trait:ident, $op_fn: ident) => {
        impl<T, Row, Col> $op_trait<Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait,
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T>,
                Prod<Row, Col>: ArrayLen<<T as $op_trait>::Output>
        {
            type Output = Matrix<<T as $op_trait>::Output, Row, Col>;

            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                let xs: ArrayVec<_> = self.0.into_inner().into();
                let ys: ArrayVec<_> = rhs.0.into_inner().into();

                let mut res = ArrayVec::new();

                for (x, y) in xs.into_iter().zip(ys) {
                    res.push($op_trait::$op_fn(x, y));
                }

                Matrix::from_flat_array(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (&T T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<Matrix<T, Row, Col>> for &'a Matrix<T, Row, Col>
            where
                &'a T: $op_trait<T>,
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T>,
                Prod<Row, Col>: ArrayLen<<&'a T as $op_trait<T>>::Output>
        {
            type Output = Matrix<<&'a T as $op_trait<T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: Matrix<T, Row, Col>) -> Self::Output {
                let xs = self.0.as_slice();
                let ys: ArrayVec<_> = rhs.0.into_inner().into();

                let mut res = ArrayVec::new();

                for (x, y) in xs.into_iter().zip(ys) {
                    res.push($op_trait::$op_fn(x, y));
                }

                Matrix::from_flat_array(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for Matrix<T, Row, Col>
            where
                T: $op_trait<&'a T>,
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T>,
                Prod<Row, Col>: ArrayLen<<T as $op_trait<&'a T>>::Output>
        {
            type Output = Matrix<<T as $op_trait<&'a T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                let xs: ArrayVec<_> = self.0.into_inner().into();
                let ys = rhs.0.as_slice();

                let mut res = ArrayVec::new();

                for (x, y) in xs.into_iter().zip(ys) {
                    res.push($op_trait::$op_fn(x, y));
                }

                Matrix::from_flat_array(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (&T &T : $op_trait:ident, $op_fn: ident) => {
        impl<'a, 'b, T, Row, Col> $op_trait<&'a Matrix<T, Row, Col>> for &'b Matrix<T, Row, Col>
            where
                &'b T: $op_trait<&'a T>,
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T>,
                Prod<Row, Col>: ArrayLen<<&'b T as $op_trait<&'a T>>::Output>
        {
            type Output = Matrix<<&'b T as $op_trait<&'a T>>::Output, Row, Col>;

            fn $op_fn(self, rhs: &'a Matrix<T, Row, Col>) -> Self::Output {
                let xs = self.0.as_slice();
                let ys = rhs.0.as_slice();

                let mut res = ArrayVec::new();

                for (x, y) in xs.into_iter().zip(ys) {
                    res.push($op_trait::$op_fn(x, y));
                }

                Matrix::from_flat_array(res.into_inner().unwrap_or_else(|_| unreachable!()))
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
        Row: Mul<Col>,
        Prod<Row, Col>: ArrayLen<T>,
        Prod<Row, Col>: ArrayLen<<T as Mul>::Output>,
{
    type Output = Matrix<<T as Mul>::Output, Row, Col>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut arr = ArrayVec::new();

        for x in ArrayVec::from(self.0.into_inner()) {
            arr.push(x * rhs.clone());
        }

        debug_assert!(arr.is_full());
        Matrix::from_flat_array(arr.into_inner().unwrap_or_else(|_| unreachable!()))
    }
}

impl<T, N, LRow, RCol> Mul<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: Mul<T, Output = T> + Clone + ::std::iter::Sum,
        N: Mul<RCol> + typenum::Unsigned + ArrayLen<T> + for<'a> ArrayLen<&'a T>,
        <N as ArrayLen<T>>::Array: Clone,
        RCol: typenum::Unsigned,
        LRow: typenum::Unsigned + Mul<N> + Mul<RCol>,
        Prod<LRow, N>: ArrayLen<T>,
        Prod<N, RCol>: ArrayLen<T>,
        Prod<LRow, RCol>: ArrayLen<Prod<T, T>>,
        Prod<LRow, RCol>: ArrayLen<T>
{
    type Output = Matrix<Prod<T, T>, LRow, RCol>;

    fn mul(self, rhs: Matrix<T, N, RCol>) -> Self::Output {
        let mut res = ArrayVec::new();

        for lrow in self.rows_ref() {
            for rcol in rhs.cols() {
                let s = lrow.iter().cloned().zip(rcol.into_iter())
                    .map(|(a, b)| a.clone() * b)
                    .sum();
                res.push(s);
            }
        }

        debug_assert!(res.is_full());

        Matrix::from_flat_array(res.into_inner().unwrap_or_else(|_| unreachable!()))
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

impl<'a, T: 'a, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned + ArrayLen<&'a T>,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn rows_ref(&'a self) -> RowsRefIter<'a, T, Row, Col> {
        RowsRefIter(&*self.0, 0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: Mul<Col> + typenum::Unsigned,
        Col: typenum::Unsigned + ArrayLen<T>,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn rows(&self) -> RowsIter<T, Row, Col> {
        RowsIter(&*self.0, 0)
    }
}

impl<'a, T: 'a, Row, Col> Matrix<T, Row, Col>
    where
        Row: Mul<Col> + typenum::Unsigned + ArrayLen<&'a T>,
        Col: typenum::Unsigned,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn cols_ref(&'a self) -> ColsRefIter<'a, T, Row, Col> {
        ColsRefIter(&*self.0, 0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: Mul<Col> + typenum::Unsigned + ArrayLen<T>,
        Col: typenum::Unsigned,
        Prod<Row, Col>: ArrayLen<T>
{
    #[inline]
    pub fn cols(&self) -> ColsIter<T, Row, Col> {
        ColsIter(&*self.0, 0)
    }
}

macro_rules! decl_row_iter {
    ($name:ident, $item:ty) => {
        pub struct $name<'a, T: 'a, Row, Col>
            (&'a <Prod<Row, Col> as ArrayLen<T>>::Array, usize)
            where
                Row: Mul<Col>,
                Prod<Row, Col>: ArrayLen<T> + 'a;

        impl<'a, T: 'a, Row, Col> Iterator
            for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned + ArrayLen<$item>,
                Prod<Row, Col>: ArrayLen<T> + 'a,
        {
            type Item = Vector<$item, Col>;

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

                    Some(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
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
                Row: Mul<Col> + typenum::Unsigned,
                Col: typenum::Unsigned + ArrayLen<$item>,
                Prod<Row, Col>: ArrayLen<T> + 'a
            {}
    }
}

macro_rules! decl_col_iter {
    ($name:ident, $item:ty) => {
        pub struct $name<'a, T: 'a, Row, Col>
            (&'a <Prod<Row, Col> as ArrayLen<T>>::Array, usize)
            where
                Row: Mul<Col>,
                Prod<Row, Col>: ArrayLen<T> + 'a;

        impl<'a, T: 'a, Row, Col> Iterator
            for $name<'a, T, Row, Col>
            where
                $item: Clone,
                Row: Mul<Col> + typenum::Unsigned + ArrayLen<$item>,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T> + 'a
        {
            type Item = Vector<$item, Row>;

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

                    Some(Vector::new(arr.into_inner().unwrap_or_else(|_| unreachable!())))
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
                Row: Mul<Col> + typenum::Unsigned + ArrayLen<$item>,
                Col: typenum::Unsigned,
                Prod<Row, Col>: ArrayLen<T> + 'a
            {}
    };
}

decl_row_iter!(RowsRefIter, &'a T);
decl_row_iter!(RowsIter, T);
decl_col_iter!(ColsRefIter, &'a T);
decl_col_iter!(ColsIter, T);


#[test]
fn test_matrix_rows_cols_iter() {
    let mut m: Matrix<i32, U3, U3> = Default::default();
    m[(0,0)] = 1;
    m[(1,1)] = 2;
    m[(2,2)] = 3;
    m[(0,2)] = 4;
    assert_eq!(m.dim(), (3, 3));

    let mut rows = m.rows_ref();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[&1, &0, &4]));
    assert!(rows.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(rows.next().unwrap().iter().eq(&[&0, &0, &3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols_ref();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[&1, &0, &0]));
    assert!(cols.next().unwrap().iter().eq(&[&0, &2, &0]));
    assert!(cols.next().unwrap().iter().eq(&[&4, &0, &3]));
    assert_eq!(cols.next(), None);

    let mut rows = m.rows();

    assert_eq!(rows.len(), 3);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(rows.next().unwrap().iter().eq(&[0, 0, 3]));
    assert_eq!(rows.next(), None);

    let mut cols = m.cols();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2, 0]));
    assert!(cols.next().unwrap().iter().eq(&[4, 0, 3]));
    assert_eq!(cols.next(), None);
}

