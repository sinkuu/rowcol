pub mod ops;
pub use self::ops::{Cofactor, Determinant, Inverse};

use typenum::{self, Prod, Same, Mod, UInt};
use typenum::consts::*;

use num;
use num::Complex;

use std::ops::{Add, Sub, Mul, Div, Neg, Rem,
    AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::fmt::{Debug, Display, Formatter};
use std::fmt::Result as FmtResult;

use vector::{Vector, ArrayLen};

/// A fixed-size matrix allocated on the stack.
///
/// ```rust
/// use rowcol::prelude::*;
///
/// let mut m = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
///
/// assert_eq!(m.determinant(), -2);
/// assert_eq!(m.transposed().determinant(), -2);
/// let m2 = m * 2;
/// assert_eq!(m2.determinant(), -8);
///
/// assert_eq!(m[(0, 0)], 1);
/// m[(0, 0)] = 2;
/// assert_eq!(m, Matrix::new([[2, 2], [3, 4]]));
///
/// assert_eq!(m[(U1::new(), U1::new())], 4);
/// // assert_eq!(m[(U2::new(), U2::new())], 1); // Error - statically checked!
/// ```
///
/// [`prelude`] provides typenum constants (`U1`, `U2`, ...), operation traits (e.g.,
/// [`Determinant`]), and alias for `Matrix`.
///
/// [`prelude`]: ../prelude/index.html
/// [`Determinant`]: ../prelude/trait.Determinant.html
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
    /// # use rowcol::prelude::*;
    /// let mat = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    /// ```
    pub fn new(rows: <Row as ArrayLen<<Col as ArrayLen<T>>::Array>>::Array)
        -> Matrix<T, Row, Col>
    {
        Matrix(Vector::<<Col as ArrayLen<T>>::Array, Row>::new(rows).into_iter()
               .map(Vector::new).collect())
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    /// Number of rows and columns in this matrix.
    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (Row::to_usize(), Col::to_usize())
    }

    /// Number of rows in this matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        Row::to_usize()
    }

    /// Number of columns in this matrix.
    #[inline]
    pub fn cols(&self) -> usize {
        Col::to_usize()
    }

    pub fn generate<F>(mut f: F) -> Self where F: FnMut((usize, usize)) -> T {
        Matrix((0..Row::to_usize()).map(|i| Vector::generate(|j| f((i, j)))).collect())
    }

    pub fn all<F>(&self, mut pred: F) -> bool where F: FnMut((usize, usize), &T) -> bool {
        self.0.iter()
            .enumerate()
            .all(|(i, col)| col.iter().enumerate().all(|(j, v)| pred((i, j), v)))
    }

    pub fn any<F>(&self, mut pred: F) -> bool where F: FnMut((usize, usize), &T) -> bool {
        self.0.iter().enumerate().any(|(i, col)| col.iter().enumerate().any(|(j, v)| pred((i, j), v)))
    }

    pub fn map<F, U>(self, mut f: F) -> Matrix<U, Row, Col>
        where
            F: FnMut(T) -> U,
            Row: ArrayLen<Vector<U, Col>>,
            Col: ArrayLen<U>,
    {
        Matrix(self.0.into_iter()
               .map(|row| row.into_iter().map(|a| f(a)).collect())
               .collect())
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &T {
        debug_assert!(i < Row::to_usize());
        debug_assert!(j < Col::to_usize());

        self.0.get_unchecked(i).get_unchecked(j)
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        debug_assert!(i < Row::to_usize());
        debug_assert!(j < Col::to_usize());

        self.0.get_unchecked_mut(i).get_unchecked_mut(j)
    }
}

impl<T, N> Matrix<T, N, N>
    where
        T: num::Zero + num::One,
        N: ArrayLen<Vector<T, N>> + ArrayLen<T>,
{
    /// Creates an identity matrix. This function is only implemented for square matrix types.
    ///
    /// ```rust
    /// # use rowcol::prelude::*;
    /// let id: Matrix<f32, U3, U3> = Matrix::identity();
    /// ```
    #[inline]
    pub fn identity() -> Self {
        Matrix::generate(|(i, j)| if i == j { T::one() } else { T::zero() })
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
        Matrix(v.into_chunks::<Col>().collect())
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T> + ArrayLen<Vector<T, Row>>,
{
    /// Returns the transpose of this matrix.
    pub fn transposed(&self) -> Matrix<T, Col, Row> {
        Matrix(self.cols_iter().collect())
    }
}

// TODO: fn transpose(&mut self) for Matrix<T, N, N>?

#[test]
fn test_transposed() {
    let mat = Matrix::<i32, U2, U2>::new([[1, 2], [3, 4]]);
    assert_eq!(mat.transposed(), Matrix::new([[1, 3], [2, 4]]));
}

impl<T, Row, Col> Default for Matrix<T, Row, Col>
    where
        T: Default,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
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

// This implementation allocates.
#[cfg(feature = "std")]
impl<T, Row, Col> Display for Matrix<T, Row, Col>
    where
        T: Display,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<String, Col>>,
        Col: ArrayLen<T> + ArrayLen<String> + ArrayLen<usize>,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        #[cfg(feature = "unicode_width")]
        fn str_len(s: &str) -> usize {
            use unicode_width::UnicodeWidthStr;
            s.width()
        }
        #[cfg(not(feature = "unicode_width"))]
        fn str_len(s: &str) -> usize {
            s.len()
        }

        let mut ws = Vector::<usize, Col>::default();
        let mut hs = Vector::<usize, Col>::default();

        let ss: Vector<Vector<String, Col>, Row> = (0..Row::to_usize()).map(|i| {
            (0..Col::to_usize()).map(|j| {
                let s = format!("{}", self[(i, j)]);

                let len = s.lines().map(|l| str_len(l)).max().unwrap_or(0);
                if ws[j] < len { ws[j] = len; }

                let lin = s.lines().count();
                if hs[i] < lin { hs[i] = lin; }

                s
            }).collect()
        }).collect();

        for (i, row) in ss.into_iter().enumerate() {
            use std::fmt::Write;

            for n in 0..hs[i] {
                try!(f.write_char(
                    if i == 0 && n == 0 {
                        '⎡'
                    } else if i == Row::to_usize()-1 && n == hs[i]-1 {
                        '⎣'
                    } else {
                        '⎢'
                    }));

                for (j, col) in row.iter().enumerate() {
                    let maxlen = col.lines().map(|l| str_len(l)).max().unwrap_or(0);
                    let leftpad = (ws[j] - maxlen) / 2;

                    for _ in 0..leftpad { try!(f.write_char(' ')); }

                    let line = col.lines().nth(n).unwrap_or("");
                    try!(f.write_str(line));

                    let rightpad = ws[j] - str_len(line) - leftpad +
                         if j != Col::to_usize()-1 { 1 } else { 0 };
                    for _ in 0..rightpad { try!(f.write_char(' ')); }
                }

                try!(f.write_str(
                    if i == 0 && n == 0{
                        "⎤\n"
                    } else if i == Row::to_usize()-1 && n == hs[i]-1 {
                        "⎦"
                    } else {
                        "⎥\n"
                    }));
            }
        }

        Ok(())
    }
}

#[cfg(feature = "std")]
#[test]
fn test_display() {
    let m = Matrix::<i32, U2, U2>::new([[111, 2], [3, 4]]);
    assert_eq!(format!("{}", m), "⎡111 2⎤\n\
                                  ⎣ 3  4⎦");
    let m2 = Matrix::new([[1, 2], [3, 4]]);
    assert_eq!(format!("{}", Matrix::<Matrix<i32, U2, U2>, U2, U2>::new([[m, m2], [m2.transposed(), m]])),
        "⎡⎡111 2⎤  ⎡1 2⎤ ⎤\n\
         ⎢⎣ 3  4⎦  ⎣3 4⎦ ⎥\n\
         ⎢ ⎡1 3⎤  ⎡111 2⎤⎥\n\
         ⎣ ⎣2 4⎦  ⎣ 3  4⎦⎦");
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

impl<T, Row, Col, UIRow, BIRow, UICol, BICol> Index<(UInt<UIRow, BIRow>, UInt<UICol, BICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UIRow: typenum::Unsigned,
        UICol: typenum::Unsigned,
        BIRow: typenum::Bit,
        BICol: typenum::Bit,
        UInt<UIRow, BIRow>: typenum::Unsigned + typenum::Cmp<Row>,
        UInt<UICol, BICol>: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<UInt<UIRow, BIRow>, Row>: Same<typenum::Less>,
        typenum::Compare<UInt<UICol, BICol>, Col>: Same<typenum::Less>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: (UInt<UIRow, BIRow>, UInt<UICol, BICol>)) -> &T {
        unsafe {
            self.0.get_unchecked(<UInt<UIRow, BIRow> as typenum::Unsigned>::to_usize())
                .get_unchecked(<UInt<UICol, BICol> as typenum::Unsigned>::to_usize())
        }
    }
}

impl<T, Row, Col, UIRow, BIRow, UICol, BICol> IndexMut<(UInt<UIRow, BIRow>, UInt<UICol, BICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UIRow: typenum::Unsigned,
        UICol: typenum::Unsigned,
        BIRow: typenum::Bit,
        BICol: typenum::Bit,
        UInt<UIRow, BIRow>: typenum::Unsigned + typenum::Cmp<Row>,
        UInt<UICol, BICol>: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<UInt<UIRow, BIRow>, Row>: Same<typenum::Less>,
        typenum::Compare<UInt<UICol, BICol>, Col>: Same<typenum::Less>,
{
    #[inline]
    fn index_mut(&mut self, _: (UInt<UIRow, BIRow>, UInt<UICol, BICol>)) -> &mut T {
        unsafe {
            self.0.get_unchecked_mut(<UInt<UIRow, BIRow> as typenum::Unsigned>::to_usize())
                .get_unchecked_mut(<UInt<UICol, BICol> as typenum::Unsigned>::to_usize())
        }
    }
}

// U0 = UTerm
impl<T, Row, Col> Index<(U0, U0)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        U0: typenum::Cmp<Row> + typenum::Cmp<Col>,
        typenum::Compare<U0, Row>: Same<typenum::Less>,
        typenum::Compare<U0, Col>: Same<typenum::Less>,
{
    type Output = T;

    fn index(&self, _: (U0, U0)) -> &T {
        unsafe {
            self.0.get_unchecked(0).get_unchecked(0)
        }
    }
}

impl<T, Row, Col> IndexMut<(U0, U0)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        U0: typenum::Cmp<Row> + typenum::Cmp<Col>,
        typenum::Compare<U0, Row>: Same<typenum::Less>,
        typenum::Compare<U0, Col>: Same<typenum::Less>,
{
    fn index_mut(&mut self, _: (U0, U0)) -> &mut T {
        unsafe {
            self.0.get_unchecked_mut(0).get_unchecked_mut(0)
        }
    }
}

impl<T, Row, Col, UIRow, BIRow> Index<(UInt<UIRow, BIRow>, U0)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UIRow: typenum::Unsigned,
        BIRow: typenum::Bit,
        UInt<UIRow, BIRow>: typenum::Unsigned + typenum::Cmp<Row>,
        U0: typenum::Cmp<Col>,
        typenum::Compare<UInt<UIRow, BIRow>, Row>: Same<typenum::Less>,
        typenum::Compare<U0, Col>: Same<typenum::Less>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: (UInt<UIRow, BIRow>, U0)) -> &T {
        unsafe {
            self.0.get_unchecked(<UInt<UIRow, BIRow> as typenum::Unsigned>::to_usize())
                .get_unchecked(0)
        }
    }
}

impl<T, Row, Col, UIRow, BIRow> IndexMut<(UInt<UIRow, BIRow>, U0)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UIRow: typenum::Unsigned,
        BIRow: typenum::Bit,
        UInt<UIRow, BIRow>: typenum::Unsigned + typenum::Cmp<Row>,
        U0: typenum::Cmp<Col>,
        typenum::Compare<UInt<UIRow, BIRow>, Row>: Same<typenum::Less>,
        typenum::Compare<U0, Col>: Same<typenum::Less>,
{
    #[inline]
    fn index_mut(&mut self, _: (UInt<UIRow, BIRow>, U0)) -> &mut T {
        unsafe {
            self.0.get_unchecked_mut(<UInt<UIRow, BIRow> as typenum::Unsigned>::to_usize())
                .get_unchecked_mut(0)
        }
    }
}

impl<T, Row, Col, UICol, BICol> Index<(U0, UInt<UICol, BICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UICol: typenum::Unsigned,
        BICol: typenum::Bit,
        U0: typenum::Cmp<Row>,
        UInt<UICol, BICol>: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<U0, Row>: Same<typenum::Less>,
        typenum::Compare<UInt<UICol, BICol>, Col>: Same<typenum::Less>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: (U0, UInt<UICol, BICol>)) -> &T {
        unsafe {
            self.0.get_unchecked(0)
                .get_unchecked(<UInt<UICol, BICol> as typenum::Unsigned>::to_usize())
        }
    }
}

impl<T, Row, Col, UICol, BICol> IndexMut<(U0, UInt<UICol, BICol>)> for Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
        UICol: typenum::Unsigned,
        BICol: typenum::Bit,
        U0: typenum::Cmp<Row>,
        UInt<UICol, BICol>: typenum::Unsigned + typenum::Cmp<Col>,
        typenum::Compare<U0, Row>: Same<typenum::Less>,
        typenum::Compare<UInt<UICol, BICol>, Col>: Same<typenum::Less>,
{
    #[inline]
    fn index_mut(&mut self, _: (U0, UInt<UICol, BICol>)) -> &mut T {
        unsafe {
            self.0.get_unchecked_mut(0)
                .get_unchecked_mut(<UInt<UICol, BICol> as typenum::Unsigned>::to_usize())
        }
    }
}

impl<T, Row, Col> num::Zero for Matrix<T, Row, Col>
    where
        T: num::Zero,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    fn zero() -> Self {
        Matrix::generate(|_| T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|col| col.iter().all(|v| v.is_zero()))
    }
}

#[test]
fn test_zero() {
    use num::Zero;
    let mut m = Matrix::<f32, U3, U3>::zero();
    assert!(m.is_zero());
    m[(0,0)] = 1.0;
    assert!(!m.is_zero());
}

impl<T, Row, Col> num::Bounded for Matrix<T, Row, Col>
    where
        T: num::Bounded,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    #[inline]
    fn min_value() -> Self {
        Matrix::generate(|_| T::min_value())
    }

    #[inline]
    fn max_value() -> Self {
        Matrix::generate(|_| T::max_value())
    }
}

#[test]
fn test_bounded() {
    use num::Bounded;
    assert_eq!(Matrix::<u32, U2, U2>::min_value(), Matrix::new([[0, 0], [0, 0]]));
}

/// Creates an identity matrix, equivalent to `Matrix::identity`. Implemented for square matrix types.
impl<T, N> num::One for Matrix<T, N, N>
    where
        T: num::Zero + num::One + Clone + Mul<T, Output = T>,
        N: ArrayLen<T> + ArrayLen<Vector<T, N>>,
{
    #[inline]
    fn one() -> Self {
        Matrix::identity()
    }
}

impl<T, Row, Col> Matrix<Complex<T>, Row, Col>
    where
        T: Neg<Output = T> + num::Num + Clone,
        Row: ArrayLen<Vector<Complex<T>, Col>>,
        Col: ArrayLen<Complex<T>>,
{
    #[inline]
    pub fn conjugate(&self) -> Matrix<Complex<T>, Row, Col> {
        self.clone().map(|a| a.conj())
    }
}

#[test]
fn test_conjugate() {
    let m =
        Matrix::<Complex<i32>, U2, U2>::new([[Complex::new(1,1), Complex::new(1,-1)],
                                            [Complex::new(-2,1), Complex::new(1,1)]]);
    assert!((m.conjugate() + m).all(|_, a| a.im == 0));
}

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
        T: Mul + Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as Mul>::Output, Col>>,
        Col: ArrayLen<T> + ArrayLen<<T as Mul>::Output>,
{
    type Output = Matrix<<T as Mul>::Output, Row, Col>;

    fn mul(self, rhs: T) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| row * rhs.clone()).collect())
    }
}

impl<'a, T, Row, Col> Mul<&'a T> for Matrix<T, Row, Col>
    where
        T: Mul<&'a T>,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as Mul<&'a T>>::Output, Col>>,
        Col: ArrayLen<T> + ArrayLen<<T as Mul<&'a T>>::Output>,
{
    type Output = Matrix<<T as Mul<&'a T>>::Output, Row, Col>;

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

impl<'a, T, Row, Col> MulAssign<&'a T> for Matrix<T, Row, Col>
    where
        T: MulAssign<&'a T> + Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        for row in self.0.iter_mut() {
            for a in row.iter_mut() {
                *a *= rhs;
            }
        }
    }
}

// matrix * matrix

impl<T, N, LRow, RCol> Mul<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: num::Zero + Add<T, Output = T> + Mul<T, Output = T> + Clone,
        N: ArrayLen<Vector<T, RCol>> +
           ArrayLen<T>,
        RCol: ArrayLen<T>,
        LRow: ArrayLen<Vector<T, RCol>> + ArrayLen<Vector<T, N>>,
{
    type Output = Matrix<Prod<T, T>, LRow, RCol>;

    fn mul(self, rhs: Matrix<T, N, RCol>) -> Self::Output {
        Matrix(self.0.into_iter().map(|lrow| {
            rhs.cols_iter().map(|rcol| {
                lrow.iter().cloned().zip(rcol).map(|(a, b)| a * b).fold(T::zero(), Add::add)
            }).collect()
        }).collect())
    }
}

impl<T, N, LRow, RCol> MulAssign<Matrix<T, N, RCol>> for Matrix<T, LRow, N>
    where
        T: num::Zero + Add<T, Output = T> + Mul<T, Output = T> + Clone,
        N: ArrayLen<Vector<T, RCol>> +
           ArrayLen<T>,
        RCol: ArrayLen<T>,
        LRow: ArrayLen<Vector<T, RCol>> + ArrayLen<Vector<T, N>>,
{
    fn mul_assign(&mut self, rhs: Matrix<T, N, RCol>) {
        *self = Matrix(self.0.clone().into_iter().map(|lrow| {
            rhs.cols_iter().map(|rcol| {
                lrow.iter().cloned().zip(rcol).map(|(a, b)| a * b).fold(T::zero(), Add::add)
            }).collect()
        }).collect())
    }
}

// matrix * vector

impl<T, Row, Col> Mul<Vector<T, Col>> for Matrix<T, Row, Col>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + num::Zero + Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    type Output = Vector<T, Row>;

    fn mul(self, rhs: Vector<T, Col>) -> Self::Output {
        self.0.into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(rhs.iter().cloned())
                    .map(|(a, b)| a * b.clone()).fold(T::zero(), Add::add)
            })
            .collect()
    }
}

// matrix / x

impl<T, U, Row, Col> Div<U> for Matrix<T, Row, Col>
    where
        T: Div<U>,
        U: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<<T as Div<U>>::Output, Col>>,
        Col: ArrayLen<T> + ArrayLen<<T as Div<U>>::Output>,
{
    type Output = Matrix<<T as Div<U>>::Output, Row, Col>;

    fn div(self, rhs: U) -> Self::Output {
        Matrix(self.0.into_iter().map(|row| {
            row / rhs.clone()
        }).collect())
    }
}

impl<T, U, Row, Col> DivAssign<U> for Matrix<T, Row, Col>
    where
        T: DivAssign<U>,
        U: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn div_assign(&mut self, rhs: U) {
        for row in self.0.iter_mut() {
            for a in row.iter_mut() {
                *a /= rhs.clone();
            }
        }
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
    assert_eq!(m4 * m5 * 2 / 2, Matrix::new([[1], [9]]));

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
    /// Returns an iterator over rows of this matrix.
    #[inline]
    pub fn rows_iter(&self) -> RowsIter<T, Row, Col> {
        RowsIter(&self.0, 0, Row::to_usize())
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    /// Returns an iterator over columns of this matrix.
    #[inline]
    pub fn cols_iter(&self) -> ColsIter<T, Row, Col> {
        ColsIter(&self.0, 0, Col::to_usize())
    }
}

pub struct RowsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize, usize)
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
        if self.1 < self.2 {
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

impl<'a, T: 'a, Row, Col> DoubleEndedIterator for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn next_back(&mut self) -> Option<Vector<T, Col>> {
        if self.1 < self.2 {
            let s = self.2;
            let ret = self.0.as_ref()[s-1].clone();
            self.2 -= 1;
            Some(ret)
        } else {
            None
        }
    }
}

pub struct ColsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize, usize)
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
        debug_assert!(self.2 <= Col::to_usize());
        if self.1 < self.2 {
            let s = self.1;
            self.1 += 1;

            let v = self.0.iter().map(|row| unsafe { row.as_slice().get_unchecked(s).clone() }).collect();
            Some(v)
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

impl<'a, T: 'a, Row, Col> DoubleEndedIterator
    for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    fn next_back(&mut self) -> Option<Vector<T, Row>> {
        if self.1 < self.2 {
            let s = self.2;
            self.2 -= 1;

            Some(self.0.iter().map(|row| row.as_ref()[s-1].clone()).collect())
        } else {
            None
        }
    }
}

#[test]
fn test_matrix_rows_cols_iter() {
    let mut m: Matrix<i32, U2, U3> = Default::default();
    m[(0,0)] = 1;
    m[(1,1)] = 2;
    m[(1,2)] = 3;
    m[(0,2)] = 4;
    assert_eq!(m.dim(), (2, 3));
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);

    let mut rows = m.rows_iter();

    assert_eq!(rows.len(), 2);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 3]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut rows = m.rows_iter().rev();

    assert_eq!(rows.len(), 2);
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 3]));
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut cols = m.cols_iter();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2]));
    assert!(cols.next().unwrap().iter().eq(&[4, 3]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);

    let mut cols = m.cols_iter().rev();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[4, 3]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2]));
    assert!(cols.next().unwrap().iter().eq(&[1, 0]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);
}

