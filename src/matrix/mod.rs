pub mod ops;
pub use self::ops::{Cofactor, Determinant, Inverse};

pub mod iter;
use self::iter::*;

mod arith;

use typenum::{self, Prod, Same, Mod, UInt};
use typenum::consts::*;

use num;
use num::Complex;

use std::ops::{Mul, Rem, Neg, Index, IndexMut};
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
/// assert_eq!(m.transposed().determinant(), -2);
/// assert_eq!((m * 2).determinant(), -8);
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
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<Vector<(String, usize), Col>> +
            ArrayLen<usize>,
        Col: ArrayLen<T> + ArrayLen<(String, usize)> + ArrayLen<usize>,
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
        let mut hs = Vector::<usize, Row>::default();

        let ss: Vector<Vector<(String, usize), Col>, Row> =
            (0..Row::to_usize()).map(|i| {
                (0..Col::to_usize()).map(|j| {
                    let s = format!("{}", self[(i, j)]);

                    let len = s.lines().map(|l| str_len(l)).max().unwrap_or(0);
                    if ws[j] < len { ws[j] = len; }

                    let lin = s.lines().count();
                    if hs[i] < lin { hs[i] = lin; }

                    (s, len)
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

                for (j, &(ref col, ref maxlen)) in row.iter().enumerate() {
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
    assert_eq!(format!("{}",
                       Matrix::<Matrix<i32, U2, U2>, U3, U2>
                           ::new([[m, m2], [m2.transposed(), m], [m2, m.transposed()]])),
        "⎡⎡111 2⎤  ⎡1 2⎤ ⎤\n\
         ⎢⎣ 3  4⎦  ⎣3 4⎦ ⎥\n\
         ⎢ ⎡1 3⎤  ⎡111 2⎤⎥\n\
         ⎢ ⎣2 4⎦  ⎣ 3  4⎦⎥\n\
         ⎢ ⎡1 2⎤  ⎡111 3⎤⎥\n\
         ⎣ ⎣3 4⎦  ⎣ 2  4⎦⎦");
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

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    /// Returns an iterator over rows of this matrix.
    #[inline]
    pub fn rows_iter(&self) -> RowsIter<T, Row, Col> {
        RowsIter::new(&self.0)
    }

    /// Returns an iterator over columns of this matrix.
    #[inline]
    pub fn cols_iter(&self) -> ColsIter<T, Row, Col> {
        ColsIter::new(&self.0)
    }
}

impl<T, Row, Col> Matrix<T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    /// Returns an iterator over rows of this matrix by reference.
    #[inline]
    pub fn rows_iter_ref(&self) -> RowsIterRef<T, Row, Col> {
        RowsIterRef::new(&self.0)
    }

    /// Returns an iterator over columns of this matrix by reference.
    #[inline]
    pub fn cols_iter_ref(&self) -> ColsIterRef<T, Row, Col> {
        ColsIterRef::new(&self.0)
    }
}
