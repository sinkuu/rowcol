/// Creates a [`Vector`] containing given elements.
///
/// # Example
///
/// ```rust
/// # #[macro_use] extern crate rowcol;
/// # use rowcol::prelude::*;
/// # fn main() {
/// assert_eq!(vector![1, 2, 3, 4] + vector![4, 3, 2, 1],
///            vector![5, 5, 5, 5]);
/// # }
/// ```
///
/// # Note
///
/// The compiler cannot figure out the type of `Vector::new([1.0, 2.0])` from itself.
/// This macro counts the number of elements you provided, and hints it to the compiler.
///
/// That is to say,
///
/// ```rust,ignore
/// vector![1.0, 2.0]
/// ```
///
/// is equivalent to
///
/// ```rust,ignore
/// Vector::<_, U2>::new([1.0, 2.0])
/// ```
///
/// [`Vector`]: ./vector/struct.Vector.html
#[macro_export]
macro_rules! vector {
    (@count) => {
        $crate::typenum::consts::U0
    };

    (@count $elem:expr) => {
        $crate::typenum::consts::U1
    };

    (@count $first:expr, $($rest:tt),*) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add< vector!(@count $($rest),*) >>::Output
    };

    ($($elem:expr),*) => {
        $crate::vector::Vector::<_, vector!(@count $($elem),*)>::new([ $($elem),* ])
    };
}

/// Creates a [`Matrix`] from given elements.
///
/// # Example
///
/// ```rust
/// # #[macro_use] extern crate rowcol;
/// # use rowcol::prelude::*;
/// # fn main() {
/// let m1 = matrix![[1, 2], [3, 4]];
/// assert_eq!(m1 * 2, matrix![[2, 4], [6, 8]]);
/// # }
/// ```
///
/// # Note
///
/// The compiler cannot figure out the type of `Matrix::new([[1.0, 2.0], [3.0, 4.0]])`
/// from itself. This macro counts the number of elements you provided, and hints it to the compiler.
///
/// That is to say,
///
/// ```rust,ignore
/// matrix![[1.0, 2.0], [3.0, 4.0]]
/// ```
///
/// is equivalent to
///
/// ```rust,ignore
/// Matrix::<_, U2, U2>::new([[1.0, 2.0], [3.0, 4.0]])
/// ```
///
/// [`Matrix`]: ./matrix/struct.Matrix.html
#[macro_export]
macro_rules! matrix {
    (@countrow) => { $crate::typenum::consts::U0 };

    (@countrow [$($elem:expr),*] ) => { $crate::typenum::consts::U1 };

    (@countrow [$($elem:expr),*], $($rest:tt)*) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add<matrix!(@countrow $($rest)*)>>::Output
    };

    (@countcol) => { $crate::typenum::consts::U0 };

    (@countcol []) => { $crate::typenum::consts::U0 };

    (@countcol [$first:expr] ) => {
        $crate::typenum::consts::U1
    };

    (@countcol [$first:expr, $($rest:expr),*] ) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add<matrix!(@countcol [$($rest),*])>>::Output
    };

    (@countcol [$($elem:expr),*], $($drop:tt)*) => { matrix!(@countcol [$($elem),*]) };

    ($( [$($elem:expr),*] ),*) => {
        {
            $crate::matrix::Matrix::<_,
                matrix!(@countrow $( [ $( $elem ),* ] ),*),
                matrix!(@countcol $( [ $( $elem ),* ] ),*)>::new([ $([ $($elem),* ]),* ])
        }
    };
}

#[test]
fn test_macro() {
    use ::prelude::*;

    assert_eq!(vector![1.0, 2.0], Vector::<f32, U2>::new([1.0, 2.0]));

    let v0: Vector<i32, _> = vector![];
    assert_eq!(v0.len(), 0);
    assert!(v0.is_empty());
    assert_eq!(vector![1].len(), 1);
    assert_eq!(vector![1, 2].len(), 2);

    assert_eq!(matrix![[1.0, 2.0], [3.0, 4.0]],
               Matrix::<f32, U2, U2>::new([[1.0, 2.0], [3.0, 4.0]]));

    assert_eq!(matrix![[1], [2]].dim(), (2, 1));
    assert_eq!(matrix![[1, 2]].dim(), (1, 2));
    let m00: Matrix<f32, _, _> = matrix![];
    assert_eq!(m00.dim(), (0, 0));
    let m20: Matrix<f32, _, _> = matrix![[],[]];
    assert_eq!(m20.dim(), (2, 0));
}
