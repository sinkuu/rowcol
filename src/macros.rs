#[macro_export]
macro_rules! vector {
    (@count) => {
        $crate::typenum::consts::U0
    };

    (@count $elem:expr) => {
        $crate::typenum::consts::U1
    };

    (@count $first:expr,$($rest:tt),*) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add< vector!(@count $($rest),*) >>::Output
    };

    ($( $elem:expr ),*) => {
        $crate::vector::Vector::<_, vector!(@count $($elem),*)>::new([ $($elem),* ])
    };
}

#[macro_export]
macro_rules! matrix {
    (@countrow) => { $crate::typenum::consts::U0 };

    (@countrow [$($elem:expr),*] ) => { $crate::typenum::consts::U1 };

    (@countrow [$($elem:expr),*], $($rem:tt)*) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add<matrix!(@countrow $($rem)*)>>::Output
    };

    (@countcol) => { $crate::typenum::consts::U0 };

    (@countcol []) => { $crate::typenum::consts::U0 };

    (@countcol [$first:expr] ) => {
        $crate::typenum::consts::U1
    };

    (@countcol [$first:expr, $($rest:expr),*] ) => {
        <$crate::typenum::consts::U1 as ::std::ops::Add<matrix!(@countcol [$($rest),*])>>::Output
    };

    (@countcol [$($elem:expr),*], $($rem:tt)*) => { matrix!(@countcol [$($elem),*]) };

    ($( [ $($elem:expr),* ] ),*) => {
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
    let v0: Vector<i32, _> = vector![];
    assert_eq!(v0.len(), 0);
    assert!(v0.is_empty());
    assert_eq!(vector![1].len(), 1);
    assert_eq!(vector![1, 2].len(), 2);

    assert_eq!(matrix![[1], [2]].dim(), (2, 1));
    assert_eq!(matrix![[1, 2]].dim(), (1, 2));
    let m00: Matrix<f32, _, _> = matrix![];
    assert_eq!(m00.dim(), (0, 0));
    let m20: Matrix<f32, _, _> = matrix![[],[]];
    assert_eq!(m20.dim(), (2, 0));
}
