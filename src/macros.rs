#[macro_export]
macro_rules! matrix {
    (@countrow) => { typenum::consts::U0 };

    (@countrow [$($elem:expr),*] ) => { typenum::consts::U1 };

    (@countrow [$($elem:expr),*], $($rem:tt)*) => {
        <typenum::consts::U1 as ::std::ops::Add<matrix!(@countrow $($rem)*)>>::Output
    };

    (@countcol) => { typenum::consts::U0 };

    (@countcol []) => { typenum::consts::U0 };

    (@countcol [$first:expr] ) => {
        typenum::consts::U1
    };

    (@countcol [$first:expr, $($rest:expr),*] ) => {
        <typenum::consts::U1 as ::std::ops::Add<matrix!(@countcol [$($rest),*])>>::Output
    };

    (@countcol [$($elem:expr),*], $($rem:tt)*) => { matrix!(@countcol [$($elem),*]) };

    ($( [ $($elem:expr),* ] ),*) => {
        {
            use $crate::typenum;
            use $crate::matrix::Matrix;

            Matrix::<_,
                matrix!(@countrow $( [ $( $elem ),* ] ),*),
                matrix!(@countcol $( [ $( $elem ),* ] ),*)>::new([ $([ $($elem),* ]),* ])
        }
    };
}

#[test]
fn test_macro() {
    let m = matrix![[1], [2]];
    assert_eq!(m.dim(), (2, 1));
    let m = matrix![[1, 2]];
    assert_eq!(m.dim(), (1, 2));
}
