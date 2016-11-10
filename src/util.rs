use typenum;
use typenum::operator_aliases::Compare;
use typenum::{Greater, Equal, Less};

pub type Min<A, B> = <(A, B, Compare<A, B>) as MinImpl>::Output;

pub trait MinImpl {
    type Output: typenum::Unsigned;
}

impl<A, B> MinImpl for (A, B, Equal)
    where A: typenum::Unsigned, B: typenum::Unsigned,
{
    type Output = A;
}

impl<A, B> MinImpl for (A, B, Greater)
    where A: typenum::Unsigned, B: typenum::Unsigned,
{
    type Output = B;
}

impl<A, B> MinImpl for (A, B, Less)
    where A: typenum::Unsigned, B: typenum::Unsigned,
{
    type Output = A;
}

#[test]
fn test_typenum_min() {
    use typenum::consts::*;
    use typenum::Unsigned;

    assert_eq!(<Min<U3, U5>>::to_usize(), 3);
    assert_eq!(<Min<U6, U5>>::to_usize(), 5);
    assert_eq!(<Min<U4, U4>>::to_usize(), 4);
}
