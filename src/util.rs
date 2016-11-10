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

