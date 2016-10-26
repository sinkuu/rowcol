use typenum::consts::*;
use typenum::operator_aliases::Mod;
use typenum::type_operators::Same;
use typenum::marker_traits::Unsigned;

use arrayvec::{self, ArrayVec};

use std::ops::{Deref, DerefMut, Add, Sub, Rem};
use std::marker::PhantomData;

/// A fixed-size vector whose elements are allocated on the stack.
///
/// ```rust
/// # use static_matrix::typenum::consts::*;
/// # use static_matrix::Vector;
///
/// let arr = Vector::<i32, U5>::new([1, 2, 3, 4, 5]);
/// assert_eq!(*arr, [1, 2, 3, 4, 5]);
/// ```
#[derive(Debug)]
pub struct Vector<T, N: ArrayLen<T>>(N::Array);

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
    pub fn as_slice(&self) -> &[T] {
        self.0.as_ref()
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        Iter(self.as_slice().iter())
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

impl<T, N> AsRef<[T]> for Vector<T, N>
    where N: ArrayLen<T>
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, N> AsMut<[T]> for Vector<T, N>
    where N: ArrayLen<T>
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_slice_mut()
    }
}

impl<T, U, N> PartialEq<Vector<U, N>> for Vector<T, N>
    where
        T: PartialEq<U>,
        N: ArrayLen<T> + ArrayLen<U>,
{
    #[inline]
    fn eq(&self, other: &Vector<U, N>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T, N> Eq for Vector<T, N>
    where
        T: PartialEq,
        N: ArrayLen<T>,
{
}

macro_rules! impl_vector_arith {
    (T T : $op_trait:ident, $op_fn:ident) => {
        impl<T, U, N> $op_trait<Vector<U, N>> for Vector<T, N>
            where
                N: ArrayLen<T>,
                T: $op_trait<U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<T as $op_trait<U>>::Output>,
        {
            type Output = Vector<<T as $op_trait<U>>::Output, N>;

            fn $op_fn(self, other: Vector<U, N>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (a, b) in ArrayVec::from(self.into_inner()).into_iter().zip(ArrayVec::from(other.into_inner())) {
                    res.push($op_trait::$op_fn(a, b));
                }

                debug_assert!(res.is_full());
                Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (T &T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, T, U, N> $op_trait<&'a Vector<U, N>> for Vector<T, N>
            where
                N: ArrayLen<T>,
                T: $op_trait<&'a U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<T as $op_trait<&'a U>>::Output>,
        {
            type Output = Vector<<T as $op_trait<&'a U>>::Output, N>;

            fn $op_fn(self, other: &'a Vector<U, N>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (a, b) in ArrayVec::from(self.into_inner()).into_iter().zip(other.as_slice()) {
                    res.push($op_trait::$op_fn(a, b));
                }

                debug_assert!(res.is_full());
                Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (&T T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, T, U, N> $op_trait<Vector<U, N>> for &'a Vector<T, N>
            where
                N: ArrayLen<T>,
                &'a T: $op_trait<U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<&'a T as $op_trait<U>>::Output>,
        {
            type Output = Vector<<&'a T as $op_trait<U>>::Output, N>;

            fn $op_fn(self, other: Vector<U, N>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (a, b) in self.as_slice().into_iter().zip(ArrayVec::from(other.into_inner())) {
                    res.push($op_trait::$op_fn(a, b));
                }

                debug_assert!(res.is_full());
                Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };

    (&T &T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, 'b, T, U, N> $op_trait<&'a Vector<U, N>> for &'b Vector<T, N>
            where
                N: ArrayLen<T>,
                &'b T: $op_trait<&'a U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<&'b T as $op_trait<&'a U>>::Output>,
        {
            type Output = Vector<<&'b T as $op_trait<&'a U>>::Output, N>;

            fn $op_fn(self, other: &'a Vector<U, N>) -> Self::Output {
                let mut res = ArrayVec::new();

                for (a, b) in self.as_slice().into_iter().zip(other.as_slice()) {
                    res.push($op_trait::$op_fn(a, b));
                }

                debug_assert!(res.is_full());
                Vector::new(res.into_inner().unwrap_or_else(|_| unreachable!()))
            }
        }
    };
}

impl_vector_arith!(T T: Add, add);
impl_vector_arith!(T T: Sub, sub);
impl_vector_arith!(T &T: Add, add);
impl_vector_arith!(T &T: Sub, sub);
impl_vector_arith!(&T T: Add, add);
impl_vector_arith!(&T T: Sub, sub);
impl_vector_arith!(&T &T: Add, add);
impl_vector_arith!(&T &T: Sub, sub);

#[test]
fn test_vector_arith() {
    let a = Vector::<i32, U3>::new([1, 2, 3]);
    let b = Vector::new([4, 5, 6]);
    let a_plus_b = Vector::new([5, 7, 9]);
    let a_minus_b = Vector::new([-3, -3, -3]);
    assert_eq!(a + b, a_plus_b);
    assert_eq!(&a + b, a_plus_b);
    assert_eq!(a + &b, a_plus_b);
    assert_eq!(&a + &b, a_plus_b);
    assert_eq!(a - b, a_minus_b);
    assert_eq!(&a - b, a_minus_b);
    assert_eq!(a - &b, a_minus_b);
    assert_eq!(&a - &b, a_minus_b);
}

pub struct Iter<'a, T: 'a>(::std::slice::Iter<'a, T>);

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T, N> IntoIterator for Vector<T, N> where N: ArrayLen<T> {
    type Item = T;
    type IntoIter = IntoIter<T,  N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter(ArrayVec::from(self.into_inner()).into_iter())
    }
}

pub struct IntoIter<T, N>(arrayvec::IntoIter<N::Array>) where N: ArrayLen<T>;

impl<T, N> Iterator for IntoIter<T, N> where N: ArrayLen<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct VectorChunks<T, N, I> where N: ArrayLen<T> {
    it: ::arrayvec::IntoIter<N::Array>,
    _i: PhantomData<I>,
}

impl<T, N, I> VectorChunks<T, N, I> where N: ArrayLen<T> {
    fn new(a: ArrayVec<N::Array>) -> Self {
        VectorChunks {
            it: a.into_iter(),
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
        if self.it.size_hint().1 == Some(0) {
            None
        } else {
            let mut res = ArrayVec::new();
            res.extend((&mut self.it).take(I::to_usize()));
            debug_assert!(res.is_full());
            Some(Vector(res.into_inner().unwrap_or_else(|_| unreachable!())))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.it.len() / I::to_usize();
        (len, Some(len))
    }
}

impl<T, N, I> ExactSizeIterator for VectorChunks<T, N, I>
    where
        N: ArrayLen<T> + Rem<I>,
        I: Unsigned + ArrayLen<T>,
        Mod<N, I>: Same<U0>,
{
}

#[test]
fn test_vector_chunks() {
    let arr: Vector<i32, U6> = Vector::new([1, 2, 3, 4, 5, 6]);
    let mut it = arr.into_chunks::<U2>();
    let a: [i32; 2] = it.next().unwrap().into_inner();
    assert_eq!(a, [1, 2]);
    let a: [i32; 2] = it.next().unwrap().into_inner();
    assert_eq!(a, [3, 4]);
    assert_eq!(it.len(), 1);
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

