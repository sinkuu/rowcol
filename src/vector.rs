use typenum;
use typenum::uint::UInt;
use typenum::consts::*;
use typenum::operator_aliases::Mod;
use typenum::type_operators::Same;

use num;
use num::Float;

use nodrop::NoDrop;

use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div, Neg, Rem,
    AddAssign, SubAssign, MulAssign, DivAssign, Index, IndexMut};
use std::marker::PhantomData;
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};
use std::mem;
use std::ptr;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::iter::FromIterator;
use std::cmp::{self, Ordering};


/// A fixed-size vector whose elements are allocated on the stack.
///
/// ```rust
/// # #[macro_use] extern crate rowcol;
/// # use rowcol::prelude::*;
/// # fn main() {
/// let arr = vector![1, 2, 3, 4, 5];
/// assert_eq!(*arr, [1, 2, 3, 4, 5]);
/// # }
/// ```
pub struct Vector<T, N: ArrayLen<T>>(N::Array);

impl<T, N: ArrayLen<T>> Vector<T, N> {
    /// Creates a vector from an array. Using [`vector` macro](../macro.vector.html) is more
    /// convenient.
    #[inline]
    pub fn new(array: N::Array) -> Self {
        Vector(array)
    }

    #[inline]
    pub fn generate<F>(f: F) -> Self where F: FnMut(usize) -> T {
        (0..N::to_usize()).map(f).collect()
    }

    /// Returns the inner array.
    #[inline]
    pub fn into_inner(self) -> N::Array {
        self.0
    }

    /// Returns a slice of entire vector.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.0.as_ref()
    }

    /// Returns a mutable slice of entire vector.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }

    /// Returns an iterator over this vector.
    #[inline]
    pub fn iter(&self) -> SliceIter<T> {
        self.as_slice().iter()
    }

    /// Returns an mutable iterator over this vector.
    #[inline]
    pub fn iter_mut(&mut self) -> SliceIterMut<T> {
        self.as_slice_mut().iter_mut()
    }

    /// Splits this vector into chunks of `I`-length vectors. `N % I` must be zero.
    ///
    /// ```rust
    /// # use rowcol::prelude::*;
    /// let v = Vector::<i32, U4>::new([1, 2, 3, 4]);
    /// let mut it = v.into_chunks::<U2>();
    /// assert_eq!(it.next(), Some(Vector::new([1, 2])));
    /// assert_eq!(it.next(), Some(Vector::new([3, 4])));
    /// assert_eq!(it.next(), None);
    /// ```
    #[inline]
    pub fn into_chunks<I>(self) -> Chunks<T, N, I>
        where
            N: Rem<I>,
            Mod<N, I>: Same<U0>
    {
        Chunks::new(self.into_iter())
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> &T {
        debug_assert!(i < self.len());
        self.as_slice().get_unchecked(i)
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        debug_assert!(i < self.len());
        self.as_slice_mut().get_unchecked_mut(i)
    }

    #[inline]
    pub fn len(&self) -> usize {
        N::to_usize()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        N::to_usize() == 0
    }
}

impl<T, N> Debug for Vector<T, N>
    where T: Debug, N: ArrayLen<T>
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        try!(f.write_str("Vector"));
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, N> Clone for Vector<T, N>
    where
        T: Clone,
        N: ArrayLen<T>
{
    #[inline]
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

impl<T, N> Copy for Vector<T, N>
    where
        T: Copy,
        N::Array: Copy,
        N: ArrayLen<T>
{
}

impl<T, N> Default for Vector<T, N>
    where
        T: Default,
        N: ArrayLen<T>,
{
    #[inline]
    fn default() -> Self {
        Vector::generate(|_| T::default())
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

impl<T, N> num::Zero for Vector<T, N> where T: num::Zero + Clone, N: ArrayLen<T> {
    #[inline]
    fn zero() -> Self {
        Vector::generate(|_| T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iter().all(num::Zero::is_zero)
    }
}

impl<T, N> num::Bounded for Vector<T, N> where T: num::Bounded, N: ArrayLen<T> {
    #[inline]
    fn min_value() -> Self {
        Vector::generate(|_| T::min_value())
    }

    #[inline]
    fn max_value() -> Self {
        Vector::generate(|_| T::max_value())
    }
}

impl<T, N> Index<usize> for Vector<T, N> where N: ArrayLen<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &T {
        &self.as_slice()[idx]
    }
}

impl<T, N> IndexMut<usize> for Vector<T, N> where N: ArrayLen<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.as_slice_mut()[idx]
    }
}

impl<T, N, IU, IB> Index<UInt<IU, IB>> for Vector<T, N>
    where
        N: ArrayLen<T>,
        IU: typenum::Unsigned,
        IB: typenum::Bit,
        UInt<IU, IB>: typenum::Cmp<N>,
        typenum::Compare<UInt<IU, IB>, N>: Same<typenum::Less>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: UInt<IU, IB>) -> &T {
        unsafe {
            self.get_unchecked(<UInt<IU, IB> as typenum::Unsigned>::to_usize())
        }
    }
}

impl<T, N> Index<U0> for Vector<T, N>
    where N: ArrayLen<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, _: U0) -> &T {
        unsafe {
            self.get_unchecked(0)
        }
    }
}

impl<T, N, IU, IB> IndexMut<UInt<IU, IB>> for Vector<T, N>
    where
        N: ArrayLen<T>,
        IU: typenum::Unsigned,
        IB: typenum::Bit,
        UInt<IU, IB>: typenum::Cmp<N>,
        typenum::Compare<UInt<IU, IB>, N>: Same<typenum::Less>,
{
    #[inline]
    fn index_mut(&mut self, _: UInt<IU, IB>) -> &mut T {
        unsafe {
            self.get_unchecked_mut(<UInt<IU, IB> as typenum::Unsigned>::to_usize())
        }
    }
}

impl<T, N> IndexMut<U0> for Vector<T, N>
    where N: ArrayLen<T>,
{
    #[inline]
    fn index_mut(&mut self, _: U0) -> &mut T {
        unsafe {
            self.get_unchecked_mut(0)
        }
    }
}

macro_rules! impl_vector_arith {
    (T T : $op_trait:ident, $op_fn:ident) => {
        impl<T, U, N> $op_trait<Vector<U, N>> for Vector<T, N>
            where
                T: $op_trait<U> ,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<T as $op_trait<U>>::Output>,
        {
            type Output = Vector<<T as $op_trait<U>>::Output, N>;

            fn $op_fn(self, other: Vector<U, N>) -> Self::Output {
                self.into_iter().zip(other).map(|(a, b)| $op_trait::$op_fn(a, b)).collect()
            }
        }
    };

    (T &T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, T, U, N> $op_trait<&'a Vector<U, N>> for Vector<T, N>
            where
                T: $op_trait<&'a U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<T as $op_trait<&'a U>>::Output>,
        {
            type Output = Vector<<T as $op_trait<&'a U>>::Output, N>;

            fn $op_fn(self, other: &'a Vector<U, N>) -> Self::Output {
                self.into_iter().zip(other.iter()).map(|(a, b)| $op_trait::$op_fn(a, b)).collect()
            }
        }
    };

    (&T T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, T, U, N> $op_trait<Vector<U, N>> for &'a Vector<T, N>
            where
                &'a T: $op_trait<U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<&'a T as $op_trait<U>>::Output>,
        {
            type Output = Vector<<&'a T as $op_trait<U>>::Output, N>;

            fn $op_fn(self, other: Vector<U, N>) -> Self::Output {
                self.iter().zip(other).map(|(a, b)| $op_trait::$op_fn(a, b)).collect()
            }
        }
    };

    (&T &T : $op_trait:ident, $op_fn:ident) => {
        impl<'a, 'b, T, U, N> $op_trait<&'a Vector<U, N>> for &'b Vector<T, N>
            where
                &'b T: $op_trait<&'a U>,
                N: ArrayLen<T> + ArrayLen<U> + ArrayLen<<&'b T as $op_trait<&'a U>>::Output>,
        {
            type Output = Vector<<&'b T as $op_trait<&'a U>>::Output, N>;

            fn $op_fn(self, other: &'a Vector<U, N>) -> Self::Output {
                self.iter().zip(other.iter()).map(|(a, b)| $op_trait::$op_fn(a, b)).collect()
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

impl<T, U, N> AddAssign<Vector<U, N>> for Vector<T, N>
    where
        T: AddAssign<U>,
        N: ArrayLen<T> + ArrayLen<U>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for (a, b) in self.iter_mut().zip(rhs) {
            *a += b;
        }
    }
}

impl<T, U, N> SubAssign<Vector<U, N>> for Vector<T, N>
    where
        T: SubAssign<U>,
        N: ArrayLen<T> + ArrayLen<U>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for (a, b) in self.iter_mut().zip(rhs) {
            *a -= b;
        }
    }
}

impl<T, U, N> Mul<U> for Vector<T, N>
    where
        T: Mul<U>,
        U: Clone,
        N: ArrayLen<T> + ArrayLen<<T as Mul<U>>::Output>,
{
    type Output = Vector<<T as Mul<U>>::Output, N>;

    #[inline]
    fn mul(self, rhs: U) -> Self::Output {
        self.into_iter().map(|e| e * rhs.clone()).collect()
    }
}

impl<T, U, N> MulAssign<U> for Vector<T, N>
    where
        T: MulAssign<U>,
        U: Clone,
        N: ArrayLen<T>,
{
    fn mul_assign(&mut self, rhs: U) {
        for a in self.iter_mut() {
            *a *= rhs.clone();
        }
    }
}

impl<T, U, N> Div<U> for Vector<T, N>
    where
        T: Div<U>,
        U: Clone,
        N: ArrayLen<T> + ArrayLen<<T as Div<U>>::Output>,
{
    type Output = Vector<<T as Div<U>>::Output, N>;

    #[inline]
    fn div(self, rhs: U) -> Self::Output {
        self.into_iter().map(|e| e / rhs.clone()).collect()
    }
}

impl<T, U, N> DivAssign<U> for Vector<T, N>
    where
        T: DivAssign<U>,
        U: Clone,
        N: ArrayLen<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: U) {
        for a in self.iter_mut() {
            *a /= rhs.clone();
        }
    }
}


impl<T, N> Neg for Vector<T, N>
    where T: Neg, N: ArrayLen<T> + ArrayLen<<T as Neg>::Output>
{
    type Output = Vector<<T as Neg>::Output, N>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.into_iter().map(|e| -e).collect()
    }
}

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

    assert_eq!(a * 2, Vector::new([2, 4, 6]));
    assert_eq!(b / 2, Vector::new([2, 2, 3]));

    let mut a = a;
    a *= 4;
    assert_eq!(a, Vector::new([4, 8, 12]));
    a /= 2;
    assert_eq!(a, Vector::new([2, 4, 6]));
    a -= Vector::new([1, 1, 1]);
    assert_eq!(a, Vector::new([1, 3, 5]));
    a += Vector::new([1, 2, 3]);
    assert_eq!(a, Vector::new([2, 5, 8]));
}

impl<T, N> Vector<T, N>
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + num::Zero + Clone,
        N: ArrayLen<T>,
{
    /// Returns the dot product of this vector and the other.
    pub fn dot(&self, other: &Vector<T, N>) -> T {
        self.iter().cloned().zip(other.iter().cloned()).map(|(a, b)| a * b)
            .fold(T::zero(), Add::add)
    }
}

#[test]
fn test_dot() {
    let v  = Vector::<i32, U3>::new([1, 2, 3]);
    let v2 = Vector::<i32, U3>::new([4, 5, 6]);
    assert_eq!(v.dot(&v2), 32);
}

impl<T> Vector<T, U3>
    where
        T: Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    pub fn cross(&self, other: &Vector<T, U3>) -> Vector<T, U3> {
        macro_rules! idx {
            ($mat:ident, $i:ident) => {
                $mat[$i::new()].clone()
            }
        }

        Vector::new([
            idx!(self,U1)*idx!(other,U2) - idx!(self,U2)*idx!(other,U1),
            idx!(self,U2)*idx!(other,U0) - idx!(self,U0)*idx!(other,U2),
            idx!(self,U0)*idx!(other,U1) - idx!(self,U1)*idx!(other,U0),
        ])
    }
}

#[test]
fn test_cross() {
    let v  = Vector::<i32, U3>::new([1, 2, 3]);
    let v2 = Vector::<i32, U3>::new([4, 5, 6]);
    assert_eq!(v.cross(&v2), Vector::new([-3, 6, -3]));
}

impl<T, N> Vector<T, N>
    where
        T: Float + Clone,
        N: ArrayLen<T>,
{
    pub fn norm(&self) -> T {
        let prod = self.iter().cloned().map(|x| x.powi(2)).fold(T::zero(), Add::add);
        prod.sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Vector<T, N> {
        self.clone() / self.norm()
    }
}

#[test]
fn test_norm() {
    let v = Vector::<f32, U2>::new([3.0, 4.0]);
    assert_relative_eq!(v.norm(), 5.0);
    let n = v.normalized();
    assert_relative_eq!(n[0], 0.6);
    assert_relative_eq!(n[1], 0.8);
}

impl<T, N> Vector<T, N>
    where
        T: Clone + num::Zero + Add<T, Output = T>,
        N: ArrayLen<T>,
{
    /// Returns the sum of the entries in this vector.
    pub fn sum(&self) -> T {
        self.iter().cloned().fold(T::zero(), Add::add)
    }
}

#[test]
fn test_sum() {
    let v = Vector::<i32, U2>::new([1, 2]);
    assert_eq!(v.sum(), 3);
}

impl<T, U, B> Vector<T, UInt<U, B>>
    where
        U: typenum::Unsigned,
        B: typenum::Bit,
        T: Clone + Ord,
        UInt<U, B>: ArrayLen<T>,
{
    /// Returns the maximum of all elements of this vector.
    ///
    /// `Vector<T, U0>` does not have this method.
    #[inline]
    pub fn max(&self) -> T {
        // this `unwrap` never panic
        self.iter().cloned().max().unwrap()
    }

    /// Returns the minimum of all elements of this vector.
    ///
    /// `Vector<T, U0>` does not have this method.
    #[inline]
    pub fn min(&self) -> T {
        // this `unwrap` never panic
        self.iter().cloned().min().unwrap()
    }
}

impl<T, U, B> Vector<T, UInt<U, B>>
    where
        U: typenum::Unsigned,
        B: typenum::Bit,
        T: Clone,
        UInt<U, B>: ArrayLen<T>,
{
    /// Returns the maximum of all elements of this vector, using supplied comparison function.
    ///
    /// `Vector<T, U0>` does not have this method.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rowcol::prelude::*;
    /// let v = Vector::<i32, U3>::new([-1, 2, -3]);
    /// assert_eq!(v.max(), 2);
    /// assert_eq!(v.max_by(|a, b| a.abs().cmp(&b.abs())), -3);
    /// ```
    pub fn max_by<F>(&self, mut comp: F) -> T where F: FnMut(&T, &T) -> Ordering {
        let mut it = self.iter();
        let mut st = it.next().unwrap(); // this `unwrap` never panic

        for v in it {
            if comp(v, st) == Ordering::Greater {
                st = v;
            }
        }

        st.clone()
    }

    /// Returns the minimum of all elements of this vector, using supplied comparison function.
    ///
    /// `Vector<T, U0>` does not have this method.
    pub fn min_by<F>(&self, mut comp: F) -> T where F: FnMut(&T, &T) -> Ordering {
        self.max_by(|a, b| comp(b, a))
    }
}

#[test]
fn test_minmax() {
    let v = Vector::<i32, U3>::new([1, 2, 3]);
    assert_eq!(v.max(), 3);
    assert_eq!(v.min(), 1);
    assert_eq!(v.max_by(|a, b| (a - 5).abs().cmp(&(b - 5).abs())), 1);
    assert_eq!(v.min_by(|a, b| (a - 5).abs().cmp(&(b - 5).abs())), 3);
}

impl<T, N> Vector<T, N>
    where
        T: Clone + Ord,
        N: ArrayLen<T>,
{
    /// Returns the clamped version of this vector to `[min, max]`.
    pub fn clamped(&self, min: T, max: T) -> Vector<T, N> {
        self.iter().cloned().map(|v| cmp::min(&max, cmp::max(&min, &v)).clone()).collect()
    }

    /// Clamps this vector to `[min, max]`.
    pub fn clamp(&mut self, min: T, max: T) {
        for v in self.iter_mut() {
            *v = cmp::min(&max, cmp::max(&min, v)).clone();
        }
    }
}

#[test]
fn test_clamp() {
    let mut v = Vector::<i32, U3>::new([1, 5, 8]);
    assert_eq!(v.clamped(4, 6), Vector::new([4, 5, 6]));
    v.clamp(3, 4);
    assert_eq!(v, Vector::new([3, 4, 4]));
}

impl<T, N> FromIterator<T> for Vector<T, N> where N: ArrayLen<T> {
    fn from_iter<I>(iter: I) -> Self
        where I: IntoIterator<Item = T>
    {
        let mut it = iter.into_iter();

        let arr = unsafe {
            // NOTE: somehow `NoDrop` drops performance - it should be optimized out
            let mut arr = NoDrop::new(mem::uninitialized::<N::Array>());

            for i in 0..N::to_usize() {
                let item = match it.next() {
                    Some(item) => item,
                    None => {
                        let mut arr = arr.into_inner();
                        for j in 0..i {
                            mem::drop(ptr::read(arr.as_mut().get_unchecked_mut(j)));
                        }
                        mem::forget(arr);

                        panic!("Vector<_, U{0}> can only be created with exactly {0} elements.",
                               N::to_usize())
                    }
                };
                ptr::write(arr.as_mut().get_unchecked_mut(i), item);
            }

            // making this `assert_eq` slows down matrix multiplication by 7x in release build!
            debug_assert_eq!(it.count(), 0,
                             "Vector<_, U{0}> can only be created with exactly {0} elements.",
                             N::to_usize());

            arr.into_inner()
        };

        Vector::new(arr)
    }
}

impl<T, N> IntoIterator for Vector<T, N> where N: ArrayLen<T> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            arr: NoDrop::new(self.into_inner()),
            next: 0,
            back: N::to_usize(),
        }
    }
}

pub struct IntoIter<T, N> where N: ArrayLen<T> {
    arr: NoDrop<N::Array>,
    next: usize,
    back: usize,
}

impl<T, N> Drop for IntoIter<T, N> where N: ArrayLen<T> {
    fn drop(&mut self) {
        for i in self.next..self.back {
            mem::drop(unsafe { ptr::read(self.arr.as_ref().get_unchecked(i)) });
        }
    }
}

impl<T, N> Iterator for IntoIter<T, N> where N: ArrayLen<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        debug_assert!(self.back <= N::to_usize());

        if self.next < self.back {
            let i = self.next;
            self.next += 1;

            Some(unsafe {
                ptr::read(self.arr.as_ref().get_unchecked(i))
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<T, N> DoubleEndedIterator for IntoIter<T, N> where N: ArrayLen<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.back <= N::to_usize());

        if self.back > 0 {
            self.back -= 1;

            Some(unsafe {
                ptr::read(self.arr.as_ref().get_unchecked(self.back))
            })
        } else {
            None
        }
    }
}

#[test]
fn test_vec_intoiter_next_back() {
    let v = Vector::<i32, U3>::new([1, 2, 3]);
    assert!(v.into_iter().rev().eq(vec![3, 2, 1]));
}

impl<T, N> ExactSizeIterator for IntoIter<T, N> where N: ArrayLen<T> {
    #[inline]
    fn len(&self) -> usize {
        self.back - self.next
    }
}

pub struct Chunks<T, N, I> where N: ArrayLen<T> {
    it: IntoIter<T, N>,
    _i: PhantomData<I>,
}

impl<T, N, I> Chunks<T, N, I> where N: ArrayLen<T> {
    fn new(it: IntoIter<T, N>) -> Self {
        Chunks {
            it: it,
            _i: PhantomData,
        }
    }
}

impl<T, N, I> Iterator for Chunks<T, N, I>
    where
        N: ArrayLen<T> + Rem<I>,
        I: ArrayLen<T>,
        Mod<N, I>: Same<U0>,
{
    type Item = Vector<T, I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.it.len() == 0 { // TODO: use is_empty() when it is stabilized
            None
        } else {
            Some((&mut self.it).take(I::to_usize()).collect())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.it.len() / I::to_usize();
        (len, Some(len))
    }
}

impl<T, N, I> DoubleEndedIterator for Chunks<T, N, I>
    where
        N: ArrayLen<T> + Rem<I>,
        I: ArrayLen<T>,
        Mod<N, I>: Same<U0>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.it.len() == 0 {
            None
        } else {
            Some((&mut self.it).rev().take(I::to_usize()).collect::<Vector<T, I>>().into_iter()
                 .rev().collect())
        }
    }
}

impl<T, N, I> ExactSizeIterator for Chunks<T, N, I>
    where
        N: ArrayLen<T> + Rem<I>,
        I: ArrayLen<T>,
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

    let mut it = arr.into_chunks::<U2>().rev();
    assert_eq!(it.next(), Some(Vector::new([5, 6])));
    assert_eq!(it.next(), Some(Vector::new([3, 4])));
    assert_eq!(it.next(), Some(Vector::new([1, 2])));
    assert_eq!(it.next(), None);
}

pub trait ArrayLen<T>: typenum::Unsigned {
    type Array: AsRef<[T]> + AsMut<[T]>;
}

macro_rules! impl_arraylen {
    ($tn:ident, $len:expr) => {
        impl<T> ArrayLen<T> for $tn {
            type Array = [T; $len];
        }
    }
}

impl_arraylen!(U0, 0);
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
    // rustc bug (broken MIR) https://github.com/rust-lang/rust/issues/28828
    // use typenum::Diff;
    // let a: Vector<i32, Diff<U8, U3>> = Default::default();
    let a: Vector<i32, <U8 as Sub<U3>>::Output> = Default::default();
    assert_eq!(a.len(), 5);
    let _: [i32; 5] = a.0;
}

