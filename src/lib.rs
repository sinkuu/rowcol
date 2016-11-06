//! `rowcol` crate provides fixed-size [`Vector`] and [`Matrix`].
//! Their length or number of rows/columns are specified as type parameters,
//! using type-level numbers provided by `typenum` crate.
//! This lets vectors and matrices be internally represented as fixed-length arrays,
//! therefore they are allocated on the stack, and becomes `Copy` if their content
//! is `Copy`. Also, errors like computing the determinant of a non-square matrix
//! can be detected at compile-time, instead of causing runtime panic.
//!
//! [`Vector`]: ./vector/struct.Vector.html
//! [`Matrix`]: ./matrix/struct.Matrix.html

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#[cfg(not(any(test, feature = "std")))]
extern crate core as std;

pub extern crate typenum;
extern crate num;
extern crate nodrop;
#[cfg(feature = "unicode_width")] extern crate unicode_width;

pub mod prelude;
pub mod vector;
pub mod matrix;
