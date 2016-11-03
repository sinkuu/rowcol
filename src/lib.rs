//! `rowcol` crate provides fixed-size [`Vector`] and [`Matrix`].
//! Their length or number of rows/columns are provided as type parameter
//! using type-level numerics provided by `typenum` crate.
//! This let vectors and matrices be internally represented as fixed-length arrays,
//! therefore they are allocated on the stack, and becomes `Copy` if their content
//! is `Copy`. Also, errors like computing the determinant of a non-square matrix
//! can be detected at compile-time, instead of runtime panic.
//!
//! [`Vector`]: ./vector/struct.Vector.html
//! [`Matrix`]: ./vector/struct.Matrix.html

#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(not(feature = "std"))]
extern crate core as std;

pub extern crate typenum;
extern crate arrayvec;
extern crate num;

pub mod prelude;
pub mod vector;
pub mod matrix;

pub use vector::Vector;
pub use matrix::Matrix;

