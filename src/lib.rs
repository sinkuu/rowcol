//! rowcol crate provides fixed-size [`Vector`] and [`Matrix`].
//! Their length or number of rows/columns are specified as type parameters,
//! using type-level numbers provided by [typenum](https://docs.rs/crate/typenum/1.5.1) crate.
//! This lets vectors and matrices be internally represented as fixed-length arrays,
//! therefore they are allocated on the stack, and becomes `Copy` if their content
//! is `Copy`. Also, errors like computing the determinant of a non-square matrix
//! can be detected at compile-time, instead of causing runtime panic.
//!
//! ```rust
//! #[macro_use] extern crate rowcol;
//! use rowcol::prelude::*;
//!
//! fn fib(n: usize) -> u64 {
//!     // inferred as `f: Vector<u64, U2>`
//!     let f = vector![1, 0];
//!
//!     // inferred as `a: Matrix<u64, U2, U2>`
//!     let a = matrix![[1, 1], [1, 0]];
//!
//!     (a.pow(n) * f)[1]
//! }
//!
//! # fn main() {
//! assert_eq!(fib(0), 0);
//! assert_eq!(fib(10), 55);
//! assert_eq!(fib(50), 12586269025);
//! # }
//! ```
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
#[macro_use] extern crate approx;

pub mod prelude;
pub mod vector;
pub mod matrix;
#[macro_use] mod macros;
mod util;

