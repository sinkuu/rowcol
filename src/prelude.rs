//! rowcol prelude.

pub use matrix::{Matrix, Determinant, Cofactor, Inverse};
pub use vector::Vector;
pub use typenum::consts::*;

type Matrix2f32 = Matrix<f32, U2, U2>;
type Matrix2f64 = Matrix<f32, U2, U2>;
type Matrix3f32 = Matrix<f32, U3, U3>;
type Matrix3f64 = Matrix<f64, U3, U3>;
type Matrix4f32 = Matrix<f32, U4, U4>;
type Matrix4f64 = Matrix<f64, U4, U4>;
