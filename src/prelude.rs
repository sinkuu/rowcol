//! rowcol prelude.

pub use matrix::{Matrix};
pub use matrix::ops::{Determinant, Cofactor, Inverse};
pub use vector::Vector;
pub use typenum::consts::*;

pub type Vector1f32 = Vector<f32, U1>;
pub type Vector1f64 = Vector<f64, U1>;
pub type Vector2f32 = Vector<f32, U2>;
pub type Vector2f64 = Vector<f64, U2>;
pub type Vector3f32 = Vector<f32, U3>;
pub type Vector3f64 = Vector<f64, U3>;
pub type Vector4f32 = Vector<f32, U4>;
pub type Vector4f64 = Vector<f64, U4>;
pub type Vector5f32 = Vector<f32, U5>;
pub type Vector5f64 = Vector<f64, U5>;
pub type Vector6f32 = Vector<f32, U6>;
pub type Vector6f64 = Vector<f64, U6>;
pub type Vector7f32 = Vector<f32, U7>;
pub type Vector7f64 = Vector<f64, U7>;
pub type Vector8f32 = Vector<f32, U8>;
pub type Vector8f64 = Vector<f64, U8>;
pub type Vector9f32 = Vector<f32, U9>;
pub type Vector9f64 = Vector<f64, U9>;
pub type Vector10f32 = Vector<f32, U10>;
pub type Vector10f64 = Vector<f64, U10>;

pub type Matrix2f32 = Matrix<f32, U2, U2>;
pub type Matrix2f64 = Matrix<f32, U2, U2>;
pub type Matrix3f32 = Matrix<f32, U3, U3>;
pub type Matrix3f64 = Matrix<f64, U3, U3>;
pub type Matrix4f32 = Matrix<f32, U4, U4>;
pub type Matrix4f64 = Matrix<f64, U4, U4>;
