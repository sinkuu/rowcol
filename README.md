# rowcol

[![Build Status](https://travis-ci.org/sinkuu/rowcol.svg?branch=master)](https://travis-ci.org/sinkuu/rowcol) [![crates.io](https://img.shields.io/crates/v/rowcol.svg?style=flat)](https://crates.io/crates/rowcol) [![docs.rs](https://docs.rs/rowcol/badge.svg)](https://docs.rs/rowcol)

`rowcol` crate provides fixed-size `Vector` and `Matrix`.
Their length or number of rows/columns are specified as type parameters,
using type-level numbers provided by `typenum` crate.
This lets vectors and matrices be internally represented as fixed-length arrays,
therefore they are allocated on the stack, and becomes `Copy` if their content
is `Copy`. Also, errors like computing the determinant of a non-square matrix
can be detected at compile-time, instead of causing runtime panic.
