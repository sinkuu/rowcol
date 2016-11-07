#![feature(test)]

extern crate rowcol;
use rowcol::prelude::*;

extern crate test;
use test::Bencher;

#[bench]
fn bench_mul(b: &mut Bencher) {
    let m = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));
    let v = Vector::<f32, U8>::generate(|i| i as f32);

    b.iter(|| {
        let mut m = m;

        m *= 2.5;
        m *= m;
        m = m * m;
        m * v
    })
}

#[bench]
fn bench_div(b: &mut Bencher) {
    let m = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));

    b.iter(|| {
        let mut m = m;

        m /= 2.5;
        m = m / 1.3;
        m = m / &1.3;
        m
    })
}

#[bench]
fn bench_add_sub(b: &mut Bencher) {
    let m1 = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));
    let m2 = Matrix::<f32, U8, U8>::generate(|(i, j)| 1.234 * (j as f32 - i as f32).powi(i as i32));

    b.iter(|| {
        let mut m = m1;

        m += m2;
        m = m + m1;
        m = m + &m1;
        m -= m1;
        m = m + m2;
        m = m + &m2;
        m
    })
}

#[bench]
fn bench_inv(b: &mut Bencher) {
    let m = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));

    b.iter(|| m.inverse().unwrap())
}

#[bench]
fn bench_det(b: &mut Bencher) {
    let m = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));

    b.iter(|| m.determinant())
}

#[bench]
fn bench_transpose(b: &mut Bencher) {
    let m = Matrix::<f32, U8, U8>::generate(|(i, j)| 0.12 * ((j+i) as f32).powi(i as i32));

    b.iter(|| m.transposed());
}
