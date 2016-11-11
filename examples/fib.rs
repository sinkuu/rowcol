#[macro_use] extern crate rowcol;
extern crate num;

use rowcol::prelude::*;

fn main() {
    // f: Vector<i32, U2>
    let f = vector![1, 0];
    // a: Matrix<u32, U2, U2>
    let a = matrix![[1, 1], [1, 0]];

    // p: Matrix<u32, U2, U2> (inferred)
    let mut p = Matrix::identity();

    for n in 0..50 {
        println!("fib({:2}) = {}", n, (p * f)[1]);
        p *= a;
    }
}
