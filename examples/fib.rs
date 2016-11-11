#[macro_use] extern crate rowcol;
extern crate num;

use rowcol::prelude::*;

fn main() {
    let f: Vector<u64, _> = vector![1, 0];
    let a: Matrix<u64, _, _> = matrix![[1, 1], [1, 0]];

    // p: Matrix<u64, U2, U2> (inferred)
    let mut p = Matrix::identity();

    for n in 0..50 {
        println!("fib({:2}) = {}", n, (p * f)[1]);
        p *= a;
    }
}
