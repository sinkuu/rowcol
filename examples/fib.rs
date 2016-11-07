extern crate rowcol;
extern crate num;

use rowcol::prelude::*;

fn main() {
    let f = Vector::<u64, U2>::new([1, 0]);
    let a = Matrix::<u64, U2, U2>::new([[1, 1], [1, 0]]);

    let mut p = Matrix::<u64, U2, U2>::identity();

    for n in 0..51 {
        println!("fib({:2}) = {}", n, (p * f)[1]);
        p *= a;
    }
}
