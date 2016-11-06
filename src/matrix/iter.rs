use ::vector::{Vector, ArrayLen};

pub struct RowsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize, usize)
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a;

impl<'a, T: 'a, Row, Col> RowsIter<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a
{
    pub fn new(vec: &'a Vector<Vector<T, Col>, Row>) -> Self {
        RowsIter(vec, 0, Row::to_usize())
    }
}

impl<'a, T: 'a, Row, Col> Iterator
    for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    type Item = Vector<T, Col>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.1 < self.2 {
            let s = self.1;
            let ret = self.0.as_ref()[s].clone();
            self.1 += 1;
            Some(ret)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = Row::to_usize() - self.1;
        (s, Some(s))
    }

    #[inline]
    fn count(self) -> usize {
        Row::to_usize() - self.1
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.1 += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    // avoiding an assert in the provided method
    #[inline]
    fn len(&self) -> usize {
        Row::to_usize() - self.1
    }
}

impl<'a, T: 'a, Row, Col> DoubleEndedIterator for RowsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>>,
        Col: ArrayLen<T>,
{
    fn next_back(&mut self) -> Option<Vector<T, Col>> {
        if self.1 < self.2 {
            let s = self.2;
            let ret = self.0.as_ref()[s-1].clone();
            self.2 -= 1;
            Some(ret)
        } else {
            None
        }
    }
}

pub struct ColsIter<'a, T: 'a, Row, Col>
    (&'a Vector<Vector<T, Col>, Row>, usize, usize)
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a;

impl<'a, T: 'a, Row, Col> ColsIter<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a
{
    pub fn new(vec: &'a Vector<Vector<T, Col>, Row>) -> Self {
        ColsIter(vec, 0, Col::to_usize())
    }
}

impl<'a, T: 'a, Row, Col> Iterator
    for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    type Item = Vector<T, Row>;

    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.2 <= Col::to_usize());
        if self.1 < self.2 {
            let s = self.1;
            self.1 += 1;

            let v = self.0.iter().map(|row| unsafe { row.as_slice().get_unchecked(s).clone() }).collect();
            Some(v)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let s = Col::to_usize() - self.1;
        (s, Some(s))
    }

    #[inline]
    fn count(self) -> usize {
        Col::to_usize() - self.1
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.1 += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    // avoiding an assert in the provided method
    #[inline]
    fn len(&self) -> usize {
        Col::to_usize() - self.1
    }
}

impl<'a, T: 'a, Row, Col> DoubleEndedIterator
    for ColsIter<'a, T, Row, Col>
    where
        T: Clone,
        Row: ArrayLen<Vector<T, Col>> + ArrayLen<T>,
        Col: ArrayLen<T>,
{
    fn next_back(&mut self) -> Option<Vector<T, Row>> {
        if self.1 < self.2 {
            let s = self.2;
            self.2 -= 1;

            Some(self.0.iter().map(|row| row.as_ref()[s-1].clone()).collect())
        } else {
            None
        }
    }
}

pub struct RowsIterRef<'a, T: 'a, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    mat: &'a Vector<Vector<T, Col>, Row>,
    front: usize,
    back: usize,
}

impl<'a, T: 'a, Row, Col> RowsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    pub fn new(mat: &'a Vector<Vector<T, Col>, Row>) -> Self {
        RowsIterRef {
            mat: mat,
            front: 0,
            back: Row::to_usize(),
        }
    }
}

impl<'a, T: 'a, Row, Col> Iterator for RowsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<&'a [T]> {
        debug_assert!(self.back <= Row::to_usize());

        if self.front < self.back {
            let i = self.front;
            self.front += 1;

            let row = unsafe { self.mat.get_unchecked(i) };

            Some(row.as_ref())
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.front += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for RowsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a, T: 'a, Row, Col> DoubleEndedIterator for RowsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    fn next_back(&mut self) -> Option<&'a [T]> {
        debug_assert!(self.back <= Row::to_usize());

        if self.front < self.back {
            self.back -= 1;
            let row = unsafe { self.mat.get_unchecked(self.back) };
            Some(row.as_ref())
        } else {
            None
        }
    }
}

pub struct ColsIterRef<'a, T: 'a, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    mat: &'a Vector<Vector<T, Col>, Row>,
    front: usize,
    back: usize,
}

impl<'a, T: 'a, Row, Col> ColsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    pub fn new(mat: &'a Vector<Vector<T, Col>, Row>) -> Self {
        ColsIterRef {
            mat: mat,
            front: 0,
            back: Col::to_usize(),
        }
    }
}

impl<'a, T: 'a, Row, Col> Iterator for ColsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    type Item = ColsIterRefInner<'a, T, Row, Col>;

    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.back <= Col::to_usize());

        if self.front < self.back {
            let i = self.front;
            self.front += 1;

            Some(ColsIterRefInner::new(self.mat, i))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.front += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for ColsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a, T: 'a, Row, Col> DoubleEndedIterator for ColsIterRef<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        debug_assert!(self.back <= Col::to_usize());

        if self.front < self.back {
            self.back -= 1;
            Some(ColsIterRefInner::new(self.mat, self.back))
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ColsIterRefInner<'a, T: 'a, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    mat: &'a Vector<Vector<T, Col>, Row>,
    col: usize,
    front: usize,
    back: usize,
}

impl<'a, T: 'a, Row, Col> ColsIterRefInner<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    pub fn new(mat: &'a Vector<Vector<T, Col>, Row>, col: usize) -> Self {
        debug_assert!(col < Col::to_usize());
        ColsIterRefInner {
            mat: mat,
            col: col,
            front: 0,
            back: Row::to_usize(),
        }
    }
}

impl<'a, T: 'a, Row, Col> Iterator for ColsIterRefInner<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.front < self.back {
            let i = self.front;
            self.front += 1;
            Some(unsafe {
                self.mat.get_unchecked(i).get_unchecked(self.col)
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.front += n;
        self.next()
    }
}

impl<'a, T: 'a, Row, Col> ExactSizeIterator for ColsIterRefInner<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'a, T: 'a, Row, Col> DoubleEndedIterator for ColsIterRefInner<'a, T, Row, Col>
    where
        Row: ArrayLen<Vector<T, Col>> + 'a,
        Col: ArrayLen<T> + 'a,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            Some(unsafe {
                self.mat.get_unchecked(self.back).get_unchecked(self.col)
            })
        } else {
            None
        }
    }
}

#[test]
fn test_matrix_rows_cols_iter() {
    use typenum::consts::*;
    use super::Matrix;

    let mut m: Matrix<i32, U2, U3> = Default::default();
    m[(0,0)] = 1;
    m[(1,1)] = 2;
    m[(1,2)] = 3;
    m[(0,2)] = 4;
    assert_eq!(m.dim(), (2, 3));
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);

    let mut rows = m.rows_iter();

    assert_eq!(rows.len(), 2);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 3]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut rows = m.rows_iter().rev();

    assert_eq!(rows.len(), 2);
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 3]));
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut rows = m.rows_iter_ref();

    assert_eq!(rows.len(), 2);
    assert!(rows.next().unwrap().iter().eq(&[1, 0, 4]));
    assert!(rows.next().unwrap().iter().eq(&[0, 2, 3]));
    assert_eq!(rows.next(), None);
    assert_eq!(rows.count(), 0);

    let mut cols = m.cols_iter();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[1, 0]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2]));
    assert!(cols.next().unwrap().iter().eq(&[4, 3]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);

    let mut cols = m.cols_iter().rev();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().iter().eq(&[4, 3]));
    assert!(cols.next().unwrap().iter().eq(&[0, 2]));
    assert!(cols.next().unwrap().iter().eq(&[1, 0]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);

    let mut cols = m.cols_iter_ref();

    assert_eq!(cols.len(), 3);
    assert!(cols.next().unwrap().eq(&[1, 0]));
    assert!(cols.next().unwrap().eq(&[0, 2]));
    assert!(cols.next().unwrap().eq(&[4, 3]));
    assert_eq!(cols.next(), None);
    assert_eq!(cols.count(), 0);
}

