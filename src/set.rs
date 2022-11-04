use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;
use std::iter::Chain;

pub trait Set<T> {
    type Iter<'a>: Iterator<Item = &'a T>
    where
        T: 'a,
        Self: 'a;

    fn insert(&mut self, item: T) -> bool;
    fn contains(&self, item: &T) -> bool;
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    fn remove(&mut self, item: &T) -> bool;

    fn len(&self) -> usize {
        let iter = self.iter();
        let (lower, Some(higher)) = iter.size_hint() else { return iter.count() };
        if lower == higher {
            higher
        } else {
            // meh, maybe we should ask for an impl of the len after all?
            iter.count()
        }
    }

    /// Consumes self and other and returns a new set representing the intersection of self and
    /// other.
    fn intersection<S>(self, other: S) -> Intersection<Self, S>
    where
        S: Set<T>,
        Self: Sized,
    {
        Intersection {
            s1: self,
            s2: other,
        }
    }

    /// Returns an iterator of the elements in the intersection of self and other, without
    /// consuming either.
    fn intersection_iter<'a, S>(&'a self, other: &'a S) -> IntersectionIter<'a, T, Self, S>
    where
        S: Set<T>,
        T: 'a,
        Self: Sized,
    {
        if self.len() > other.len() {
            IntersectionIter::Left {
                iter: self.iter(),
                other,
            }
        } else {
            IntersectionIter::Right {
                iter: other.iter(),
                other: self,
            }
        }
    }

    /// consumes the two sets and returns a new set that is the union of the two sets.
    fn union<S>(self, other: S) -> Union<Self, S>
    where
        Self: Sized,
        S: Set<T>,
    {
        Union {
            s1: self,
            s2: other,
        }
    }

    /// returns an iterator to the element in the union of self and other, without consuming
    /// either.
    fn union_iter<'a, S>(&'a self, other: &'a S) -> UnionIter<'a, T, Self, S>
    where
        S: Set<T>,
        Self: Sized,
    {
        UnionIter {
            iter1: self.iter().into(),
            set1: self,
            iter2: other.iter().into(),
        }
    }

    /// Consumes self and other and returns a new Set representing the difference between the two.
    fn difference<S>(self, other: S) -> Difference<Self, S>
    where
        S: Set<T>,
        Self: Sized,
    {
        Difference {
            s1: self,
            s2: other,
        }
    }

    fn difference_iter<'a, S>(&'a self, other: &'a S) -> DifferenceIter<'a, T, Self, S>
    where
        S: Set<T>,
        Self: Sized,
    {
        DifferenceIter {
            iter: self.iter(),
            s2: &other,
        }
    }

    fn symetric_difference<S>(self, other: S) -> SymmetricDifference<Self, S>
    where
        S: Set<T>,
        Self: Sized,
    {
        SymmetricDifference {
            s1: self,
            s2: other,
        }
    }
}

pub struct SymmetricDifference<S1, S2> {
    s1: S1,
    s2: S2,
}

impl<T, S1, S2> Set<T> for SymmetricDifference<S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Iter<'a> = SymmetricDifferenceIter<'a, T, S1, S2>
    where
        T: 'a,
        Self: 'a;

    fn insert(&mut self, item: T) -> bool {
        let s1_contains = self.s1.contains(&item);
        let s2_contains = self.s2.contains(&item);
        // if in the intersection, remove it:
        if s1_contains && s2_contains {
            self.s1.remove(&item);
            false
        } else if s1_contains || s2_contains {
            // it's already in the intersection.
            return true;
        } else {
            // arbitrarely add it to s1
            self.s1.insert(item)
        }
    }

    fn contains(&self, item: &T) -> bool {
        self.s1.contains(item) ^ self.s2.contains(item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        SymmetricDifferenceIter {
            iter: self.s1.iter().chain(self.s2.iter()),
            s1: &self.s1,
            s2: &self.s2,
        }
    }

    fn remove(&mut self, item: &T) -> bool {
        self.s1.remove(item) || self.s2.remove(item)
    }
}

pub struct SymmetricDifferenceIter<'a, T: 'a, S1: Set<T>, S2: Set<T>> {
    iter: Chain<S1::Iter<'a>, S2::Iter<'a>>,
    s1: &'a S1,
    s2: &'a S2,
}

impl<'a, T, S1, S2> Iterator for SymmetricDifferenceIter<'a, T, S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .find(|x| !(self.s1.contains(x) && self.s2.contains(x)))
    }
}

pub struct Difference<S1, S2> {
    s1: S1,
    s2: S2,
}

impl<T, S1, S2> Set<T> for Difference<S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Iter<'a> = DifferenceIter<'a, T, S1, S2>
    where
        T: 'a,
        Self: 'a;

    fn insert(&mut self, item: T) -> bool {
        let s1_contains = self.s1.contains(&item);
        let s2_contains = self.s2.contains(&item);
        if s1_contains && s2_contains {
            self.s1.remove(&item)
        } else if !(s1_contains || s2_contains) {
            // arbitrarely inset in s1
            self.s1.insert(item)
        } else {
            // it's already in the difference
            true
        }
    }

    fn contains(&self, item: &T) -> bool {
        self.s1.contains(item) && !self.s2.contains(item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        DifferenceIter {
            iter: self.s1.iter(),
            s2: &self.s2,
        }
    }

    fn remove(&mut self, item: &T) -> bool {
        self.s1.remove(item) || self.s2.remove(item)
    }
}

pub struct DifferenceIter<'a, T: 'a, S1: Set<T> + 'a, S2: Set<T>> {
    iter: S1::Iter<'a>,
    s2: &'a S2,
}

impl<'a, T, S1, S2> Iterator for DifferenceIter<'a, T, S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|x| !self.s2.contains(x))
    }
}

pub struct Union<S1, S2> {
    s1: S1,
    s2: S2,
}

impl<T, S1, S2> Set<T> for Union<S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Iter<'a> = UnionIter<'a, T, S1, S2>
    where
        T: 'a,
        Self: 'a;

    fn len(&self) -> usize {
        self.s1.len() + self.s2.len() - self.s1.intersection_iter(&self.s2).count()
    }

    fn insert(&mut self, item: T) -> bool {
        // insert arbitrarely in s1
        self.s1.insert(item)
    }

    fn contains(&self, item: &T) -> bool {
        self.s1.contains(item) || self.s2.contains(item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        UnionIter {
            iter1: self.s1.iter().into(),
            set1: &self.s1,
            iter2: self.s2.iter().into(),
        }
    }

    fn remove(&mut self, item: &T) -> bool {
        self.s1.remove(item) || self.s2.remove(item)
    }
}

pub struct UnionIter<'a, T: 'a, S1: Set<T>, S2: Set<T> + 'a> {
    iter1: Option<S1::Iter<'a>>,
    set1: &'a S1,
    iter2: Option<S2::Iter<'a>>,
}

impl<'a, T, S1, S2> Iterator for UnionIter<'a, T, S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter1.as_mut().and_then(|i| i.next()).or_else(|| {
            self.iter2
                .as_mut()
                .and_then(|i| i.find(|x| !self.set1.contains(*x)))
        })
    }
}

pub struct Intersection<S1, S2> {
    s1: S1,
    s2: S2,
}

impl<T, S1, S2> Set<T> for Intersection<S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
    T: Clone,
{
    type Iter<'a> = IntersectionIter<'a, T, S1, S2>
    where
        T: 'a,
        Self: 'a;

    fn len(&self) -> usize {
        self.iter().count()
    }

    fn insert(&mut self, item: T) -> bool {
        let s1_contains = self.s1.insert(item.clone());
        let s2_contains = self.s2.insert(item);
        // only return true if both contained the value before, otherwise, the value could not be
        // present in the intersection.
        s1_contains && s2_contains
    }

    fn contains(&self, item: &T) -> bool {
        self.s1.contains(item) && self.s2.contains(item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        if self.s1.len() < self.s2.len() {
            IntersectionIter::Left {
                iter: self.s1.iter(),
                other: &self.s2,
            }
        } else {
            IntersectionIter::Right {
                iter: self.s2.iter(),
                other: &self.s1,
            }
        }
    }

    fn remove(&mut self, item: &T) -> bool {
        self.s1.remove(item) || self.s2.remove(item)
    }
}

pub enum IntersectionIter<'a, T: 'a, S1: Set<T> + 'a, S2: Set<T> + 'a> {
    Left { iter: S1::Iter<'a>, other: &'a S2 },
    Right { iter: S2::Iter<'a>, other: &'a S1 },
}

impl<'a, T, S1, S2> Iterator for IntersectionIter<'a, T, S1, S2>
where
    S1: Set<T>,
    S2: Set<T>,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntersectionIter::Left { iter, other } => iter.find(|x| other.contains(x)),
            IntersectionIter::Right { iter, other } => iter.find(|x| other.contains(x)),
        }
    }
}

impl<T> Set<T> for HashSet<T>
where
    T: Hash + Eq,
{
    type Iter<'a> = std::collections::hash_set::Iter<'a, T>
    where
        T: 'a,
        Self: 'a;

    fn len(&self) -> usize {
        HashSet::len(self)
    }

    fn insert(&mut self, item: T) -> bool {
        HashSet::insert(self, item)
    }

    fn contains(&self, item: &T) -> bool {
        HashSet::contains(self, item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        HashSet::iter(self)
    }

    fn remove(&mut self, item: &T) -> bool {
        HashSet::remove(self, item)
    }
}

impl<T> Set<T> for BTreeSet<T>
where
    T: Ord + Eq,
{
    type Iter<'a> = std::collections::btree_set::Iter<'a, T>
    where
        T: 'a,
        Self: 'a;

    fn len(&self) -> usize {
        BTreeSet::len(self)
    }

    fn insert(&mut self, item: T) -> bool {
        BTreeSet::insert(self, item)
    }

    fn contains(&self, item: &T) -> bool {
        BTreeSet::contains(self, item)
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        BTreeSet::iter(self)
    }

    fn remove(&mut self, item: &T) -> bool {
        BTreeSet::remove(self, item)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_homogenous_union() {
        let set1 = HashSet::from([1, 2]);
        let set2 = HashSet::from([3, 4]);
        let u = set1.union(set2);
        assert_eq!(
            u.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_heterogenous_union() {
        let set1 = HashSet::from([1, 2]);
        let set2 = BTreeSet::from([3, 4]);
        let u = set1.union(set2);
        assert_eq!(
            u.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_union_len() {
        let set1 = HashSet::from([1, 2]);
        let set2 = BTreeSet::from([3, 4]);
        let u = set1.union(set2);
        assert_eq!(u.len(), 4);
    }
}
