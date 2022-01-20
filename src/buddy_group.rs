use core::alloc::{AllocError, Allocator, Layout};
use core::cmp::Reverse;
use core::{mem, ptr};

use rawpointer::PointerExt;

use super::BuddyAllocator;
use crate::MemoryBlock;

#[derive(Debug, Clone, Copy)]
struct BuddyAllocatorStub {
    storage: MemoryBlock,
}
impl BuddyAllocatorStub {
    fn new(buddy: BuddyAllocator) -> Self {
        Self {
            storage: buddy.storage,
        }
    }

    fn expand(self, min_block: usize) -> BuddyAllocator {
        BuddyAllocator {
            storage: self.storage,
            min_block,
        }
    }
}

fn split_into_pow2_regions<A: Allocator>(
    target: &mut alloc::vec::Vec<MemoryBlock, A>,
    source: MemoryBlock,
    min_size: usize,
) {
    debug_assert!(min_size > 0);

    let buddysize = if source.len().is_power_of_two() {
        source.len()
    } else {
        source.len().next_power_of_two() >> 1
    };

    if buddysize < min_size {
        return;
    }

    // Safety: index is always in-bounds, as per the above calculation
    let (buddy_storage, leftover) = unsafe { source.split_at(buddysize) };
    target.push(buddy_storage);
    split_into_pow2_regions(target, leftover, min_size);
}

/// A group of buddy allocators operating as a single allocator.
/// Has two additional features compared to the regular buddy allocator:
///
/// 1. Can be used for areas that are not power-of-two-sized
/// 2. Can be used noncontiguous memory areas
///
/// To make these possible, a small amount of extra metadata is stored on
/// one of the block. Also, the performance suffers a bit as an additional
/// layer of indirection is created. Also, when allocating, all of the blocks
/// are tried in order from largest to smallest, and the allocator slows
/// down a bit when the first storage blocks are full.
pub struct BuddyGroupAllocator {
    /// Primary allocator used for bookkeeping and such
    primary: BuddyAllocator,
    /// Pointer to bookkeeping of this allocator
    bookkeep_ptr: *const BuddyAllocatorStub,
    /// Suballocator count, excluding the first one
    secondary_count: usize,
}

unsafe impl Send for BuddyGroupAllocator {}

impl BuddyGroupAllocator {
    pub fn new(blocks: &mut [MemoryBlock], min_block: usize) -> Self {
        assert!(!blocks.is_empty(), "Empty input block group");

        // Sort largest input blocks first
        blocks.sort_unstable_by_key(|s| Reverse(s.len()));

        // Filter out blocks that are too small
        let mut i = 0;
        for block in blocks.iter() {
            if block.len() < min_block * 2 {
                break;
            }
            i += 1;
        }
        let blocks = &mut blocks[..i];
        assert!(!blocks.is_empty(), "None of the blocks is large enough");

        // Create the first buddy allocator from the largest block
        // Safety: asserted above
        let (first_block, mut blocks) = unsafe { blocks.split_first_mut().unwrap_unchecked() };
        let buddysize = if first_block.len().is_power_of_two() {
            first_block.len()
        } else {
            first_block.len().next_power_of_two() >> 1
        };
        // Safety: index is always in-bounds, as per the above calculation
        let (buddy0_storage, leftover) = unsafe { first_block.split_at(buddysize) };

        let buddy0 = BuddyAllocator::new(buddy0_storage, min_block);

        // Use the newly created allocator to store temp allocation
        let mut areas = alloc::vec::Vec::new_in(&buddy0);
        split_into_pow2_regions(&mut areas, leftover, min_block * 2);

        // Partition each block into power-of-two-sized blocks
        while let Some((head, tail)) = blocks.split_first_mut() {
            blocks = tail;
            split_into_pow2_regions(&mut areas, *head, min_block * 2);
        }

        // Sort largest allocator blocks first
        areas.sort_unstable_by_key(|s| Reverse(s.len()));

        // Reserve space for the suballocators
        let secondary_count = areas.len();
        let bookkeep_size = mem::size_of::<BuddyAllocatorStub>() * secondary_count;
        // Safety: always fullfills preconditions
        let bookkeep_layout = unsafe { Layout::from_size_align_unchecked(bookkeep_size, 8) };
        let bookkeep = buddy0
            .allocate(bookkeep_layout)
            .expect("Failed to allocate bookkeeping")
            .as_mut_ptr() as *mut BuddyAllocatorStub;

        // Build buddy allocators from the blocks
        for (i, area) in areas.into_iter().enumerate() {
            // Safety: memory is allocated above
            unsafe {
                ptr::write(
                    bookkeep.add(i),
                    BuddyAllocatorStub::new(BuddyAllocator::new(area, min_block)),
                );
            }
        }

        Self {
            primary: buddy0,
            bookkeep_ptr: bookkeep,
            secondary_count,
        }
    }

    fn min_block(&self) -> usize {
        self.primary.min_block()
    }

    /// Calculates free memory. This is an expensive function,
    /// but might be useful for diagnostic purposes and testing.
    pub fn memory_available(&self) -> usize {
        let mut result = self.primary.memory_available();

        for i in 0..self.secondary_count {
            // Safety: the loop keeps within bounds
            result += (unsafe { &*self.bookkeep_ptr.add(i) })
                .expand(self.min_block())
                .memory_available();
        }

        result
    }

    /// Returns None for the primary allocator, and Some(index) for secondaries
    fn resolve_owner(&self, ptr: ptr::NonNull<u8>) -> Option<usize> {
        // Safety: invariants upheld by BuddyAllocator
        let primary_end = unsafe { self.primary.storage.nn().add(self.primary.size()) };
        if self.primary.storage.nn() <= ptr && ptr <= primary_end {
            return None;
        }

        for i in 0..self.secondary_count {
            // Safety: the loop keeps within bounds
            let secondary = (unsafe { &*self.bookkeep_ptr.add(i) }).expand(self.min_block());
            // Safety: invariants upheld by BuddyAllocator
            let secondary_end = unsafe { secondary.storage.nn().add(secondary.size()) };
            if secondary.storage.nn() <= ptr && ptr <= secondary_end {
                return Some(i);
            }
        }

        panic!("Given pointer is not owned by this allocator");
    }
}
unsafe impl Allocator for BuddyGroupAllocator {
    fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        // Attempt allocation with (faster) the primary allocator
        if let Ok(ok) = self.primary.allocate(layout) {
            return Ok(ok);
        }

        // Walk secondary allocators in order and attempt to allocate
        for i in 0..self.secondary_count {
            // Safety: the loop keeps within bounds
            if let Ok(ok) = (unsafe { &*self.bookkeep_ptr.add(i) })
                .expand(self.min_block())
                .allocate(layout)
            {
                return Ok(ok);
            }
        }

        Err(AllocError) // All allocators are full
    }

    unsafe fn deallocate(&self, ptr: ptr::NonNull<u8>, layout: Layout) {
        if let Some(i) = self.resolve_owner(ptr) {
            // Safety: the index is within bounds
            { &*self.bookkeep_ptr.add(i) }
                .expand(self.min_block())
                .deallocate(ptr, layout);
        } else {
            self.primary.deallocate(ptr, layout)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::alloc::{Allocator, Layout};
    use core::ptr;

    use super::*;

    #[test]
    fn simple() {
        let backing1 = MemoryBlock::test_new(1024);
        let backing2 = MemoryBlock::test_new(500);
        let backing3 = MemoryBlock::test_new(200);
        let allocator = BuddyGroupAllocator::new(&mut [backing1, backing2, backing3], 64);

        assert!(allocator.memory_available() > 1200);

        let b0 = allocator
            .allocate(Layout::from_size_align(128, 1).unwrap())
            .expect("alloc");

        assert_eq!(b0.len(), 128);

        // Check that we do not get a segfault or anything
        unsafe {
            ptr::write(b0.as_mut_ptr(), 1);
            assert_eq!(ptr::read(b0.as_mut_ptr()), 1);
        }

        unsafe {
            allocator.deallocate(
                b0.as_non_null_ptr(),
                Layout::from_size_align(b0.len(), 1).unwrap(),
            );
        }

        backing1.test_destroy();
        backing2.test_destroy();
        backing3.test_destroy();
    }

    #[test]
    fn multiple() {
        let backing1 = MemoryBlock::test_new(256);
        let backing2 = MemoryBlock::test_new(128);
        let backing3 = MemoryBlock::test_new(128);
        let allocator = BuddyGroupAllocator::new(&mut [backing1, backing2, backing3], 64);

        dbg!(allocator.memory_available());

        let blocks: Vec<_> = (0..4)
            .map(|_| {
                allocator
                    .allocate(Layout::from_size_align(64, 1).unwrap())
                    .expect("alloc")
            })
            .collect();

        allocator
            .allocate(Layout::from_size_align(64, 1).unwrap())
            .expect_err("alloc should have failed");

        for block in &blocks {
            // Check that we do not get a segfault or anything
            unsafe {
                ptr::write(block.as_mut_ptr(), 1);
                assert_eq!(ptr::read(block.as_mut_ptr()), 1);
            }
        }

        for block in blocks {
            unsafe {
                allocator.deallocate(
                    block.as_non_null_ptr(),
                    Layout::from_size_align(block.len(), 1).unwrap(),
                );
            }
        }

        dbg!(allocator.memory_available());

        backing1.test_destroy();
        backing2.test_destroy();
        backing3.test_destroy();
    }
}
