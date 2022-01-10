//! See http://bitsquid.blogspot.com/2015/08/allocation-adventures-3-buddy-allocator.html

use core::alloc::{AllocError, Allocator, Layout};
use core::{mem, ptr};

use rawpointer::PointerExt;
use static_assertions as sa;

use crate::util::div_ceil;

type OptPtr<T> = Option<ptr::NonNull<T>>;
sa::assert_eq_size!(OptPtr<u8>, *mut u8);
sa::assert_eq_size!(OptPtr<*mut u8>, *mut u8);

#[cfg(feature = "extra-checks")]
const HEADER_MAGIC: [u8; 8] = *b"LLHeader";

/// A linked list header
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LLHeader {
    #[cfg(feature = "extra-checks")]
    magic: [u8; 8],
    prev: OptPtr<LLHeader>,
    next: OptPtr<LLHeader>,
}
impl LLHeader {
    const EMPTY: Self = LLHeader {
        #[cfg(feature = "extra-checks")]
        magic: HEADER_MAGIC,
        prev: None,
        next: None,
    };
}

/// Given a block of memory, and a minimal block size, subdivides
/// the memory region into a binary tree of power-of-two-bytes sized
/// blocks.
///
/// This is not a good choice for `#[global_allocator]`, but is excellent
/// option for allocating physical page frames in a kernel.
///
/// The allocator has minimal overhead: 0.5 bits per `min_block` plus a pointer
/// per level, meaning that the overhead with 4GiB of memory are and 4KiB `min_block`
/// is about 0.0015% (`(((4<<30)//(4<<10))/2/8 + 20*8) / (4<<30) * 100`).
///
/// The operations are also fast. Worst-case complexity both allocation and
/// deallocation is `O(log d)` (where `d = size/min_block`). Note that this is
/// constant time over allocation size, giving caller a concrete upper bound
/// on performed operations.
///
/// /// This is a base-level allocator that manges it's bookkeeping memory.
///
/// ## Example
///
/// ```txt
///           |-------------------------------------------------------|
/// Top       |  Toplevel division, `size` bytes of memory            |
///           |---------------------------|---------------------------|
/// Upper     | 0                         | 1                         |
///           |-------------|-------------|-------------|-------------|
/// Middle    | 0.0         | 0.1         | 1.1         | 1.2         |
///           |------|------|------|------|             |------|------|
/// Lower     | 0.0.0| 0.0.1| 0.1.0| 0.1.1|      |      | 1.2.0| 1.2.1|
///           |-|----|----|-|--|---|--|---|------|------|---|--|--|---|
///             |         |   free   free        |        used   free
///        bookkeeping    |                  used (C)      (D)
///        frame (A)    used (B)
/// ```
///
/// Simple binary tree encoding for this tree is:
///    AB    C D
/// `001100001010`
///
/// For a bigger tree, say 16 GiB with 4KiB blocks, the tree size is
/// `(16<<30) / (4<<10) = 0x400000` bits = `0x80000` bytes, which takes
/// 128 * 4KiB frames (512MiB) by itself, same as storing a bitmap for
/// each frame. However, the whole are isn't needed unless the space has
/// maximum internal fragmentation. On the other hand, storing a traditional
/// pointer-based binary tree only needs the memory when nodes are allocated,
/// but would consume even more memory. Pointer-chasing is also inefficient.
///
/// Instead of trees, we use the first couple of bytes on free blocks to
/// store pointers to previous and next free block. This way, the free blocks
/// from a doubly-linked list.
///
pub struct BuddyAllocator {
    /// Pointer to the beginning of an area
    base: ptr::NonNull<u8>,
    /// Size of the area, in bytes
    /// This must be a power of two
    /// Max allocation is half of this
    size: usize,
    /// Size of the area, in bytes
    /// This must be a power of two and >= sizeof(BookkeepHeader).
    min_block: usize,
}
impl BuddyAllocator {
    pub fn new(storage: &mut [u8], min_block: usize) -> Self {
        let size = storage.len();

        debug_assert!(size >= 2);
        debug_assert!(size.is_power_of_two());
        debug_assert!(min_block.is_power_of_two());
        debug_assert!(min_block <= size / 2);
        debug_assert!(min_block >= mem::size_of::<LLHeader>());

        let base = unsafe { ptr::NonNull::new_unchecked(storage.as_mut_ptr()) };
        let mut result = Self {
            base,
            size,
            min_block,
        };

        let bk_size = result.bookkeeping_size();
        let ffb_level = result.block_level(bk_size);
        debug_assert!(
            ffb_level != 0,
            "Bookkeeping info requires the whole structure"
        );

        // Zero put the bookkeeping area
        // Safety: the pointer acquired from a slice is valid
        unsafe {
            ptr::write_bytes(result.base.as_mut(), 0, bk_size);
        }

        // Split all tops from top until this one, mark right side free
        for level in 1..(ffb_level + 1) {
            let offset = result.block_size_on_level(level);
            unsafe {
                let mut free_area = result.base.add(offset).cast::<_>();
                *free_area.as_mut() = LLHeader::EMPTY;
                result.set_free_block_ptr_for(level, Some(free_area));
                debug_assert_eq!(result.free_block_ptr_for(level), Some(free_area));
            }

            result.flip_buddy_bit(level, 0);
        }

        // Levels below this have no free blocks yet
        for level in (ffb_level + 1)..result.levels() {
            result.set_free_block_ptr_for(level, None);
        }

        result
    }

    /// size == (1<<num_levels) * min_block
    /// => log2(size/min_block) == num_levels
    fn levels(&self) -> usize {
        ((self.size / self.min_block).trailing_zeros() + 1) as usize
    }

    /// Level of a block
    fn block_level(&self, block_size: usize) -> usize {
        debug_assert!(block_size.is_power_of_two());
        debug_assert!(block_size >= self.min_block);
        debug_assert!(block_size <= self.size);
        (self.size / block_size).trailing_zeros() as usize
    }

    /// Level of a block
    fn block_size_on_level(&self, level: usize) -> usize {
        assert!(level < self.levels());
        self.size / 2usize.pow(level as u32)
    }

    /// Level of a block
    fn blocks_on_level(&self, level: usize) -> usize {
        assert!(level != 0);
        2usize.pow((level - 1) as u32)
    }

    /// Level of a block
    fn block_index_on_level(&self, level: usize, block: ptr::NonNull<u8>) -> usize {
        let a = self.base.as_ptr() as usize;
        let b = block.as_ptr() as usize;
        let offset = b - a;
        offset / self.block_size_on_level(level)
    }

    fn tree_bitmap_size_bytes(&self) -> usize {
        div_ceil(1 << (self.levels() - 1), 8)
    }

    /// Bookkeeping data consists of two items;
    /// * an array of level pointers for levels 1..=total
    /// * a bitmap tree with a bit (a XOR b) for every pair of nodes
    fn bookkeeping_size(&self) -> usize {
        // Level 0 doesn't need a pointer, as it never has free blocks
        let list_ptr_arr_size = (self.levels() - 1) * mem::size_of::<*mut u8>();
        (list_ptr_arr_size + self.tree_bitmap_size_bytes())
            .next_power_of_two()
            .max(self.min_block)
    }

    fn free_block_ptr_for(&self, level: usize) -> OptPtr<LLHeader> {
        debug_assert!(
            level != 0,
            "Level 0 has no free blocks, and no freelist ptr"
        );
        unsafe { *self.base.cast::<OptPtr<LLHeader>>().add(level - 1).as_ptr() }
    }

    fn set_free_block_ptr_for(&self, level: usize, v: OptPtr<LLHeader>) {
        debug_assert!(
            level != 0,
            "Level 0 never has free blocks, and no freelist ptr either"
        );
        unsafe {
            *self.base.cast::<OptPtr<LLHeader>>().add(level - 1).as_mut() = v;
        }
    }

    fn buddy_bitmap_start(&self) -> ptr::NonNull<u8> {
        unsafe {
            self.base
                .cast::<OptPtr<LLHeader>>()
                .add(self.levels() - 1) // -1 ??
                .cast()
        }
    }

    fn buddy_bit(&self, level: usize, bit_index: usize) -> bool {
        assert!(level != 0, "Level 0 has no buddy bitmap");
        debug_assert!(bit_index < self.blocks_on_level(level));

        let bitmap = self.buddy_bitmap_start();
        let level_start_bit = (1 << (level - 1)) - 1;
        let i = level_start_bit + bit_index;
        let byte = i / 8;
        let bit = i % 8;
        debug_assert!(byte < self.size); // Sanity check
        unsafe { (*bitmap.add(byte).as_mut()) & (1 << bit) != 0 }
    }

    /// Returns the old value, i.e. true if either both or none are in use after this.
    /// This means that for `deallocate`, true means that the buddy was empty,
    /// and the empty blocks can now be merged.
    fn flip_buddy_bit(&self, level: usize, bit_index: usize) -> bool {
        assert!(level != 0, "Level 0 has no buddy bitmap");
        debug_assert!(bit_index < self.blocks_on_level(level));
        let bitmap = self.buddy_bitmap_start();
        let level_start_bit = (1 << (level - 1)) - 1;
        let i = level_start_bit + bit_index;
        let byte = i / 8;
        let bit = i % 8;
        let bit_mask = 1 << bit;
        debug_assert!(byte < self.size); // Sanity check
        unsafe {
            let old_byte = *bitmap.add(byte).as_mut();
            *bitmap.add(byte).as_mut() = old_byte ^ bit_mask;
            old_byte & bit_mask != 0
        }
    }

    fn buddy_bit_for_block(&self, level: usize, block: ptr::NonNull<u8>) -> bool {
        let block_index = self.block_index_on_level(level, block);
        self.buddy_bit(level, block_index / 2)
    }

    /// Returns the old value, i.e. true if either both or none are in use after this.
    /// This means that for `deallocate`, true means that the buddy was empty,
    /// and the empty blocks can now be merged.
    fn flip_buddy_bit_for_block(&self, level: usize, block: ptr::NonNull<u8>) -> bool {
        let block_index = self.block_index_on_level(level, block);
        self.flip_buddy_bit(level, block_index / 2)
    }

    /// Calculates free memory. This is an expensive function,
    /// but might be useful for diagnostic purposes and testing.
    pub fn memory_available(&self) -> usize {
        let mut result = 0;

        for level in 1..self.levels() {
            if let Some(fbp) = self.free_block_ptr_for(level) {
                result += self.block_size_on_level(level);

                // Before
                let mut cursor = fbp;
                while let Some(p) = unsafe { cursor.as_ref().prev } {
                    #[cfg(feature = "extra-checks")]
                    assert_eq!(
                        unsafe { cursor.as_mut() }.magic,
                        HEADER_MAGIC,
                        "Invalid header"
                    );
                    cursor = p;
                    result += self.block_size_on_level(level);
                }

                // After
                let mut cursor = fbp;
                while let Some(p) = unsafe { cursor.as_ref().next } {
                    #[cfg(feature = "extra-checks")]
                    assert_eq!(
                        unsafe { cursor.as_mut() }.magic,
                        HEADER_MAGIC,
                        "Invalid header"
                    );
                    cursor = p;
                    result += self.block_size_on_level(level);
                }
            }
        }

        result
    }

    #[cfg(test)]
    pub fn dump_freelists(&self) {
        for level in 1..self.levels() {
            if let Some(fbp) = self.free_block_ptr_for(level) {
                println!("?? {:?}", fbp);

                let mut c = 0;

                // Before
                let mut cursor = fbp;
                while let Some(p) = unsafe { cursor.as_ref().prev } {
                    cursor = p;
                    c += 1;
                }

                // After
                let mut cursor = fbp;
                while let Some(p) = unsafe { cursor.as_ref().next } {
                    cursor = p;
                    c += 1;
                }

                println!(
                    "Level {} ({:>4}): {:?}",
                    level,
                    self.block_size_on_level(level),
                    c
                );
            } else {
                println!(
                    "Level {} ({:>4}): (empty)",
                    level,
                    self.block_size_on_level(level)
                );
            }
        }
    }

    #[cfg(test)]
    pub fn tree_snapshot(&self) -> Vec<u8> {
        let sz = self.tree_bitmap_size_bytes();
        let mut result = vec![0; sz];
        unsafe {
            ptr::copy_nonoverlapping(self.buddy_bitmap_start().as_ptr(), result.as_mut_ptr(), sz);
        }
        result
    }

    unsafe fn consume_free_block(&self, level: usize, mut block: ptr::NonNull<LLHeader>) {
        #[cfg(feature = "extra-checks")]
        assert_eq!(block.as_mut().magic, HEADER_MAGIC, "Invalid header");
        let prev = block.as_mut().prev.take();
        let next = block.as_mut().next.take();

        if let Some(mut p) = prev {
            (*p.as_mut()).next = next;
            if let Some(mut n) = next {
                (*n.as_mut()).prev = Some(p);
            }
            self.set_free_block_ptr_for(level, Some(p));
        } else if let Some(mut n) = next {
            (*n.as_mut()).prev = None;
            self.set_free_block_ptr_for(level, Some(n));
        } else {
            // This is the last free block on this level
            self.set_free_block_ptr_for(level, None);
        }
    }

    /// Recursive helper for handling allocation
    fn allocate_on_level(&self, level: usize) -> Result<ptr::NonNull<u8>, AllocError> {
        if level == 0 {
            // Out of memory; top-level block cannot be free as bookkeeping uses part of it
            Err(AllocError)
        } else if let Some(block) = self.free_block_ptr_for(level) {
            // This level has a free block, consume it
            unsafe {
                self.consume_free_block(level, block);
            }
            self.flip_buddy_bit_for_block(level, block.cast());
            Ok(block.cast())
        } else {
            // This level has no blocks left, recursively allocate a block above this
            let block = self.allocate_on_level(level - 1)?;

            // Split the newly allocated block, and
            // set up a the buddy as the only free block
            let size = self.block_size_on_level(level);
            unsafe {
                let mut buddy: ptr::NonNull<LLHeader> = block.cast::<u8>().add(size).cast();
                *buddy.as_mut() = LLHeader::EMPTY;
                self.set_free_block_ptr_for(level, Some(buddy));
            }

            self.flip_buddy_bit_for_block(level, block);

            Ok(block)
        }
    }

    /// Recursive helper for handling deallocation
    fn deallocate_on_level(&self, target: ptr::NonNull<u8>, level: usize) {
        if level == 0 {
            // We still have some bookkeeping info at least, not possible
            panic!("Level 0 blocks cannot be deallocated externally");
        }

        let mut block: ptr::NonNull<LLHeader> = target.cast();
        let size = self.block_size_on_level(level);
        debug_assert!(size >= self.min_block);

        if let Some(mut freelist_head) = self.free_block_ptr_for(level) {
            // This level has other free entries as well. We could simply
            // traverse the linked list of free blocks to obtain it's location,
            // but instead we are using a buddy XOR bitmap.

            let buddy_is_empty = self.flip_buddy_bit_for_block(level, block.cast());
            if buddy_is_empty {
                // Merge adjacent empty blocks
                let buddy: ptr::NonNull<u8> = if self.block_index_on_level(level, target) % 2 == 0 {
                    // We are left, buddy is right
                    unsafe { target.add(size).cast() }
                } else {
                    // Buddy is left, we are right
                    unsafe { target.sub(size).cast() }
                };

                unsafe { self.consume_free_block(level, buddy.cast()) };

                self.deallocate_on_level(target.min(buddy), level - 1);
            } else {
                // Insert self to list of free blocks on this layer
                unsafe {
                    let next = mem::replace(&mut freelist_head.as_mut().next, Some(block));
                    *block.as_mut() = LLHeader {
                        #[cfg(feature = "extra-checks")]
                        magic: HEADER_MAGIC,
                        prev: Some(freelist_head),
                        next,
                    }
                }
            }
        } else {
            // No (other) free entries on this level
            unsafe {
                *block.as_mut() = LLHeader::EMPTY;
            }
            self.set_free_block_ptr_for(level, Some(block));

            debug_assert!(!self.buddy_bit_for_block(level, target));
            self.flip_buddy_bit_for_block(level, target);
        }
    }
}
unsafe impl Allocator for BuddyAllocator {
    fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        let req_size = if layout.size() < self.min_block {
            self.min_block
        } else {
            layout.size().next_power_of_two()
        };
        assert!(
            req_size <= self.size / 2,
            "Trying to allocate too large block"
        );

        let level = self.block_level(req_size);
        let block = self.allocate_on_level(level)?;
        unsafe {
            debug_assert!(
                block.add(req_size) <= self.base.add(self.size),
                "Attempting to return a block that's out-of-bounds"
            );
        }
        Ok(ptr::NonNull::slice_from_raw_parts(block, req_size))
    }

    unsafe fn deallocate(&self, ptr: ptr::NonNull<u8>, layout: Layout) {
        let req_size = if layout.size() < self.min_block {
            self.min_block
        } else {
            layout.size().next_power_of_two()
        };

        assert!(
            ptr.add(req_size) <= self.base.add(self.size),
            "Deallocation outside memory area"
        );

        debug_assert!(req_size <= self.size);
        let level = self.block_level(req_size);
        self.deallocate_on_level(ptr, level);
    }
}

#[cfg(test)]
mod tests_layered {
    use core::alloc::{AllocError, Allocator, Layout};
    use core::ptr;

    use super::*;

    #[test]
    fn internal_calculations() {
        let mut backing = [0; 1024];
        let allocator = BuddyAllocator::new(&mut backing, 128);

        assert_eq!(allocator.levels(), 4);

        assert_eq!(allocator.block_size_on_level(0), 1024);
        assert_eq!(allocator.block_size_on_level(1), 512);
        assert_eq!(allocator.block_size_on_level(2), 256);
        assert_eq!(allocator.block_size_on_level(3), 128);

        assert_eq!(allocator.block_level(1024), 0);
        assert_eq!(allocator.block_level(512), 1);
        assert_eq!(allocator.block_level(256), 2);
        assert_eq!(allocator.block_level(128), 3);

        #[cfg(not(feature = "extra-checks"))]
        {
            let mut backing = [0; 64];
            let allocator = BuddyAllocator::new(&mut backing, 16);

            assert_eq!(allocator.levels(), 3);

            assert_eq!(allocator.block_size_on_level(0), 64);
            assert_eq!(allocator.block_size_on_level(1), 32);
            assert_eq!(allocator.block_size_on_level(2), 16);

            assert_eq!(allocator.block_level(64), 0);
            assert_eq!(allocator.block_level(32), 1);
            assert_eq!(allocator.block_level(16), 2);
        }

        let mut backing = [0; 64];
        let allocator = BuddyAllocator::new(&mut backing, 32);

        assert_eq!(allocator.levels(), 2);

        assert_eq!(allocator.block_size_on_level(0), 64);
        assert_eq!(allocator.block_size_on_level(1), 32);

        assert_eq!(allocator.block_level(64), 0);
        assert_eq!(allocator.block_level(32), 1);

        let mut backing = [0; 2048];
        let allocator = BuddyAllocator::new(&mut backing, 32);

        assert_eq!(allocator.levels(), 7);

        assert_eq!(allocator.block_size_on_level(0), 2048);
        assert_eq!(allocator.block_size_on_level(6), 32);

        assert_eq!(allocator.block_level(2048), 0);
        assert_eq!(allocator.block_level(32), 6);

        let mut backing = vec![0; 1 << 20];
        let allocator = BuddyAllocator::new(&mut backing, 4 << 10);

        assert_eq!(allocator.levels(), 9);

        assert_eq!(allocator.block_size_on_level(0), 1 << 20);
        assert_eq!(allocator.block_size_on_level(8), 4 << 10);

        assert_eq!(allocator.block_level(1 << 20), 0);
        assert_eq!(allocator.block_level(4 << 10), 8);
    }

    #[test]
    fn internal_buddy_bittree() {
        let mut backing = vec![0; 64];
        // Bypass initialization, create a truly empty state
        let allocator = BuddyAllocator {
            base: unsafe { ptr::NonNull::new_unchecked(backing.as_mut_ptr()) },
            size: backing.len(),
            min_block: 16,
        };

        let a = allocator.tree_snapshot();
        allocator.flip_buddy_bit(1, 0);
        allocator.flip_buddy_bit(1, 0);
        assert_eq!(a, allocator.tree_snapshot());
        allocator.flip_buddy_bit(1, 0);
        assert_ne!(a, allocator.tree_snapshot());
        allocator.flip_buddy_bit(2, 0);
        allocator.flip_buddy_bit(3, 0);
        allocator.flip_buddy_bit(1, 0);
        allocator.flip_buddy_bit(2, 0);
        allocator.flip_buddy_bit(3, 0);
        assert_eq!(a, allocator.tree_snapshot());
    }

    #[test]
    fn simple() {
        let mut backing = vec![0; 1024];
        let allocator = BuddyAllocator::new(&mut backing, 64);

        allocator.dump_freelists();
        assert_eq!(allocator.memory_available(), 1024 - 64);

        let b0 = allocator
            .allocate(Layout::from_size_align(128, 1).unwrap())
            .expect("alloc");

        assert_eq!(b0.len(), 128);

        // Check that we do not get a segfault or anything
        unsafe {
            ptr::write(b0.as_mut_ptr(), 1);
            assert_eq!(ptr::read(b0.as_mut_ptr()), 1);
        }

        // Check that we can allocate the remaining larger blocks
        let b1 = allocator
            .allocate(Layout::from_size_align(256, 1).unwrap())
            .expect("alloc");

        assert_eq!(b1.len(), 256);
        assert!(b0.as_ptr() != b1.as_ptr());

        allocator.dump_freelists();
        let b2 = allocator
            .allocate(Layout::from_size_align(512, 1).unwrap())
            .expect("alloc");

        assert_eq!(b2.len(), 512);
        assert!(b0.as_ptr() != b2.as_ptr());
        assert!(b1.as_ptr() != b2.as_ptr());

        // Now we should be out of 256 blocks
        allocator.dump_freelists();
        assert_eq!(
            allocator.allocate(Layout::from_size_align(512, 1).unwrap()),
            Err(AllocError)
        );

        // But there should still be a single 64 block left
        let b3 = allocator
            .allocate(Layout::from_size_align(64, 1).unwrap())
            .expect("alloc");

        assert_eq!(b3.len(), 64);
        assert!(b0.as_ptr() != b3.as_ptr());
        assert!(b1.as_ptr() != b3.as_ptr());
        assert!(b2.as_ptr() != b3.as_ptr());

        assert_eq!(allocator.memory_available(), 0);

        // Deallocate all blocks
        for b in [b0, b1, b2, b3] {
            unsafe {
                allocator.deallocate(
                    b.as_non_null_ptr(),
                    Layout::from_size_align(b.len(), 1).unwrap(),
                );
            }
        }

        assert_eq!(allocator.memory_available(), 1024 - 64);
    }

    #[test]
    fn merges_buddies() {
        let mut backing = [0; 128];
        let allocator = BuddyAllocator::new(&mut backing, 32);

        let b0 = allocator
            .allocate(Layout::from_size_align(32, 1).unwrap())
            .expect("alloc");

        let b1 = allocator
            .allocate(Layout::from_size_align(32, 1).unwrap())
            .expect("alloc");

        // Now the subdivisions should look like this:
        //          64               64
        // |-----------------|-----------------|
        // |      Split      |      Split      |
        // |--------|--------|--------|--------|
        // |Bookkeep|  Free  |   b0   |   b1   |
        // |--------|--------|--------|--------|
        //     32       32       32       32

        assert_eq!(allocator.memory_available(), 32);

        // Free both b0, b1 and make sure their parent region becomes available

        unsafe {
            allocator.deallocate(
                b0.as_non_null_ptr(),
                Layout::from_size_align(b0.len(), 1).unwrap(),
            );
            allocator.deallocate(
                b1.as_non_null_ptr(),
                Layout::from_size_align(b1.len(), 1).unwrap(),
            );
        }
    }

    #[test]
    fn merges_propagate() {
        let mut backing = [0; 256];
        let allocator = BuddyAllocator::new(&mut backing, 32);

        // Now the subdivisions should look like this:
        //         128               128
        // |-----------------|-----------------|
        // |      Split      |      Free       | 128
        // |--------|--------|-----------------|
        // | Split  |  Free  |                 |  64
        // |----+---|        |                 |
        // |Book| F |        |                 |  32
        // |----|---|--------|--------|--------|
        //     32       32       32       32

        // Consume left 128 completely
        let _c0 = allocator
            .allocate(Layout::from_size_align(32, 1).unwrap())
            .expect("alloc");
        let _c1 = allocator
            .allocate(Layout::from_size_align(64, 1).unwrap())
            .expect("alloc");

        assert_eq!(allocator.memory_available(), 256 / 2);

        // Reserve bottom-level slots on the right
        let right = [
            allocator
                .allocate(Layout::from_size_align(32, 1).unwrap())
                .expect("alloc"),
            allocator
                .allocate(Layout::from_size_align(32, 1).unwrap())
                .expect("alloc"),
            allocator
                .allocate(Layout::from_size_align(32, 1).unwrap())
                .expect("alloc"),
            allocator
                .allocate(Layout::from_size_align(32, 1).unwrap())
                .expect("alloc"),
        ];

        // Now the subdivisions should look like this:
        //         128               128
        // |-----------------|-----------------|
        // |      Split      |      Split      | 128
        // |--------|--------|--------|--------|
        // | Split  |  Used  |  Split | Split  |  64
        // |----+---|  (c1)  |--------|--------|
        // |Book|c0 |        | b0 |b1 | b2 |b3 |  32
        // |----|---|--------|--------|--------|
        //     32       32       32       32

        assert_eq!(allocator.memory_available(), 0);

        for r in right {
            unsafe {
                allocator.deallocate(
                    r.as_non_null_ptr(),
                    Layout::from_size_align(r.len(), 1).unwrap(),
                );
            }
        }

        // Ensure that the right side deallocation has propagated

        assert_eq!(allocator.memory_available(), 256 / 2);

        let _ = allocator
            .allocate(Layout::from_size_align(128, 1).unwrap())
            .expect("alloc");
    }

    #[test]
    fn pathological_fragmenter() {
        let mut backing = [0; 1024];
        let allocator = BuddyAllocator::new(&mut backing, 64);

        // Allocate all min-size blocks
        let mut blocks = Vec::new();
        for _ in 0..(1024 / 64 - 1) {
            println!("Pathological fl {}", allocator.memory_available());
            allocator.dump_freelists();
            println!("Pathological alloc");
            blocks.push(
                allocator
                    .allocate(Layout::from_size_align(64, 1).unwrap())
                    .expect("alloc"),
            );
        }

        assert_eq!(allocator.memory_available(), 0);

        // Check that we do not get a segfault or anything
        for block in &blocks {
            unsafe {
                ptr::write(block.as_mut_ptr(), 1);
                assert_eq!(ptr::read(block.as_mut_ptr()), 1);
            }
        }

        // Free every other block
        for block in blocks.iter().step_by(2) {
            unsafe {
                allocator.deallocate(
                    block.as_non_null_ptr(),
                    Layout::from_size_align(block.len(), 1).unwrap(),
                );
            }
        }

        // Now allocating a bigger block should fail
        assert_eq!(
            allocator.allocate(Layout::from_size_align(128, 1).unwrap()),
            Err(AllocError)
        );
    }

    #[test]
    fn on_nonzeroed_memory() {
        let mut backing = [0xdd; 1024];
        let allocator = BuddyAllocator::new(&mut backing, 64);

        let a = allocator
            .allocate(Layout::from_size_align(64, 1).unwrap())
            .expect("alloc");

        let b = allocator
            .allocate(Layout::from_size_align(256, 1).unwrap())
            .expect("alloc");

        unsafe {
            // Going deep into the implementation details here.
            // This check can be removed later if data in blocks is overwritten
            // by something more that the freelist pointers
            assert_eq!(ptr::read(a.as_mut_ptr().add(0x20)), 0xdd);

            allocator.deallocate(
                a.as_non_null_ptr(),
                Layout::from_size_align(a.len(), 1).unwrap(),
            );
            allocator.deallocate(
                b.as_non_null_ptr(),
                Layout::from_size_align(b.len(), 1).unwrap(),
            );
        }

        assert_eq!(allocator.memory_available(), 1024 - 64);
    }

    #[test]
    fn almost_fuzz() {
        // for total_size in [64, 256, 1024] {
        //     for min_block in [16, 32] {
        for total_size in [256, 1024] {
            for min_block in [32, 64, 128] {
                println!(
                    "Configuration: total_size={}, min_block={}",
                    total_size, min_block
                );
                let mut backing = vec![0x12; total_size];
                let allocator = BuddyAllocator::new(&mut backing, min_block);

                let state = allocator.tree_snapshot();

                for i in 0..3 {
                    println!("########################## Round {} {:?}", i, state);
                    allocator.dump_freelists();

                    assert_eq!(
                        state,
                        allocator.tree_snapshot(),
                        "Alloc-dealloc changed internal state"
                    );

                    let b = allocator
                        .allocate(Layout::from_size_align(min_block, 1).unwrap())
                        .expect("alloc");

                    println!(
                        "dealloc {:?}\n        {:?}",
                        state,
                        allocator.tree_snapshot()
                    );
                    allocator.dump_freelists();

                    unsafe {
                        allocator.deallocate(
                            b.as_non_null_ptr(),
                            Layout::from_size_align(b.len(), 1).unwrap(),
                        );
                    }
                }
            }
        }
    }
}
