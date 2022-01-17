use core::alloc::AllocError;
use core::{mem, ptr};

/// An allocator that can only handle items of a single fixed size, storing
/// the bookkeeping into a linked list of free nodes. Has minimal overhead
/// (only a single item) and does both allocation and deallocation in O(1).
/// Smallest allocation unit has same size as a pointer.
///
/// `free_ptr` points to a singly-linked list of free nodes.
/// Each node has a pointer to the next free node. Null means end of list.
pub struct BlockLLAllocator {
    base: ptr::NonNull<u8>,
    size: usize,
    item_size: usize,
}

unsafe impl Send for BlockLLAllocator {}

impl BlockLLAllocator {
    pub fn new(block: &mut [u8], item_size: usize) -> Self {
        assert!(!block.is_empty(), "Empty input block");

        assert!(
            block.len() % item_size == 0,
            "backing block is not a multiple of item size"
        );

        assert!(
            block.len() >= item_size * 2,
            "backing block too small, cannot fit any data"
        );

        assert!(
            item_size >= mem::size_of::<*mut u8>(),
            "item_size is smaller than a pointer"
        );

        let result = Self {
            base: unsafe { ptr::NonNull::new_unchecked(block.as_mut_ptr()) },
            size: block.len(),
            item_size,
        };

        // Go through items and initialize the free-ll and the first item as free_ptr
        let mut prev: *mut u8 = ptr::null_mut();
        for i in (0..(result.capacity() + 1)).rev() {
            unsafe {
                let ptr: *mut *mut u8 = block.as_mut_ptr().add(item_size * i).cast();
                *ptr = prev;
                prev = ptr as *mut u8;
            }
        }

        result
    }

    pub fn item_size(&self) -> usize {
        self.item_size
    }

    /// How many items this allocator can store
    pub fn capacity(&self) -> usize {
        (self.size / self.item_size) - 1
    }

    /// Calculates free memory, in bytes. This is an expensive function,
    /// but might be useful for diagnostic purposes and testing.
    pub fn memory_available(&self) -> usize {
        self.slots_available() * self.item_size
    }

    /// Calculates free memory, in items. This is an expensive function,
    /// but might be useful for diagnostic purposes and testing.
    pub fn slots_available(&self) -> usize {
        let mut result = 0;
        let mut cursor = self.next_free();
        while let Some(entry) = cursor {
            let next = unsafe { *entry.as_ptr() };
            cursor = next.map(|p| p.cast());
            result += 1;
        }
        result
    }

    pub fn is_full(&self) -> bool {
        self.next_free().is_none()
    }

    fn next_free(&self) -> Option<ptr::NonNull<Option<ptr::NonNull<u8>>>> {
        let field = self.base.cast();
        unsafe { *field.as_ptr() }
    }

    fn set_next_free(&self, next_free: Option<ptr::NonNull<Option<ptr::NonNull<u8>>>>) {
        let field = self.base.cast();
        unsafe {
            *field.as_ptr() = next_free;
        }
    }

    pub fn allocate_one(&self) -> Result<ptr::NonNull<u8>, AllocError> {
        // Obtain a free entry, if any are available
        let Some(free_ptr) = self.next_free() else {
            // Out of memory
            return Err(AllocError);
        };

        // Update bookkeeping
        let next_free = unsafe { *free_ptr.as_ptr() };
        self.set_next_free(next_free.map(|p| p.cast()));

        // Return the result we got earlier
        Ok(free_ptr.cast())
    }

    /// Safety: caller must ensure that the ptr was returned by this, and no double frees
    pub unsafe fn deallocate_one(&self, ptr: ptr::NonNull<u8>) {
        // Obtain previous free entry, if any
        let prev = self.next_free();

        // Write the link to this entry
        let field: ptr::NonNull<Option<ptr::NonNull<u8>>> = ptr.cast();
        *field.as_ptr() = prev.map(|p| p.cast());

        // Update bookkeeping
        self.set_next_free(Some(ptr.cast()));
    }
}

#[cfg(test)]
mod tests {
    use core::ptr;

    use super::*;

    #[test]
    fn simple() {
        let mut backing = vec![0; 1024];
        let allocator = BlockLLAllocator::new(&mut backing, 64);

        assert_eq!(allocator.capacity(), backing.len() / 64 - 1);
        assert_eq!(allocator.slots_available(), allocator.capacity());

        let b0 = allocator.allocate_one().expect("alloc");

        // Check that we do not get a segfault or anything
        unsafe {
            ptr::write(b0.as_ptr(), 1);
            assert_eq!(ptr::read(b0.as_ptr()), 1);
        }

        unsafe {
            allocator.deallocate_one(b0);
        }
    }

    #[test]
    fn use_full_capacity() {
        let mut backing = vec![0xdd; 1024];
        let allocator = BlockLLAllocator::new(&mut backing, 64);

        let mut blocks = Vec::new();
        for _ in 0..allocator.capacity() {
            let b = allocator.allocate_one().expect("alloc");
            // Check that we do not get a segfault or anything
            unsafe {
                ptr::write(b.as_ptr(), 1);
                assert_eq!(ptr::read(b.as_ptr()), 1);
            }
            blocks.push(b);
        }

        assert_eq!(allocator.slots_available(), 0);

        for b in blocks {
            unsafe {
                allocator.deallocate_one(b);
            }
        }

        assert_eq!(allocator.slots_available(), allocator.capacity());
    }

    #[test]
    fn no_fragmentation() {
        let mut backing = vec![0xdd; 1024];
        let allocator = BlockLLAllocator::new(&mut backing, 8);

        let mut blocks = Vec::new();
        for _ in 0..allocator.capacity() {
            blocks.push(allocator.allocate_one().expect("alloc"));
        }

        for i in (1..blocks.len()).step_by(2) {
            unsafe {
                allocator.deallocate_one(blocks[i]);
            }
        }

        assert_eq!(allocator.slots_available(), allocator.capacity() / 2);

        for i in (0..blocks.len()).step_by(2) {
            unsafe {
                allocator.deallocate_one(blocks[i]);
            }
        }

        assert_eq!(allocator.slots_available(), allocator.capacity());
    }

    #[test]
    fn almost_fuzz() {
        for (size, item_size) in [(16, 8), (64, 8), (128, 32)] {
            let mut backing = vec![0xdd; size];
            let allocator = BlockLLAllocator::new(&mut backing, item_size);

            for _ in 0..3 {
                let pre = allocator.slots_available();
                let b = allocator.allocate_one().expect("alloc");
                assert_eq!(allocator.slots_available(), pre - 1);
                unsafe {
                    allocator.deallocate_one(b);
                }
                assert_eq!(allocator.slots_available(), pre);
            }
        }
    }
}
