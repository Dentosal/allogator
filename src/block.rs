use core::ptr;
use rawpointer::PointerExt;

/// A non-null pointer with associated len, with
/// same safety guarantees as a raw pointer itself
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryBlock {
    pub ptr: ptr::NonNull<u8>,
    pub len: usize,
}
impl MemoryBlock {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn nn(&self) -> ptr::NonNull<u8> {
        self.ptr
    }

    pub fn as_mut(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// # Safety
    /// `at` must be in bounds
    pub unsafe fn split_at(self, at: usize) -> (Self, Self) {
        debug_assert!(at <= self.len);
        (
            Self {
                ptr: self.ptr,
                len: at,
            },
            Self {
                ptr: self.ptr.add(at),
                len: self.len - at,
            },
        )
    }

    #[cfg(test)]
    pub fn test_new(size: usize) -> Self {
        let v = vec![0xd7u8; size]; // Fill with nonzero bytes
        let (ptr, len, cap) = v.into_raw_parts();
        assert!(len == cap);
        assert!(len == size);
        Self {
            ptr: ptr::NonNull::new(ptr).unwrap(),
            len: size,
        }
    }

    #[cfg(test)]
    pub fn test_destroy(mut self) {
        drop(unsafe { Vec::from_raw_parts(self.ptr.as_mut(), self.len, self.len) });
    }
}
