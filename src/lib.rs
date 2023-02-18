#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(adt_const_params, generic_const_exprs, generic_arg_infer)]
#![feature(slice_ptr_get, nonnull_slice_from_raw_parts, slice_ptr_len)]
#![feature(slice_split_at_unchecked)]
#![cfg_attr(test, feature(vec_into_raw_parts))]
#![allow(incomplete_features)]
#![deny(unused_must_use)]
#![deny(clippy::missing_safety_doc)]

extern crate alloc;

mod block;
mod blockll;
mod buddy;
mod buddy_group;
mod util;

pub use self::block::MemoryBlock;

pub use self::blockll::BlockLLAllocator;
pub use self::buddy::BuddyAllocator;
pub use self::buddy_group::BuddyGroupAllocator;
