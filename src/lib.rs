#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(adt_const_params, generic_const_exprs, generic_arg_infer)]
#![feature(slice_ptr_get, nonnull_slice_from_raw_parts, slice_ptr_len)]
#![allow(incomplete_features)]
#![deny(unused_must_use)]
#![deny(clippy::missing_safety_doc)]

extern crate alloc;

mod buddy;
mod util;

pub use self::buddy::BuddyAllocator;
