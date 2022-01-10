pub(crate) fn div_ceil(a: usize, b: usize) -> usize {
    debug_assert!(b != 0);
    (a / b) + ((a % b != 0) as usize)
}

pub(crate) fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two()); // Disallows zero as well
    let align_mask = alignment - 1; // 0x100 => 0xff
    if value & align_mask == 0 {
        value
    } else {
        (value | align_mask) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(0, 1), 0);
        assert_eq!(div_ceil(1, 1), 1);
        assert_eq!(div_ceil(2, 1), 2);
        assert_eq!(div_ceil(3, 1), 3);
        assert_eq!(div_ceil(4, 1), 4);
        assert_eq!(div_ceil(5, 1), 5);
        assert_eq!(div_ceil(6, 1), 6);

        assert_eq!(div_ceil(0, 2), 0);
        assert_eq!(div_ceil(1, 2), 1);
        assert_eq!(div_ceil(2, 2), 1);
        assert_eq!(div_ceil(3, 2), 2);
        assert_eq!(div_ceil(4, 2), 2);
        assert_eq!(div_ceil(5, 2), 3);
        assert_eq!(div_ceil(6, 2), 3);

        assert_eq!(div_ceil(0, 3), 0);
        assert_eq!(div_ceil(1, 3), 1);
        assert_eq!(div_ceil(2, 3), 1);
        assert_eq!(div_ceil(3, 3), 1);
        assert_eq!(div_ceil(4, 3), 2);
        assert_eq!(div_ceil(5, 3), 2);
        assert_eq!(div_ceil(6, 3), 2);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(2, 1), 2);
        assert_eq!(align_up(0x10, 1), 0x10);
        assert_eq!(align_up(0x11, 1), 0x11);

        assert_eq!(align_up(0, 2), 0);
        assert_eq!(align_up(1, 2), 2);
        assert_eq!(align_up(2, 2), 2);
        assert_eq!(align_up(0x10, 2), 0x10);
        assert_eq!(align_up(0x11, 2), 0x12);

        assert_eq!(align_up(0, 0x10), 0x00);
        assert_eq!(align_up(1, 0x10), 0x10);
        assert_eq!(align_up(2, 0x10), 0x10);
        assert_eq!(align_up(0x10, 0x10), 0x10);
        assert_eq!(align_up(0x11, 0x10), 0x20);
    }
}
