use super::*;

#[test]
fn optimized_projection_matches_original_arbitrary_matrices() {
    let mut state = 0x918c_ad73_e200_571bu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..2048 {
        let mut input: [u64; 8] = core::array::from_fn(|_| next());
        if case < 512 {
            input.fill(0);
            input[case / 64] = 1u64 << (case % 64);
        }
        let matrices: [u64; 128] = core::array::from_fn(|_| next());
        unsafe {
            let expected: [u8; 512] =
                core::mem::transmute(affine_prepared_reference(input.as_ptr(), matrices.as_ptr()));
            let actual: [u8; 512] = core::mem::transmute(global_cse::affine27_depth(
                input.as_ptr(),
                matrices.as_ptr(),
            ));
            assert_eq!(actual, expected, "projection case={case}");
        }
    }
}
