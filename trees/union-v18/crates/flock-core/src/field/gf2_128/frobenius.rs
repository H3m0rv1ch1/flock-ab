//! Constant-address Frobenius maps for the long steps of field inversion.
//! GFNI applies eight independent public 8x8 binary matrices per vector;
//! two vectors cover all sixteen output bytes for each input byte.
use super::F128;
use core::arch::x86_64::*;
use std::sync::LazyLock;

#[repr(align(64))]
struct Matrices([[u64; 16]; 16]);

fn build(count: usize) -> Matrices {
    let mut matrices = Matrices([[0; 16]; 16]);
    for input in 0..16 {
        for bit in 0..8 {
            let mut value = if input < 8 {
                F128::new(1 << (8 * input + bit), 0)
            } else {
                F128::new(0, 1 << (8 * (input - 8) + bit))
            };
            for _ in 0..count {
                value = value.square();
            }
            let bytes = (u128::from(value.lo) | (u128::from(value.hi) << 64)).to_le_bytes();
            for (output, byte) in bytes.into_iter().enumerate() {
                for row in 0..8 {
                    matrices.0[input][output] |=
                        u64::from((byte >> row) & 1) << ((7 - row) * 8 + bit);
                }
            }
        }
    }
    matrices
}

#[inline]
pub(super) fn apply(value: F128, count: usize) -> Option<F128> {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_FROBENIUS_GFNI").is_none());
    if !*ENABLED { return None; }
    static SEVEN: LazyLock<Matrices> = LazyLock::new(|| build(7));
    static TWENTY_ONE: LazyLock<Matrices> = LazyLock::new(|| build(21));
    static SIXTY_THREE: LazyLock<Matrices> = LazyLock::new(|| build(63));
    let matrices = match count {
        7 => &*SEVEN,
        21 => &*TWENTY_ONE,
        63 => &*SIXTY_THREE,
        _ => return None,
    };
    // All required ISA features are compile-time gates on this module.
    Some(unsafe { transform(value, matrices) })
}

#[inline]
unsafe fn transform(value: F128, matrices: &Matrices) -> F128 {
    unsafe {
        let mut low = _mm512_setzero_si512();
        let mut high = low;
        macro_rules! byte {
            ($i:expr, $word:expr, $shift:expr) => {{
                let input = _mm512_set1_epi8(($word >> $shift) as i8);
                let matrix = matrices.0[$i].as_ptr();
                low = _mm512_xor_si512(low,
                    _mm512_gf2p8affine_epi64_epi8::<0>(input, _mm512_loadu_si512(matrix.cast())));
                high = _mm512_xor_si512(high,
                    _mm512_gf2p8affine_epi64_epi8::<0>(input, _mm512_loadu_si512(matrix.add(8).cast())));
            }};
        }
        byte!(0, value.lo, 0); byte!(1, value.lo, 8);
        byte!(2, value.lo, 16); byte!(3, value.lo, 24);
        byte!(4, value.lo, 32); byte!(5, value.lo, 40);
        byte!(6, value.lo, 48); byte!(7, value.lo, 56);
        byte!(8, value.hi, 0); byte!(9, value.hi, 8);
        byte!(10, value.hi, 16); byte!(11, value.hi, 24);
        byte!(12, value.hi, 32); byte!(13, value.hi, 40);
        byte!(14, value.hi, 48); byte!(15, value.hi, 56);
        // Each qword contains eight equal copies of its output byte.
        let indices = _mm512_set1_epi64(0x3830_2820_1810_0800);
        let low = _mm512_castsi512_si128(_mm512_permutexvar_epi8(indices, low));
        let high = _mm512_castsi512_si128(_mm512_permutexvar_epi8(indices, high));
        F128::new(_mm_cvtsi128_si64(low) as u64, _mm_cvtsi128_si64(high) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_match_repeated_squaring() {
        for count in [7, 21, 63] {
            let matrices = build(count);
            for bit in 0..128 {
                let value = if bit < 64 { F128::new(1 << bit, 0) }
                    else { F128::new(0, 1 << (bit - 64)) };
                let mut expected = value;
                for _ in 0..count { expected = expected.square(); }
                assert_eq!(unsafe { transform(value, &matrices) }, expected);
            }
            let mut state = 0x123456789abcdef0u64;
            for _ in 0..256 {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                let value = F128::new(state, state.rotate_left(29));
                let mut expected = value;
                for _ in 0..count { expected = expected.square(); }
                assert_eq!(unsafe { transform(value, &matrices) }, expected);
            }
        }
    }
}
