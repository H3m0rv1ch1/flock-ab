//! Exact paired input generation without intermediate eight-lane structures.
use super::*;

#[inline(always)]
pub(super) unsafe fn prepare(init:u64,base:usize)->Prepared16 {
    unsafe {
        let stride=crate::seed_pipe::GOLDEN.wrapping_mul(crate::seed_pipe::DRAWS_PER_BLOCK as u64);
        let first=init.wrapping_add((base as u64).wrapping_mul(stride));
        let mut lo=_mm512_setr_epi64(
            first as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(1,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(2,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(3,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(4,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(5,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(6,true) as u64)) as i64,
            first.wrapping_add(stride.wrapping_mul(compression_lane_offset(7,true) as u64)) as i64);
        let mut hi=_mm512_add_epi64(lo,_mm512_set1_epi64(stride.wrapping_mul(2048) as i64));
        let step=_mm512_set1_epi64(crate::seed_pipe::GOLDEN as i64);
        let indices=_mm512_setr_epi32(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
        let mut draw=|| {
            lo=_mm512_add_epi64(lo,step);hi=_mm512_add_epi64(hi,step);
            _mm512_permutex2var_epi32(mix_u64x8(lo),indices,mix_u64x8(hi))
        };
        // Keep independent SplitMix draws visible together to the scheduler.
        // Array lengths preserve exactly eight CV and sixteen message draws.
        let cv=[draw(),draw(),draw(),draw(),draw(),draw(),draw(),draw()];
        let message=[
            draw(),draw(),draw(),draw(),draw(),draw(),draw(),draw(),
            draw(),draw(),draw(),draw(),draw(),draw(),draw(),draw(),
        ];
        let counter_lo=draw();
        Prepared16 {cv,message,counter_lo,counter_hi:_mm512_setzero_si512(),
            block_len:_mm512_set1_epi32(64),flags:_mm512_set1_epi32(11)}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe fn words(v:V16)->[u32;16] {
        let mut a=[0u32;16];unsafe {_mm512_storeu_si512(a.as_mut_ptr().cast(),v)};a
    }
    #[test]
    fn fused_prepare16_matches_two_ordered_octa_inputs() {
        for seed in [0,1,0x123456789abcdef0,u64::MAX] {
            for base in [0usize,1,3,31,127,1023,2047,4095,65535,131071,260000] {
                unsafe {
                    let a=prepare16_legacy(seed,base);let b=prepare(seed,base);
                    for i in 0..8 {assert_eq!(words(a.cv[i]),words(b.cv[i]),"seed={seed:x},base={base},cv={i}");}
                    for i in 0..16 {assert_eq!(words(a.message[i]),words(b.message[i]),"seed={seed:x},base={base},message={i}");}
                    for (x,y) in [(a.counter_lo,b.counter_lo),(a.counter_hi,b.counter_hi),(a.block_len,b.block_len),(a.flags,b.flags)] {
                        assert_eq!(words(x),words(y),"seed={seed:x},base={base}");
                    }
                }
            }
        }
    }
}
