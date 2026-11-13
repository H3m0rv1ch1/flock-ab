//! Proof-local root matrices shared by folds at distinct challenge points.
//! The cache contains witness bits only; each consumer builds fresh GFNI
//! matrices from its own challenge weights. The logical cache is proof-local;
//! its backing allocation returns to the existing untagged scratch pool.
//! Every live byte is overwritten before the next cache can read it.
use super::*;
use core::arch::x86_64::*;
use rayon::prelude::*;

#[repr(C, align(64))]
pub(super) struct Block([u8; 512]);

// Only allocation capacity is recycled. take_f128 invalidates previous
// provenance; prepare or witness generation overwrites every live block
// before publication.
struct PooledStorage(Vec<F128>);

pub type RootPairGenerator = unsafe fn(u64, usize, usize, *mut u8, *mut u8, *mut u8, *mut u8);

impl Drop for PooledStorage {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.0));
    }
}

pub struct RootTileCache {
    storage: PooledStorage,
    offset: usize,
    tiles: usize,
    blocks: usize,
    word_major: bool,
    fixed_roots: bool,
    cross_quarter: bool,
    /// The witness producer skipped z's DRAM stores; this cache is then the
    /// only complete source of z (see `pcs::elided_z`).
    z_elided: bool,
    generator: Option<(u64, unsafe fn(u64, usize, *mut u8))>,
    pair_generator: Option<RootPairGenerator>,
    stage_generator: Option<unsafe fn(u64, usize, usize, *mut u8, *mut u8, *const u8)>,
}

/// Raw read-only view of witness-produced word-major root tiles: `tiles`
/// tiles of 176 groups x 512 bytes; group `g` of tile `t` starts at
/// `base + (t * 176 + g) * 512`, octa `o` of that group at `+ o * 64`.
#[derive(Clone, Copy, Debug)]
pub struct WitnessTileView {
    pub base: usize,
    pub tiles: usize,
    pub cross_quarter: bool,
    pub generator: Option<(u64, unsafe fn(u64, usize, *mut u8))>,
    pub pair_generator: Option<RootPairGenerator>,
    pub stage_generator: Option<unsafe fn(u64, usize, usize, *mut u8, *mut u8, *const u8)>,
}

#[repr(align(64))]
struct GeneratedTiles([core::mem::MaybeUninit<u8>; 4 * 176 * 512]);

#[derive(Clone, Copy)]
pub(super) struct Capture {
    ptr: *mut Block,
    tiles: usize,
    blocks: usize,
    word_major: bool,
}

// A witness stripe contains eight low u32 words, then eight high u32 words.
// Gather each wire-byte column in reversed octa-lane order for GFNI.
const WORD_TO_COLUMNS: [u8; 64] = {
    let mut a = [0; 64];
    let mut i = 0;
    while i < 64 {
        let byte = i / 8;
        let lane = 7 - i % 8;
        a[i] = (if byte < 4 { lane * 4 + byte } else { 32 + lane * 4 + byte - 4 }) as u8;
        i += 1;
    }
    a
};

/// Inverse of the witness reader's byte/bit transpose. Used only by
/// incumbent capture when checking the witness-produced word encoding.
#[inline(always)]
unsafe fn raw_words(row: __m512i) -> __m512i {
    const REVERSE: [u8; 64] = {
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 { a[i] = ((i & 8) + 7 - i % 8) as u8; i += 1; }
        a
    };
    unsafe {
        let row = _mm512_shuffle_epi8(row, _mm512_loadu_si512(REVERSE.as_ptr().cast()));
        let matrix = _mm512_gf2p8affine_epi64_epi8::<0>(
            _mm512_set1_epi64(0x0102040810204080u64 as i64), row);
        const COLUMNS_TO_WORD: [u8; 64] = {
            let mut a = [0; 64];
            let mut i = 0;
            while i < 64 {
                a[WORD_TO_COLUMNS[i] as usize] = i as u8;
                i += 1;
            }
            a
        };
        _mm512_permutexvar_epi8(_mm512_loadu_si512(COLUMNS_TO_WORD.as_ptr().cast()), matrix)
    }
}

// SAFETY: capture is confined to prepare(). Each producer claims a distinct
// tile through the existing static partition or atomic tile allocator. A
// checked RootSplicePlan emits each block in that tile exactly once. No read
// reference to the allocation exists until all producers join and fence.
unsafe impl Send for Capture {}
unsafe impl Sync for Capture {}

impl Capture {
    #[inline]
    pub(super) unsafe fn publish(&self, tile: usize, block: usize, rows: &[__m512i; 8]) {
        debug_assert!(tile < self.tiles && block < self.blocks);
        // SAFETY: each (tile, block) is exclusive to this producer; Block's
        // alignment and its 512-byte stride satisfy every streaming store.
        unsafe {
            let dst = self.ptr.add(tile * self.blocks + block).cast::<__m512i>();
            for (i, value) in rows.iter().enumerate() {
                let value = if self.word_major { raw_words(*value) } else { *value };
                _mm512_stream_si512(dst.add(i), value);
            }
        }
    }
}

fn prepare(
    splice: &RootSplicePlan,
    z: &[F128],
    k: usize,
    eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
    word_major: bool,
) -> (RootTileCache, Vec<F128>) {
    let tiles = z.len() / (k / 128) / (DIRECT_FOLD_TILE_STRIPES * 8);
    let blocks = splice.root_bits.div_ceil(64);
    assert!(tiles > 0 && blocks > 0);
    assert_eq!(z.len(), tiles * (k / 128) * DIRECT_FOLD_TILE_STRIPES * 8);
    // The shared pool's F128 allocation guarantees16-byte alignment. Three
    // spare elements let us align the live span to64 without changing the
    // allocation's original pointer or its deallocation layout.
    let fields = tiles.checked_mul(blocks).unwrap().checked_mul(32).unwrap();
    let mut storage = PooledStorage(crate::scratch::take_f128(fields.checked_add(3).unwrap()));
    let offset = storage.0.as_ptr().align_offset(64);
    assert!(offset <= 3);
    // SAFETY: fields+3 elements were reserved; offset is at most3, so all
    // live blocks fit. F128 and Block both contain only plain integer bytes.
    let aligned = unsafe { storage.0.as_mut_ptr().add(offset).cast::<Block>() };
    let mut writer = splice.clone();
    writer.capture = Some(Capture {
        ptr: aligned,
        tiles,
        blocks,
        word_major,
    });
    let folded = writer.fold(z, k, eq8_at);
    // Every live block, including the tail's zero padding, is initialized.
    // All producer joins and their SFENCEs completed before this readable
    // cache is constructed. Padding outside the aligned live span is never
    // read. On unwind, PooledStorage returns stale capacity under the same
    // write-before-read contract as any other global scratch buffer.
    (
        RootTileCache {
            storage,
            offset,
            tiles,
            blocks,
            word_major,
            fixed_roots: false,
            cross_quarter: false,
            z_elided: false,
            generator: None,
            pair_generator: None, stage_generator: None,
        },
        folded,
    )
}

pub(super) fn fold(
    source: &RootFoldSource,
    splice: &RootSplicePlan,
    z: &[F128],
    k: usize,
    eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
    eq8_indices: &(impl Fn([usize; 8]) -> [F128; 8] + Sync),
) -> Vec<F128> {
    let mut first_fold = None;
    let cache = source.cached_tiles.get_or_init(|| {
        let (cache, folded) = prepare(splice, z, k, eq8_at, false);
        first_fold = Some(folded);
        cache
    });
    if let Some(folded) = first_fold {
        folded
    } else {
        cache.fold_indexed_with_mode(eq8_at, eq8_indices, RootTileCache::batch_mode())
    }
}

impl RootTileCache {
    /// Adopt fully initialized witness-produced BLAKE3 root tiles.
    ///
    /// # Safety
    /// The aligned live span must contain this proof's selected octa words:
    /// 64-compression tiles, 176 groups, eight 64-byte stripes. A stripe holds
    /// eight low u32 words followed by eight high u32 words in octa-lane order.
    /// The last 15 root bits of every lane are zero.
    /// Wires 1089..1153 are omitted: every compression must have block_len=64
    /// and flags=11. Wire 512 is the constant-one pin. All other roots retain
    /// their original ordering, including counter wire 1088.
    /// Every producer must have joined and executed SFENCE before this call.
    /// The later RootFoldSource must use the same witness and root ordering.
    pub unsafe fn from_witness_storage(storage: Vec<F128>, compressions: usize) -> Self {
        let storage = PooledStorage(storage);
        assert!(compressions > 0 && compressions.is_multiple_of(64));
        let tiles = compressions / 64;
        let blocks = 176;
        let fields = tiles.checked_mul(blocks).unwrap().checked_mul(32).unwrap();
        let offset = storage.0.as_ptr().align_offset(64);
        assert!(offset <= 3 && offset.checked_add(fields).unwrap() <= storage.0.len());
        Self { storage, offset, tiles, blocks, word_major: true, fixed_roots: true, cross_quarter: false, z_elided: false, generator: None, pair_generator: None, stage_generator: None }
    }

    /// Exact ranked root tiles generated into consumer-local scratch.
    ///
    /// # Safety
    /// For each tile in 0..4096, `generate(seed, tile, dst)` must initialize
    /// all 176*512 bytes in the cross-quarter witness encoding. `dst` is
    /// 64-byte aligned and exclusively owned for the duration of the call.
    /// The callback must not retain it or depend on mutable external state.
    /// An optional pair callback initializes the two octa stripes at pair*128
    /// in every group and writes [2][56] packed BLAKE3 dependency ZMMs plus
    /// [2][4] low and high output ZMMs. Its three auxiliary spans are aligned,
    /// exclusively owned and disjoint from the root tile and each other.
    pub unsafe fn from_cross_quarter_generator(
        seed: u64, generate: unsafe fn(u64, usize, *mut u8), pair_generator: Option<RootPairGenerator>,
    ) -> Self {
        Self {
            storage: PooledStorage(Vec::new()), offset: 0, tiles: 4096,
            blocks: 176, word_major: true, fixed_roots: true,
            cross_quarter: true, z_elided: false, generator: Some((seed, generate)), pair_generator, stage_generator: None,
        }
    }

    /// Supply a proof-local full-witness pair generator for PCS folds.
    ///
    /// # Safety
    /// With a null final argument, overwrite two disjoint, 64-byte aligned
    /// arrays of 242 ZMMs with canonical qwords for physical octas pair*2 and
    /// pair*2+1. With a nonnull final argument, read 8192 immutable aligned
    /// bytes (64 repeated-F128 weight ZMMs, then 64 low/high-XOR ZMMs), ignore
    /// the second output pointer, and overwrite 512 aligned bytes at the first
    /// output in [half][octa][parity] order, four canonical F128 lanes per ZMM.
    /// Lane l represents compression 2*l+parity within that octa. Weights and
    /// output must be disjoint; seed/tile mapping must agree with the cache.
    pub unsafe fn set_stage_generator(&mut self, generate: unsafe fn(u64,usize,usize,*mut u8,*mut u8,*const u8)) {
        assert!(self.generator.is_some() && self.cross_quarter);
        self.stage_generator=Some(generate);
    }

    /// Record that the producer skipped z stores for some covered compressions.
    pub fn mark_z_elided(&mut self) {
        assert!(self.fixed_roots && self.word_major);
        self.z_elided = true;
    }

    pub fn z_elided(&self) -> bool {
        self.z_elided
    }

    /// Word-major witness tile view (witness-born caches only).
    pub fn witness_view(&self) -> Option<WitnessTileView> {
        (self.fixed_roots && self.word_major && self.blocks == 176).then(|| WitnessTileView {
            base: self.storage.0.as_ptr() as usize + self.offset * core::mem::size_of::<F128>(),
            tiles: self.tiles,
            cross_quarter: self.cross_quarter,
            generator: self.generator,
            pair_generator: self.pair_generator,
            stage_generator: self.stage_generator,
        })
    }

    pub(super) fn matches_geometry(&self, compressions: usize, root_bits: usize) -> bool {
        let blocks = if self.fixed_roots { 176 } else { 177 };
        root_bits == 11313 && self.blocks == blocks && self.tiles.checked_mul(64) == Some(compressions)
    }

    /// Expensive diagnostic oracle: compare every cache byte, including tail
    /// padding, with canonical-witness capture at the same root geometry.
    pub fn assert_matches_witness(&self, z: &[F128], k_log: usize, plan: &LinearFoldPlan) {
        if let Some((seed, generate)) = self.generator {
            let mut storage = crate::scratch::take_f128(self.tiles * 176 * 32 + 3);
            let offset = storage.as_ptr().align_offset(64);
            assert!(offset <= 3);
            unsafe {
                let base = storage.as_mut_ptr().add(offset).cast::<u8>();
                for tile in 0..self.tiles {
                    generate(seed, tile, base.add(tile * 176 * 512));
                }
                let cache = Self::from_cross_quarter_witness_storage(storage, self.tiles * 64);
                cache.assert_matches_witness(z, k_log, plan);
            }
            return;
        }
        let splice = RootSplicePlan::new(plan, 1 << k_log).expect("root splice geometry");
        let (reference, _) = prepare(&splice, z, 1 << k_log,
            &|_| [F128::ZERO; 8], self.word_major);
        assert_eq!(self.tiles, reference.tiles);
        assert_eq!(self.blocks + usize::from(self.fixed_roots), reference.blocks);
        for tile in 0..self.tiles {
            for block in 0..self.blocks {
                let source = block + usize::from(self.fixed_roots && block >= 14);
                let mut expected = reference.block(tile * reference.blocks + source).0;
                if self.fixed_roots && block == 13 {
                    assert!(self.word_major);
                    expected = reference.block(tile * reference.blocks + 14).0;
                    let counter = &reference.block(tile * reference.blocks + 13).0;
                    // Selected words splice the counter bit into the next
                    // source group's bit zero in each octa lane.
                    for stripe in 0..8 {
                        for lane in 0..8 {
                            let byte = stripe * 64 + lane * 4;
                            expected[byte] = (expected[byte] & !1) | (counter[byte] & 1);
                        }
                    }
                }
                assert_eq!(self.block(tile * self.blocks + block).0, expected,
                    "root cache tile {tile}, block {block}");
            }
        }
    }

    #[inline(always)]
    fn block(&self, index: usize) -> &Block {
        assert!(self.generator.is_none());
        assert!(index < self.tiles * self.blocks);
        // SAFETY: prepare initializes every block, and from_witness_storage
        // requires the same initialization before publication. The live
        // span is 64-byte aligned and fits within the owned vector.
        unsafe {
            &*self
                .storage
                .0
                .as_ptr()
                .add(self.offset)
                .cast::<Block>()
                .add(index)
        }
    }

    /// Adopt complete root storage in two-seed-group physical order.
    ///
    /// # Safety
    /// Every selected root word must be initialized at physical index
    /// p=g*256+kpair*8+lane for original compression
    /// c=(lane/2)*65536+(2*kpair+lane%2)*1024+g. All writers must be joined.
    pub unsafe fn from_cross_quarter_witness_storage(storage: Vec<F128>, compressions: usize) -> Self {
        assert_eq!(compressions, 1 << 18);
        let mut result = unsafe { Self::from_witness_storage(storage, compressions) };
        result.cross_quarter = true;
        result
    }

    fn batch_mode() -> bool {
        static BATCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_ROOT_CACHE_BATCH4").is_none()
        });
        *BATCH
    }

    fn fold(&self, eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync)) -> Vec<F128> {
        self.fold_with_mode(eq8_at, Self::batch_mode())
    }

    fn cross_quarter_indices(physical: usize) -> [usize; 8] {
        debug_assert!(physical.is_multiple_of(8) && physical < (1 << 18));
        core::array::from_fn(|lane| {
            let p = physical + lane;
            let group = p / 256;
            let kpair = (p % 256) / 8;
            let lane = p % 8;
            (lane / 2) * 65536 + (2 * kpair + lane % 2) * 1024 + group
        })
    }

    // Keep the consecutive-weight API available to existing independent
    // consumer tests. Production's reordered cache takes the indexed route.
    fn fold_with_mode(
        &self, eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync), batch_word_tiles: bool,
    ) -> Vec<F128> {
        let logical = |physical| if self.cross_quarter {
            Self::cross_quarter_indices(physical).map(|c| eq8_at(c & !7)[c & 7])
        } else { eq8_at(physical) };
        self.fold_physical_with_mode(&logical, batch_word_tiles)
    }

    fn fold_indexed_with_mode(
        &self,
        eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
        eq8_indices: &(impl Fn([usize; 8]) -> [F128; 8] + Sync),
        batch_word_tiles: bool,
    ) -> Vec<F128> {
        if !self.cross_quarter { return self.fold_with_mode(eq8_at, batch_word_tiles); }
        let logical = |physical| eq8_indices(Self::cross_quarter_indices(physical));
        self.fold_physical_with_mode(&logical, batch_word_tiles)
    }

    fn fold_physical_with_mode(
        &self, logical: &(impl Fn(usize) -> [F128; 8] + Sync), batch_word_tiles: bool,
    ) -> Vec<F128> {
        if batch_word_tiles && self.word_major && self.fixed_roots {
            self.fold_impl::<true>(logical)
        } else {
            self.fold_impl::<false>(logical)
        }
    }

    fn fold_impl<const BATCH_WORD_TILES: bool>(
        &self, logical_eq8: &(impl Fn(usize) -> [F128; 8] + Sync),
    ) -> Vec<F128> {
        static PREFETCH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_ROOT_CACHE_PREFETCH").is_none()
        });
        let prefetch_blocks = BATCH_WORD_TILES && *PREFETCH;
        // Resolve the tile group once, outside the parallel reader. Two-tile
        // groups reduced measured lincheck latency on the ranked CPU shape.
        static WORD_BATCH: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            std::env::var("FLOCK_ROOT_FOLD_BATCH")
                .ok().and_then(|s| s.parse().ok())
                .filter(|n| (1..=4).contains(n)).unwrap_or(3)
        });
        let word_batch = if BATCH_WORD_TILES { *WORD_BATCH } else { 4 };
        static TAIL1: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
            if std::env::var_os("FLOCK_NO_SL1_FOLD_TAIL1").is_some() {
                return None;
            }
            Some(std::env::var("FLOCK_SL1_TAIL1_MULT").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
        });
        let tail1 = TAIL1.map_or(0, |m| m * word_batch * rayon::current_num_threads().max(1));
        let tiles_per_worker = self.tiles.div_ceil(rayon::current_num_threads().max(1));
        let workers = self.tiles.div_ceil(tiles_per_worker);
        let stride = self.blocks * 1024;
        #[cfg(not(test))]
        let mut planes = crate::alloc_uninit_vec::<core::mem::MaybeUninit<u8>>(workers * stride);
        // The ranked prover's global recycler aligns allocations >=32 KiB to
        // 64 bytes. Core-only test binaries use System instead, so emulate that
        // alignment for the complete-reader premise rather than timing extra
        // split-line loads/stores absent from production. The leaf oracle
        // separately exercises unaligned raw buffers.
        #[cfg(test)]
        let mut plane_storage =
            crate::alloc_uninit_vec::<core::mem::MaybeUninit<u8>>(workers * stride + 63);
        #[cfg(test)]
        let plane_offset = plane_storage.as_ptr().align_offset(64);
        #[cfg(test)]
        let mut planes = &mut plane_storage[plane_offset..plane_offset + workers * stride];
        let mut active = vec![0u8; workers];
        let next = std::sync::atomic::AtomicUsize::new(0);
        let dynamic = dynamic_tiles_enabled();
        crate::rstrace::ev("rf:par0", workers as u32);
        planes
            .par_chunks_mut(stride)
            .zip(active.par_iter_mut())
            .enumerate()
            .for_each(|(worker, (dst, active))| {
                crate::rstrace::evd("rf:w0", worker as u32);
                // MaybeUninit bytes require no initialization; each live tile
                // is fully overwritten by the checked generator before read.
                let mut generated = self.generator.map(|_| unsafe {
                    Box::<GeneratedTiles>::new_uninit().assume_init()
                });
                let mut mats = [0u64; 128];
                // Const specialization lets the false worker discard the
                // entire batch array. The true worker initializes it once,
                // then fully overwrites only live slots before each call.
                let mut batch_mats = BATCH_WORD_TILES.then(|| [[0u64; 128]; 4]);
                let mut first = true;
                let (mut lo, mut hi) = if dynamic {
                    (0, 0)
                } else {
                    (
                        worker * tiles_per_worker,
                        ((worker + 1) * tiles_per_worker).min(self.tiles),
                    )
                };
                loop {
                    if lo == hi {
                        if !dynamic {
                            break;
                        }
                        // Lane sl1 (`FLOCK_NO_SL1_FOLD_TAIL1`): single-tile
                        // claims once fewer than `tail1` tiles are left, so the
                        // workers' last claims end closer together (the plane
                        // reduce waits for the last one). XOR-accumulated
                        // planes: any tile-to-worker split gives the same sum.
                        let take = if tail1 != 0
                            && self.tiles.saturating_sub(next.load(std::sync::atomic::Ordering::Relaxed)) <= tail1
                        {
                            1
                        } else {
                            word_batch
                        };
                        lo = next.fetch_add(take, std::sync::atomic::Ordering::Relaxed);
                        if lo >= self.tiles {
                            break;
                        }
                        super::sl1_rc_warm_step(self.tiles - lo);
                        hi = (lo + take).min(self.tiles);
                    }
                    if BATCH_WORD_TILES {
                        let tile = lo;
                        let count = (hi - lo).min(word_batch);
                        lo += count;
                        let source = if let Some((seed, generate)) = self.generator {
                            let base = generated.as_mut().unwrap().0.as_mut_ptr().cast::<u8>();
                            for slot in 0..count {
                                unsafe { generate(seed, tile + slot, base.add(slot * self.blocks * 512)); }
                            }
                            base.cast_const()
                        } else {
                            unsafe { self.storage.0.as_ptr().add(self.offset).cast::<u8>()
                                .add(tile * self.blocks * 512) }
                        };
                        let matrices = batch_mats.as_mut().expect("selected-word batch matrices");
                        for (slot, matrix) in matrices[..count].iter_mut().enumerate() {
                            for t in 0..DIRECT_FOLD_TILE_STRIPES {
                                let eq8 = logical_eq8(8 * ((tile + slot) * DIRECT_FOLD_TILE_STRIPES + t));
                                kernels::fold_mats_from_basis(&eq8,
                                    &mut matrix[t * 16..(t + 1) * 16]);
                            }
                        }
                        for block in 0..self.blocks {
                            // SAFETY: tile + count <= hi <= self.tiles, each
                            // source block has eight initialized 64-byte rows,
                            // and actual tile stride is blocks*512. Arrays of
                            // u64 have contiguous layout; every live matrix
                            // slot was fully overwritten above, and the leaf
                            // reads exactly count slots. Every dst byte is
                            // exclusively owned by this worker. On its first
                            // batch the leaf initializes all sixteen planes
                            // without reading dst; later calls load them once.
                            unsafe {
                                kernels::gfni_fold_word_tiles4(
                                    source.add(block * 512),
                                    matrices.as_ptr().cast::<u64>(),
                                    dst.as_mut_ptr().cast::<u8>().add(block * 1024),
                                    self.blocks * 512, count, first,
                                    WORD_TO_COLUMNS.as_ptr(),
                                    prefetch_blocks && block + 1 < self.blocks);
                            }
                        }
                        first = false;
                        continue;
                    }
                    let tile = lo;
                    lo += 1;
                    let generated_base = self.generator.map(|(seed, generate)| {
                        let base = generated.as_mut().unwrap().0.as_mut_ptr().cast::<u8>();
                        unsafe { generate(seed, tile, base); }
                        base
                    });
                    for t in 0..DIRECT_FOLD_TILE_STRIPES {
                        let eq8 = logical_eq8(8 * (tile * DIRECT_FOLD_TILE_STRIPES + t));
                        kernels::fold_mats_from_basis(&eq8, &mut mats[t * 16..(t + 1) * 16]);
                    }
                    for block in 0..self.blocks {
                        // SAFETY: the fully initialized immutable cache is
                        // 64-byte aligned; this worker owns every dst plane.
                        // Its first tile initializes all live planes before
                        // subsequent tiles load their accumulators.
                        unsafe {
                            let src = if let Some(base) = generated_base {
                                base.add(block * 512).cast::<__m512i>()
                            } else {
                                self.block(tile * self.blocks + block).0.as_ptr().cast::<__m512i>()
                            };
                            let rows = core::array::from_fn(|i| {
                                let row = _mm512_load_si512(src.add(i));
                                if self.word_major {
                                    let matrix = _mm512_permutexvar_epi8(
                                        _mm512_loadu_si512(WORD_TO_COLUMNS.as_ptr().cast()), row);
                                    _mm512_gf2p8affine_epi64_epi8::<0>(
                                        _mm512_set1_epi64(0x8040201008040201u64 as i64), matrix)
                                } else { row }
                            });
                            kernels::gfni_fold_rows(
                                &rows,
                                &mats,
                                dst.as_mut_ptr().cast::<u8>().add(block * 1024),
                                first,
                            );
                        }
                    }
                    first = false;
                }
                *active = u8::from(!first);
                crate::rstrace::evd("rf:w1", worker as u32);
            });
        crate::rstrace::ev("rf:red0", 0);
        let active_workers: Vec<usize> = active
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| (v != 0).then_some(i))
            .collect();
        let output_blocks = self.blocks + usize::from(self.fixed_roots);
        let mut out = vec![F128::ZERO; output_blocks * 64];
        out.par_chunks_mut(64).enumerate().for_each(|(block, dst)| {
            // Leave the omitted group clear. The boundary group's surviving
            // bits are moved into it after the parallel reduction joins.
            if self.fixed_roots && block == 14 { return; }
            let source = block - usize::from(self.fixed_roots && block > 14);
            // SAFETY: every active worker initialized this live block and
            // the Rayon producer join completed before reduction starts.
            unsafe {
                reduce_worker_plane_block(&planes, stride, &active_workers, source, dst);
            }
        });
        crate::rstrace::ev("lcf:reduced", 0);
        if self.fixed_roots {
            out.copy_within(833..896, 897);
            out[833..897].fill(F128::ZERO);
            // Wire 512 is root 256 (output wires 256..512 are reconstructed).
            // The already-folded constant pin equals the sum of the outer
            // weights, so this also preserves arbitrary, unnormalized bases.
            let one = out[256];
            for column in [839, 865, 866, 868] { out[column] = one; }
        }
        crate::rstrace::ev("rf:red1", 0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// The oracle reads the selected-word wire bits directly. It neither uses
    /// WORD_TO_COLUMNS nor derives its expected planes from a SIMD transpose.
    #[test]
    fn selected_word_fold4_matches_byte_plane_oracle() {
        const STRIDE: usize = 176 * 512;
        let mut source = vec![0xa7u8; 4 * STRIDE];
        for group in [13usize, 175] {
            let mut mats = [[0u64; 128]; 4];
            let mut oracle_tiles = [[0u8; 1024]; 4];
            for tile in 0..4 {
                for stripe in 0..8 {
                    let weights = core::array::from_fn::<_, 8, _>(|lane| {
                        let n = (tile * 64 + stripe * 8 + lane + 31) as u64;
                        F128::new(n.wrapping_mul(0x9e3779b97f4a7c15),
                            n.rotate_left(29).wrapping_mul(0xbf58476d1ce4e5b9))
                    });
                    kernels::fold_mats_from_basis(&weights,
                        &mut mats[tile][stripe * 16..(stripe + 1) * 16]);
                    for lane in 0..8 {
                        let n = (tile * 64 + stripe * 8 + lane + 91) as u64;
                        let mut bits = n.wrapping_mul(0xd6e8feb86659fd93)
                            ^ n.rotate_left(37).wrapping_mul(0xa0761d6478bd642f);
                        if group == 175 { bits &= (1u64 << 49) - 1; }
                        let row = tile * STRIDE + group * 512 + stripe * 64;
                        source[row + lane * 4..row + lane * 4 + 4]
                            .copy_from_slice(&(bits as u32).to_le_bytes());
                        source[row + 32 + lane * 4..row + 36 + lane * 4]
                            .copy_from_slice(&((bits >> 32) as u32).to_le_bytes());
                        let mut bytes = [0u8; 16];
                        bytes[..8].copy_from_slice(&weights[lane].lo.to_le_bytes());
                        bytes[8..].copy_from_slice(&weights[lane].hi.to_le_bytes());
                        for bit in 0..64 {
                            if bits >> bit & 1 != 0 {
                                for plane in 0..16 {
                                    oracle_tiles[tile][plane * 64 + bit] ^= bytes[plane];
                                }
                            }
                        }
                    }
                }
            }
            for count in 1..=4 {
                for seed_zero in [true, false] { for prefetch in [false, true] {
                    // Deliberately stale output, with untouched sentinels on
                    // both sides. Source group 175 ends at a true tile edge.
                    let mut expected = [0u8; 1024];
                    let mut old = vec![0x5du8; 64 + 1024 + 64];
                    let mut new = old.clone();
                    for i in 0..1024 {
                        old[64 + i] = (i as u8).wrapping_mul(73) ^ 0xc3;
                        new[64 + i] = old[64 + i];
                        expected[i] = if seed_zero { 0 } else { old[64 + i] };
                        for tile in 0..count { expected[i] ^= oracle_tiles[tile][i]; }
                    }
                    unsafe {
                        for tile in 0..count {
                            let src = source.as_ptr().add(tile * STRIDE + group * 512);
                            let rows = core::array::from_fn(|stripe| {
                                let row = _mm512_loadu_si512(src.add(stripe * 64).cast());
                                let permuted = _mm512_permutexvar_epi8(
                                    _mm512_loadu_si512(WORD_TO_COLUMNS.as_ptr().cast()), row);
                                _mm512_gf2p8affine_epi64_epi8::<0>(
                                    _mm512_set1_epi64(0x8040201008040201u64 as i64), permuted)
                            });
                            kernels::gfni_fold_rows(&rows, &mats[tile], old.as_mut_ptr().add(64),
                                seed_zero && tile == 0);
                        }
                        kernels::gfni_fold_word_tiles4(
                            source.as_ptr().add(group * 512), mats.as_ptr().cast(),
                            new.as_mut_ptr().add(64), STRIDE, count, seed_zero,
                            WORD_TO_COLUMNS.as_ptr(), prefetch);
                    }
                    assert_eq!(&old[64..1088], &expected,
                        "old oracle, group={group}, count={count}, first={seed_zero}");
                    assert_eq!(&new[64..1088], &expected,
                        "fold4 oracle, group={group}, count={count}, first={seed_zero}");
                    assert!(new[..64].iter().chain(new[1088..].iter()).all(|&b| b == 0x5d));
                    assert!(old[..64].iter().chain(old[1088..].iter()).all(|&b| b == 0x5d));
                } }
            }
        }
    }

    fn selected_word_cache_fixture(tiles: usize) -> RootTileCache {
        let blocks = 176;
        let mut storage = crate::scratch::take_f128(tiles * blocks * 32 + 3);
        storage.fill(F128::ZERO);
        let offset = storage.as_ptr().align_offset(64);
        let mut state = 0x953a81c7u64;
        for tile in 0..tiles {
            for block in 0..blocks {
                for stripe in 0..8 {
                    for lane in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let mut bits = state;
                        if block == 4 { bits |= 1; }
                        if block + 1 == blocks { bits &= (1u64 << 49) - 1; }
                        unsafe {
                            let dst = storage.as_mut_ptr().add(offset).cast::<u8>()
                                .add((tile * blocks + block) * 512 + stripe * 64).cast::<u32>();
                            dst.add(lane).write(bits as u32);
                            dst.add(8 + lane).write((bits >> 32) as u32);
                        }
                    }
                }
            }
        }
        unsafe { RootTileCache::from_witness_storage(storage, tiles * 64) }
    }

    #[test]
    fn selected_word_fold4_cache_ranges_and_worker_reduction() {
        let weights = |i: usize| core::array::from_fn(|lane| {
            let n = (i + lane + 997) as u64;
            F128::new(n.rotate_left(17).wrapping_mul(0x9e3779b97f4a7c15),
                n.wrapping_mul(0xbf58476d1ce4e5b9))
        });
        for threads in [9, 7] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                // Includes fewer tiles than workers, full four-tile ranges,
                // and ranges requiring both a full batch and a 1..3 tile tail.
                for tiles in [1, 3, 4, 5, 17, 29, 33, 43] {
                    let cache = selected_word_cache_fixture(tiles);
                    let expected = cache.fold_with_mode(&weights, false);
                    let actual = cache.fold_with_mode(&weights, true);
                    assert_eq!(actual, expected, "tiles={tiles}, threads={threads}");
                    let (a, b) = rayon::join(
                        || cache.fold_with_mode(&weights, false),
                        || cache.fold_with_mode(&weights, true));
                    assert_eq!(a, expected);
                    assert_eq!(b, expected);
                }
            });
        }
    }

    // Full-reader premise: ignored because this deliberately holds ~352 MiB
    // of cache data. Run optimized with the same target features as production.
    // Every timed fold includes all matrix preparation, scheduling and reduction.
    #[test]
    #[ignore = "large complete-reader premise; run explicitly in release mode"]
    fn selected_word_fold4_full_reader_premise() {
        use std::time::Instant;
        let cache = selected_word_cache_fixture(4096);
        let weights = |i: usize| core::array::from_fn(|lane| {
            let n = (i + lane + 31) as u64;
            F128::new(n.wrapping_mul(0x9e3779b97f4a7c15),
                n.rotate_left(29).wrapping_mul(0xbf58476d1ce4e5b9))
        });
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        pool.install(|| {
            // First executions warm both paths and populate scratch pools;
            // they are intentionally excluded from the reported samples.
            let expected = cache.fold_with_mode(&weights, false);
            assert_eq!(cache.fold_with_mode(&weights, true), expected);
            let mut samples = [Vec::new(), Vec::new()];
            for _ in 0..4 {
                for batch in [false, true, true, false] {
                    let start = Instant::now();
                    let output = cache.fold_with_mode(&weights, batch);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(output, expected, "complete-reader batch={batch}");
                    samples[usize::from(batch)].push(elapsed);
                }
            }
            for times in &mut samples { times.sort_by(f64::total_cmp); }
            let median = |v: &Vec<f64>| (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5;
            let old_ms = median(&samples[0]);
            let new_ms = median(&samples[1]);
            eprintln!("selected-word complete-reader ABBA: old={old_ms:.3}ms \
                batch4={new_ms:.3}ms improvement={:.2}% old_samples={:?} batch4_samples={:?}",
                100. * (old_ms - new_ms) / old_ms, samples[0], samples[1]);
        });
    }

    #[test]
    fn compact_fixed_roots_reweight_arbitrary_basis_and_recycle() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        pool.install(|| {
            let eq = |i: usize, salt: u64| core::array::from_fn::<_, 8, _>(|j| {
                let x = (i as u64 + j as u64 + salt).wrapping_mul(0x9e3779b97f4a7c15);
                F128::new(x.rotate_left(13) ^ (x >> 7), (x ^ (x >> 31)).wrapping_mul(0xbf58476d1ce4e5b9))
            });
            let eq1 = |i| eq(i, 31);
            let eq2 = |i| eq(i, 997);
            for pass in 0..2 {
                let tiles = 3;
                let blocks = 176;
                let mut storage = crate::scratch::take_f128(tiles * blocks * 32 + 3);
                storage.fill(F128::new(u64::MAX, u64::MAX));
                let offset = storage.as_ptr().align_offset(64);
                let mut expected1 = vec![F128::ZERO; 177 * 64];
                let mut expected2 = vec![F128::ZERO; 177 * 64];
                let mut state = 0x953a81c7u64 + pass;
                for tile in 0..tiles {
                    for block in 0..blocks {
                        for stripe in 0..8 {
                            let w1 = eq1(tile * 64 + stripe * 8);
                            let w2 = eq2(tile * 64 + stripe * 8);
                            for lane in 0..8 {
                                state ^= state << 13;
                                state ^= state >> 7;
                                state ^= state << 17;
                                let mut bits = state;
                                if block == 4 { bits |= 1; } // wire 512 is the constant pin.
                                if block + 1 == blocks { bits &= (1u64 << 49) - 1; }
                                // Selected-word encoding: eight low u32s, then eight high u32s.
                                unsafe {
                                    let dst = storage.as_mut_ptr().add(offset).cast::<u8>()
                                        .add((tile * blocks + block) * 512 + stripe * 64).cast::<u32>();
                                    dst.add(lane).write(bits as u32);
                                    dst.add(8 + lane).write((bits >> 32) as u32);
                                }
                                for bit in 0..64 {
                                    if (bits >> bit) & 1 != 0 {
                                        let compact = block * 64 + bit;
                                        let column = compact + if compact >= 833 { 64 } else { 0 };
                                        expected1[column] += w1[lane];
                                        expected2[column] += w2[lane];
                                    }
                                }
                            }
                        }
                    }
                }
                for expected in [&mut expected1, &mut expected2] {
                    let one = expected[256];
                    for column in [839, 865, 866, 868] { expected[column] = one; }
                }
                let cache = unsafe { RootTileCache::from_witness_storage(storage, tiles * 64) };
                assert!(cache.matches_geometry(tiles * 64, 11313));
                assert!(!cache.matches_geometry(tiles * 64, 11249));
                for batch in [false, true] {
                    assert_eq!(cache.fold_with_mode(&eq1, batch), expected1,
                        "pass={pass}, first basis, batch={batch}");
                    assert_eq!(cache.fold_with_mode(&eq2, batch), expected2,
                        "pass={pass}, second basis, batch={batch}");
                    let (a, b) = rayon::join(|| cache.fold_with_mode(&eq1, batch),
                        || cache.fold_with_mode(&eq2, batch));
                    assert_eq!(a, expected1);
                    assert_eq!(b, expected2);
                }
                // Drop returns deliberately nonzero live bytes to the scratch pool.
                drop(cache);
            }
        });
    }


    #[test]
    fn root_tile_cache_reweights_and_masks_tail() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| {
            let roots: Vec<usize> = (0..64)
                .chain((0..64).map(|i| 64 + i + if i >= 17 { 64 } else { 0 }))
                .chain(192..256)
                .chain(256..305)
                .collect();
            let plan = RootSplicePlan {
                actions: vec![
                    RootSpliceAction::Full(0),
                    RootSpliceAction::Start,
                    RootSpliceAction::End {
                        out: 1,
                        low_mask: (1 << 17) - 1,
                    },
                    RootSpliceAction::Full(2),
                    RootSpliceAction::Tail {
                        out: 3,
                        mask: (1 << 49) - 1,
                    },
                    RootSpliceAction::Skip,
                ]
                .into_boxed_slice(),
                source_bits: 305,
                root_bits: 241,
                capture: None,
            };
            plan.check_columns(&roots);
            let mut seed = 0x6182ab49u64;
            let mut rng = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let z: Vec<F128> = (0..9 * 64 * 3)
                .map(|_| F128 {
                    lo: rng(),
                    hi: rng(),
                })
                .collect();
            let eq1 = |i: usize| {
                core::array::from_fn(|j| F128 {
                    lo: (i + j + 1) as u64 * 0x21931,
                    hi: ((i + j + 5) as u64).rotate_left(29),
                })
            };
            let eq2 = |i: usize| {
                core::array::from_fn(|j| F128 {
                    lo: ((i + j + 13) as u64).rotate_left(37),
                    hi: (i + j + 1) as u64 * 0x821117,
                })
            };
            let reference1 = plan.fold(&z, 384, &eq1);
            let reference2 = plan.fold(&z, 384, &eq2);
            assert_ne!(reference1, reference2);
            let (cache, produced) = prepare(&plan, &z, 384, &eq1, false);
            assert_eq!(produced, reference1);
            assert_eq!(cache.fold(&eq1), reference1);
            assert_eq!(cache.fold(&eq2), reference2);
            let (a, b) = rayon::join(|| cache.fold(&eq1), || cache.fold(&eq2));
            assert_eq!(a, reference1);
            assert_eq!(b, reference2);
            let (raw_cache, raw_first) = prepare(&plan, &z, 384, &eq1, true);
            assert_eq!(raw_first, reference1);
            assert_eq!(raw_cache.fold(&eq1), reference1);
            assert_eq!(raw_cache.fold(&eq2), reference2);
            assert_eq!(raw_cache.fold_with_mode(&eq1, true), reference1);
            assert_eq!(raw_cache.fold_with_mode(&eq2, true), reference2);
            for tile in 0..cache.tiles {
                let tail = &cache.block(tile * cache.blocks + 3).0;
                for row in 0..8 {
                    assert!(tail[row * 64 + 49..row * 64 + 64].iter().all(|&v| v == 0));
                }
            }
            assert_eq!(cache.block(0) as *const Block as usize % 64, 0);
            drop(cache);
            let mut fresh_z = z.clone();
            fresh_z[0].lo ^= 1;
            let fresh_reference = plan.fold(&fresh_z, 384, &eq1);
            assert_ne!(fresh_reference, reference1);
            let (fresh_cache, fresh_first) = prepare(&plan, &fresh_z, 384, &eq1, false);
            assert_eq!(fresh_cache.block(0) as *const Block as usize % 64, 0);
            assert_eq!(fresh_first, fresh_reference);
            assert_eq!(fresh_cache.fold(&eq1), fresh_reference);
            assert_eq!(fresh_cache.fold(&eq2), plan.fold(&fresh_z, 384, &eq2));
        });
    }
}


#[cfg(test)]
mod indexed_weight_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; *state
    }
    #[test]
    fn indexed_weights_and_full_root_consumer_match_scalar_oracle() {
        use super::super::{eq8_from_factors_indices, eq8_from_factors};
        // Independent forward construction from original q/k/g labels.
        let mut labels = vec![usize::MAX; 1 << 18];
        for g in 0..1024 {
            for k in 0..64 {
                for q in 0..4 {
                    let p = g*256 + (k/2)*8 + q*2 + k%2;
                    assert_eq!(labels[p], usize::MAX);
                    labels[p] = q*65536 + k*1024 + g;
                }
            }
        }
        for p in (0..labels.len()).step_by(8) {
            assert_eq!(RootTileCache::cross_quarter_indices(p).as_slice(), &labels[p..p+8]);
        }
        let mut state = 0x4255_9931_3dc8_a64bu64;
        let mut weight_cases = 0;
        let mut reader_equalities = 0;
        let mut mutants = 0;
        for basis in 0..2 {
            let lo: Vec<_> = (0..512).map(|_| F128::new(next(&mut state), next(&mut state))).collect();
            let hi: Vec<_> = (0..512).map(|_| F128::new(next(&mut state), next(&mut state))).collect();
            let scalar = |i: usize| lo[i % 512] * hi[i / 512];
            for group in [0usize, 1, 2, 7, 8, 127, 511, 512, 1023] {
                for kp in [0usize, 1, 7, 15, 30, 31] {
                    let indices = RootTileCache::cross_quarter_indices(group*256+kp*8);
                    let expected = indices.map(&scalar);
                    assert_eq!(eq8_from_factors_indices(&lo, &hi, indices, 9, 511), expected);
                    assert_eq!(indices.map(|c| eq8_from_factors(&lo, &hi, c & !7, 9, 511)[c & 7]), expected);
                    weight_cases += 1;
                }
            }
            let tiles = 32;
            let blocks = 176;
            let mut storage = crate::scratch::take_f128(tiles*blocks*32+3);
            storage.fill(F128::new(u64::MAX,u64::MAX));
            let offset = storage.as_ptr().align_offset(64);
            let weights: Vec<_> = labels[..tiles*64].iter().map(|&i| scalar(i)).collect();
            let mut expected = vec![F128::ZERO;177*64];
            for tile in 0..tiles {
                for block in 0..blocks {
                    for stripe in 0..8 {
                        for lane in 0..8 {
                            let n = next(&mut state);
                            let mut bits = (1u64<<(n%64)) | (1u64<<(n.rotate_left(21)%64)) | (1u64<<(n.rotate_left(43)%64));
                            if block==4 { bits |= 1; }
                            if block==175 { bits &= (1u64<<49)-1; }
                            unsafe {
                                let dst=storage.as_mut_ptr().add(offset).cast::<u8>().add((tile*blocks+block)*512+stripe*64).cast::<u32>();
                                dst.add(lane).write(bits as u32);dst.add(8+lane).write((bits>>32)as u32);
                            }
                            let mut remaining=bits;
                            while remaining!=0 {
                                let bit=remaining.trailing_zeros()as usize;remaining&=remaining-1;
                                let compact=block*64+bit;
                                let column=compact+usize::from(compact>=833)*64;
                                expected[column]+=weights[tile*64+stripe*8+lane];
                            }
                        }
                    }
                }
            }
            for column in [839,865,866,868] { expected[column]=expected[256]; }
            // A bounded32-tile fixture exercises the complete real reader;
            // the full262144-label permutation was independently checked above.
            let mut cache=unsafe { RootTileCache::from_witness_storage(storage,tiles*64) };
            cache.cross_quarter=true;
            for threads in [7,9] {
                let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                for batch in [false,true] {
                    let old_calls=AtomicUsize::new(0);
                    let indexed_calls=AtomicUsize::new(0);
                    let original=|i| {old_calls.fetch_add(1,Ordering::Relaxed);eq8_from_factors(&lo,&hi,i,9,511)};
                    let indexed=|indices| {indexed_calls.fetch_add(1,Ordering::Relaxed);eq8_from_factors_indices(&lo,&hi,indices,9,511)};
                    let old=pool.install(||cache.fold_with_mode(&original,batch));
                    assert_eq!(old,expected,"old basis={basis} threads={threads} batch={batch}");
                    assert_eq!(old_calls.swap(0,Ordering::Relaxed),tiles*64);
                    let new=pool.install(||cache.fold_indexed_with_mode(&original,&indexed,batch));
                    assert_eq!(new,expected,"indexed basis={basis} threads={threads} batch={batch}");
                    assert_eq!(old_calls.load(Ordering::Relaxed),0);
                    assert_eq!(indexed_calls.load(Ordering::Relaxed),tiles*8);
                    reader_equalities+=2;
                    let wrong=|mut indices:[usize;8]| {indices[0]^=1024;eq8_from_factors_indices(&lo,&hi,indices,9,511)};
                    assert_ne!(pool.install(||cache.fold_indexed_with_mode(&original,&wrong,batch)),expected);
                    mutants+=1;
                }
            }
        }
        assert_eq!(weight_cases,108);assert_eq!(reader_equalities,16);assert_eq!(mutants,8);
        println!("INDEXED_ROOT_WEIGHTS labels=262144 arithmetic_octa_cases={weight_cases} full_reader_equalities={reader_equalities} wrong_weight_mutants_detected={mutants} original_expansions_per_reader=2048 indexed_expansions_per_reader=256 threads=7,9 reader_modes=2 bases=2 actual_production_consumer=true x4_enabled={}",super::super::lc_eq8_x4_enabled());
    }
}
