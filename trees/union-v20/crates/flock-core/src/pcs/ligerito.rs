

use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::build_eq_table;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::additive_ntt_f128::AdditiveNttF128;
use serde::{Deserialize, Serialize};

mod sparse_window;

pub(crate) fn open_timing() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_OPEN_TIMING").is_some());
    *ON
}

pub(crate) fn open_fill_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_FILL").is_none());
    *ON
}


#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LigeritoProfile {
    #[default]
    Fast,
    Slim,
    Secure,
}

/// Lane sl1: hold window for grind prewarm calls (`FLOCK_SL1_PREWARM_SCALE`
/// percent, default 100).
pub(crate) fn sl1_hold_us(base: u64) -> u64 {
    static S: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_SL1_PREWARM_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(100)
    });
    base * *S / 100
}

impl LigeritoProfile {
    pub fn log_inv_rate(self) -> usize {
        match self {
            Self::Fast | Self::Secure => 1,
            Self::Slim => 2,
        }
    }
    pub fn security_bits(self) -> usize {
        match self {
            Self::Fast | Self::Slim => 100,
            Self::Secure => 120,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Slim => "slim",
            Self::Secure => "secure",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "slim" => Some(Self::Slim),
            "secure" => Some(Self::Secure),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProverConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,
    pub queries: Vec<usize>,
    pub grinding_bits: Vec<usize>,
    pub fold_grinding_bits: Vec<usize>,
    pub ood_samples: Vec<usize>,
    pub merkle_hash: HashKind,
}

#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub log_inv_rates: Vec<usize>,
    pub recursive_steps: usize,
    pub initial_log_msg_cols: usize,
    pub initial_log_num_interleaved: usize,
    pub initial_k: usize,
    pub recursive_log_msg_cols: Vec<usize>,
    pub recursive_ks: Vec<usize>,
    pub queries: Vec<usize>,
    pub grinding_bits: Vec<usize>,
    pub fold_grinding_bits: Vec<usize>,
    pub ood_samples: Vec<usize>,
    pub merkle_hash: HashKind,
}

pub const UDR_PROXIMITY_LOSS: f64 = 0.0;

const UDR_TARGET_BITS: f64 = 100.0;

pub fn udr_queries(log_inv_rate: usize) -> usize {
    assert!(log_inv_rate > 0, "log_inv_rate=0 (rate 1) has no soundness");
    let per_q = udr_per_query_bits_asymptotic(log_inv_rate);
    (UDR_TARGET_BITS / per_q).ceil() as usize
}

pub fn default_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<ProverConfig, &'static str> {
    let initial_k = log_batch_size;
    if log_n <= initial_k {
        return Err("log_n must be > initial_k");
    }

    let mut log_inv_rates = vec![log_inv_rate];
    let mut recursive_ks = Vec::new();
    let mut recursive_log_msg_cols = Vec::new();

    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;

    {
        let block_len_log = n_running + rate_running;
        let qs = udr_queries(rate_running);
        if (1usize << block_len_log) < qs {
            return Err("L0 block_len < udr_queries — log_n too small for chosen rate");
        }
    }

    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;
        let mut next_rate = rate_running + 1;
        loop {
            let bl = 1usize << (log_msg_cols_next + next_rate);
            let qs = udr_queries(next_rate);
            if bl >= qs {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)");
            }
        }
        recursive_log_msg_cols.push(log_msg_cols_next);
        recursive_ks.push(k);
        log_inv_rates.push(next_rate);
        n_running -= k;
        rate_running = next_rate;
    }

    if recursive_ks.is_empty() {
        return Err("log_n too small — no recursive levels for the Ligerito recursion");
    }

    let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
    let n_levels = log_inv_rates.len();
    let grinding_bits = vec![0usize; n_levels];

    Ok(ProverConfig {
        log_inv_rates: log_inv_rates.clone(),
        recursive_steps: recursive_ks.len(),
        initial_log_msg_cols: log_n - initial_k,
        initial_log_num_interleaved: initial_k,
        initial_k,
        recursive_log_msg_cols,
        recursive_ks,
        queries,
        grinding_bits,
        fold_grinding_bits: vec![0usize; n_levels],
        ood_samples: vec![0usize; n_levels],
        merkle_hash: HashKind::default(),
    })
}

struct LadderShape {
    log_inv_rates: Vec<usize>,
    log_msg_cols: Vec<usize>,
    log_num_interleaved: Vec<usize>,
    k_recursive: Vec<usize>,
    yr_log_n: usize,
}

fn derive_ladder_shape(
    log_n: usize,
    initial_k: usize,
    log_inv_rate: usize,
    queries_at_rate: &dyn Fn(usize) -> usize,
) -> Result<LadderShape, String> {
    if log_n <= initial_k {
        return Err("log_n must be > initial_k".into());
    }
    let mut shape = LadderShape {
        log_inv_rates: vec![log_inv_rate],
        log_msg_cols: vec![log_n - initial_k],
        log_num_interleaved: vec![initial_k],
        k_recursive: vec![initial_k],
        yr_log_n: 0,
    };
    let mut n_running = log_n - initial_k;
    let mut rate_running = log_inv_rate;
    if (1usize << (n_running + rate_running)) < queries_at_rate(rate_running) {
        return Err("L0 block_len < queries — log_n too small for chosen rate".into());
    }
    while n_running > 5 {
        let k = 3.min(n_running);
        let log_msg_cols_next = n_running - k;
        let mut next_rate = rate_running + 1;
        loop {
            if (1usize << (log_msg_cols_next + next_rate)) >= queries_at_rate(next_rate) {
                break;
            }
            next_rate += 1;
            if next_rate > 20 {
                return Err("could not find feasible recursive rate (level too deep)".into());
            }
        }
        shape.log_inv_rates.push(next_rate);
        shape.log_msg_cols.push(log_msg_cols_next);
        shape.log_num_interleaved.push(k);
        shape.k_recursive.push(k);
        n_running -= k;
        rate_running = next_rate;
    }
    if shape.k_recursive.len() < 2 {
        return Err("log_n too small — no recursive levels for the Ligerito recursion".into());
    }
    shape.yr_log_n = n_running;
    Ok(shape)
}

macro_rules! profile_configs {
    ($($m:literal),+ $(,)?) => {
        &[
            $(
                (($m, LigeritoProfile::Fast),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_fast.toml"))),
                (($m, LigeritoProfile::Slim),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_slim.toml"))),
                (($m, LigeritoProfile::Secure),
                 include_str!(concat!("../../configs/ligerito/m", $m, "_secure.toml"))),
            )+
        ]
    };
}
const EMBEDDED_CONFIGS: &[((usize, LigeritoProfile), &str)] =
    profile_configs!(22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35);

pub fn embedded_security_config(m: usize, profile: LigeritoProfile) -> Option<&'static str> {
    EMBEDDED_CONFIGS.iter().find_map(|&(key, toml)| {
        if key == (m, profile) {
            Some(toml)
        } else {
            None
        }
    })
}

#[allow(clippy::type_complexity)]
static PARSED_EMBEDDED_CONFIGS: std::sync::LazyLock<
    Vec<(
        (usize, LigeritoProfile),
        Result<(usize, ProverConfig, VerifierConfig), String>,
    )>,
> = std::sync::LazyLock::new(|| {
    EMBEDDED_CONFIGS
        .iter()
        .map(|&(key, toml)| {
            let parsed = LigeritoSecurityConfig::from_toml_str(toml).and_then(|sec| {
                let initial_k = sec.initial_k;
                sec.to_prover_verifier_configs()
                    .map(|(pv, vc)| (initial_k, pv, vc))
            });
            (key, parsed)
        })
        .collect()
});

fn parsed_config_for(
    m: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Option<Result<(ProverConfig, VerifierConfig), String>> {
    let (_, parsed) = PARSED_EMBEDDED_CONFIGS
        .iter()
        .find(|&&(key, _)| key == (m, profile))?;
    Some(match parsed {
        Err(e) => Err(e.clone()),
        Ok((initial_k, pv, vc)) => {
            if *initial_k != log_batch_size {
                Err(format!(
                    "embedded config for (m={m}, profile={}) has \
                     initial_k={initial_k} but caller requested log_batch_size={log_batch_size}",
                    profile.as_str()
                ))
            } else {
                Ok((pv.clone(), vc.clone()))
            }
        }
    })
}

pub fn prover_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<ProverConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let (pv, _) = parsed_config_for(m, log_batch_size, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={}). \
             Add a TOML at configs/ligerito/m{m}_{}.toml and register it in \
             EMBEDDED_CONFIGS, or call default_config explicitly for ad-hoc shapes.",
            profile.as_str(),
            profile.as_str(),
        )
    })??;
    Ok(pv)
}

pub fn verifier_config_for(
    log_n: usize,
    log_batch_size: usize,
    profile: LigeritoProfile,
) -> Result<VerifierConfig, String> {
    let m = log_n + crate::pcs::LOG_PACKING;
    let (_, vc) = parsed_config_for(m, log_batch_size, profile).ok_or_else(|| {
        format!(
            "no security config registered for (m={m}, profile={})",
            profile.as_str()
        )
    })??;
    Ok(vc)
}

pub fn default_verifier_config(
    log_n: usize,
    log_batch_size: usize,
    log_inv_rate: usize,
) -> Result<VerifierConfig, &'static str> {
    let p = default_config(log_n, log_batch_size, log_inv_rate)?;
    Ok(VerifierConfig {
        log_inv_rates: p.log_inv_rates,
        recursive_steps: p.recursive_steps,
        initial_log_msg_cols: p.initial_log_msg_cols,
        initial_log_num_interleaved: p.initial_log_num_interleaved,
        initial_k: p.initial_k,
        recursive_log_msg_cols: p.recursive_log_msg_cols,
        recursive_ks: p.recursive_ks,
        queries: p.queries,
        grinding_bits: p.grinding_bits,
        fold_grinding_bits: p.fold_grinding_bits,
        ood_samples: p.ood_samples,
        merkle_hash: p.merkle_hash,
    })
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoundnessRegime {
    Udr,
    JohnsonOod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrindingStep {
    PostCommitPreQueries,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoLevelConfig {
    pub log_inv_rate: usize,
    pub log_msg_cols: usize,
    pub log_num_interleaved: usize,
    pub k_recursive: usize,
    pub regime: SoundnessRegime,
    pub eta: Option<f64>,
    #[serde(default)]
    pub proximity_loss: Option<f64>,
    pub queries: usize,
    pub grinding_bits: usize,
    #[serde(default)]
    pub fold_grinding_bits: usize,
    #[serde(default)]
    pub ood_samples: usize,
    pub target_security_bits: usize,
    pub expected_eps_pg_bits: f64,
    pub expected_eps_query_bits: f64,
    #[serde(default)]
    pub expected_eps_ood_bits: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalBlockConfig {
    pub yr_log_n: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LigeritoSecurityConfig {
    pub m: usize,
    pub log_n: usize,
    pub initial_k: usize,
    pub target_security_bits: usize,
    pub analysis_version: String,
    pub field: String,
    pub hash: String,
    pub grinding_step: GrindingStep,
    pub levels: Vec<LigeritoLevelConfig>,
    pub final_block: FinalBlockConfig,
}

const ANALYSIS_LOG_Q: f64 = 128.0;

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

const PAPER_COMPAT_TOL_BITS: f64 = 0.6;

fn paper_thm_ca_johnson_log_a(log_inv_rate: usize, eta: f64, log_msg_cols: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let gamma = 1.0 - sqrt_rho - eta;
    let m_param = ((sqrt_rho / (2.0 * eta)).ceil() as usize).max(3) as f64;
    let half = m_param + 0.5;
    let half5 = half.powi(5);
    let numerator = 2.0 * half5 + 3.0 * half * gamma * rho;
    let denominator = 3.0 * rho.powf(1.5);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    let a = (numerator / denominator) * n + half / sqrt_rho;
    a.log2()
}

fn paper_johnson_log_a(
    log_inv_rate: usize,
    eta: f64,
    log_msg_cols: usize,
    log_num_interleaved: usize,
) -> f64 {
    let base = paper_thm_ca_johnson_log_a(log_inv_rate, eta, log_msg_cols);
    let row_union_penalty = (log_num_interleaved as f64 - 1.0).max(0.0);
    base + row_union_penalty
}

fn paper_per_query_bits(log_inv_rate: usize, eta: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = 1.0 - rho.sqrt() - eta;
    (1.0 / (1.0 - gamma)).log2()
}

fn udr_gamma(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let delta = 1.0 - rho;
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    delta / 2.0 - 3.0 / (delta * n) - proximity_loss
}

fn udr_per_query_bits(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    (1.0 / (1.0 - gamma)).log2()
}

fn udr_per_query_bits_asymptotic(log_inv_rate: usize) -> f64 {
    let rho = (-(log_inv_rate as f64)).exp2();
    let gamma = (1.0 - rho) / 2.0;
    (1.0 / (1.0 - gamma)).log2()
}

fn paper_thm_1_4_log_a(log_inv_rate: usize, log_msg_cols: usize, proximity_loss: f64) -> f64 {
    let gamma = udr_gamma(log_inv_rate, log_msg_cols, proximity_loss);
    let n = ((log_msg_cols + log_inv_rate) as f64).exp2();
    (gamma * n + 1.0).log2()
}

fn johnson_interleaved_list_log2(log_inv_rate: usize, eta: f64) -> f64 {
    debug_assert!(
        eta > 0.0,
        "η must be > 0 to stay strictly below the Johnson radius"
    );
    let rho = (-(log_inv_rate as f64)).exp2();
    let sqrt_rho = rho.sqrt();
    let l_base = 1.0 / (2.0 * eta * sqrt_rho);
    l_base.log2()
}

fn paper_ood_bits(log_inv_rate: usize, eta: f64, mu_vars: usize, ood_samples: usize) -> f64 {
    let log2_l = johnson_interleaved_list_log2(log_inv_rate, eta);
    let log2_mu = (mu_vars as f64).log2();
    if ood_samples == 0 {
        ANALYSIS_LOG_Q - log2_l - log2_mu
    } else {
        ood_samples as f64 * (ANALYSIS_LOG_Q - log2_mu) - (2.0 * log2_l - 1.0)
    }
}

impl LigeritoLevelConfig {
    pub fn paper_predicted_bits(&self) -> (f64, f64) {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                let log_a = paper_johnson_log_a(
                    self.log_inv_rate,
                    eta,
                    self.log_msg_cols,
                    self.log_num_interleaved,
                );
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                let per_q = paper_per_query_bits(self.log_inv_rate, eta);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
            SoundnessRegime::Udr => {
                let proximity_loss = self
                    .proximity_loss
                    .expect("Udr regime must carry proximity_loss");
                let log_a =
                    paper_thm_1_4_log_a(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_pg = ANALYSIS_LOG_Q - log_a;
                let per_q =
                    udr_per_query_bits(self.log_inv_rate, self.log_msg_cols, proximity_loss);
                let eps_query = self.queries as f64 * per_q;
                (eps_pg, eps_query)
            }
        }
    }

    pub fn paper_predicted_ood_bits(&self) -> Option<f64> {
        match self.regime {
            SoundnessRegime::JohnsonOod => {
                let eta = self.eta.expect("JohnsonOod must have eta");
                let mu = self.log_msg_cols + self.log_num_interleaved;
                Some(paper_ood_bits(self.log_inv_rate, eta, mu, self.ood_samples))
            }
            SoundnessRegime::Udr => None,
        }
    }
}

impl LigeritoSecurityConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.log_n + 7 != self.m {
            return Err(format!(
                "log_n ({}) + LOG_PACKING (7) != m ({})",
                self.log_n, self.m
            ));
        }

        self.merkle_hash()?;

        let levels_recursive_sum: usize = self.levels.iter().skip(1).map(|lv| lv.k_recursive).sum();
        let yr_log_n = self.final_block.yr_log_n;
        if self.initial_k + levels_recursive_sum + yr_log_n != self.log_n {
            return Err(format!(
                "shape mismatch: initial_k ({}) + Σ k_recursive ({}) + yr_log_n ({}) = {} ≠ log_n ({})",
                self.initial_k,
                levels_recursive_sum,
                yr_log_n,
                self.initial_k + levels_recursive_sum + yr_log_n,
                self.log_n,
            ));
        }

        let l0 = self
            .levels
            .first()
            .ok_or_else(|| "empty levels".to_string())?;
        if l0.k_recursive != self.initial_k {
            return Err(format!(
                "L0.k_recursive ({}) must equal initial_k ({})",
                l0.k_recursive, self.initial_k
            ));
        }
        if l0.log_num_interleaved != self.initial_k {
            return Err(format!(
                "L0.log_num_interleaved ({}) must equal initial_k ({})",
                l0.log_num_interleaved, self.initial_k
            ));
        }

        let mut dim_in = self.log_n;
        for (i, lv) in self.levels.iter().enumerate() {
            if lv.log_msg_cols + lv.log_num_interleaved != dim_in {
                return Err(format!(
                    "L{i}: log_msg_cols ({}) + log_num_interleaved ({}) ≠ input dim ({dim_in})",
                    lv.log_msg_cols, lv.log_num_interleaved
                ));
            }

            match (lv.regime, lv.eta) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but eta is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!("L{i}: regime requires eta but eta is None"));
                }
                _ => {}
            }

            match (lv.regime, lv.proximity_loss) {
                (SoundnessRegime::Udr, None) => {
                    return Err(format!("L{i}: regime=udr but proximity_loss is missing"));
                }
                (SoundnessRegime::Udr, Some(eps)) if eps < 0.0 => {
                    return Err(format!("L{i}: proximity_loss must be ≥ 0, got {eps}"));
                }
                (SoundnessRegime::JohnsonOod, Some(_)) => {
                    return Err(format!("L{i}: proximity_loss is only valid for regime=udr"));
                }
                _ => {}
            }

            match lv.regime {
                SoundnessRegime::Udr if lv.ood_samples != 0 => {
                    return Err(format!(
                        "L{i}: regime=udr but ood_samples={} (unique decoding \
                         has list size 1 — no OOD binding step exists)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i == 0 && lv.ood_samples != 0 => {
                    return Err(format!(
                        "L0: ood_samples={} but L0 is bound by the opening's \
                         own evaluation claim (must be 0)",
                        lv.ood_samples
                    ));
                }
                SoundnessRegime::JohnsonOod if i > 0 && lv.ood_samples == 0 => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires ood_samples ≥ 1 \
                         past L0 (the query counts assume single-codeword \
                         binding)"
                    ));
                }
                _ => {}
            }

            match (lv.regime, lv.expected_eps_ood_bits) {
                (SoundnessRegime::Udr, Some(_)) => {
                    return Err(format!("L{i}: regime=udr but expected_eps_ood_bits is set"));
                }
                (SoundnessRegime::JohnsonOod, None) => {
                    return Err(format!(
                        "L{i}: regime=johnson_ood requires expected_eps_ood_bits"
                    ));
                }
                (SoundnessRegime::JohnsonOod, Some(declared)) => {
                    let pred = lv
                        .paper_predicted_ood_bits()
                        .expect("JohnsonOod has an OOD prediction");
                    if (declared - pred).abs() > PAPER_COMPAT_TOL_BITS {
                        return Err(format!(
                            "L{i}: expected_eps_ood_bits ({declared:.2}) doesn't \
                             match prediction ({pred:.2}); tolerance ±{:.2} bits.",
                            PAPER_COMPAT_TOL_BITS
                        ));
                    }
                }
                _ => {}
            }

            let (pg_pred, q_pred) = lv.paper_predicted_bits();
            if (lv.expected_eps_pg_bits - pg_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits. \
                     Re-derive Q, eta, or grinding so the declared diagnostic \
                     matches the formula.",
                    lv.expected_eps_pg_bits,
                    pg_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }
            if (lv.expected_eps_query_bits - q_pred).abs() > PAPER_COMPAT_TOL_BITS {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) doesn't match \
                     {analysis} prediction ({:.2}); tolerance ±{:.2} bits.",
                    lv.expected_eps_query_bits,
                    q_pred,
                    PAPER_COMPAT_TOL_BITS,
                    analysis = self.analysis_version,
                ));
            }

            if lv.target_security_bits > lv.grinding_bits
                && lv.expected_eps_query_bits + 1e-3
                    < (lv.target_security_bits - lv.grinding_bits) as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_query_bits ({:.2}) < target ({}) - grinding ({}) = {}",
                    lv.expected_eps_query_bits,
                    lv.target_security_bits,
                    lv.grinding_bits,
                    lv.target_security_bits - lv.grinding_bits
                ));
            }

            if lv.expected_eps_pg_bits + lv.fold_grinding_bits as f64 + 1e-3
                < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_pg_bits ({:.2}) + fold_grinding ({}) < target ({})",
                    lv.expected_eps_pg_bits, lv.fold_grinding_bits, lv.target_security_bits
                ));
            }

            if let Some(ood) = lv.expected_eps_ood_bits
                && ood + 1e-3 < lv.target_security_bits as f64
            {
                return Err(format!(
                    "L{i}: expected_eps_ood_bits ({ood:.2}) < target ({}); \
                         increase ood_samples",
                    lv.target_security_bits
                ));
            }

            if lv.target_security_bits < self.target_security_bits {
                return Err(format!(
                    "L{i}: target_security_bits ({}) < global target ({})",
                    lv.target_security_bits, self.target_security_bits
                ));
            }

            dim_in -= lv.k_recursive;
        }

        if dim_in != yr_log_n {
            return Err(format!(
                "after consuming all levels, dim_in ({dim_in}) ≠ yr_log_n ({yr_log_n})"
            ));
        }

        Ok(())
    }

    pub fn derive_paper_compatible(
        m: usize,
        log_inv_rate: usize,
        target_security_bits: usize,
    ) -> Result<Self, String> {
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;
        let prover = default_config(log_n, initial_k, log_inv_rate).map_err(|e| e.to_string())?;
        let r = prover.recursive_steps;
        let mut levels = Vec::with_capacity(r + 1);
        let mut log_msg_cols_per_level = Vec::with_capacity(r + 1);
        let mut log_num_interleaved_per_level = Vec::with_capacity(r + 1);
        let mut k_recursive_per_level = Vec::with_capacity(r + 1);
        log_msg_cols_per_level.push(log_n - initial_k);
        log_num_interleaved_per_level.push(initial_k);
        k_recursive_per_level.push(initial_k);
        for i in 0..r {
            log_msg_cols_per_level.push(prover.recursive_log_msg_cols[i]);
            log_num_interleaved_per_level.push(prover.recursive_ks[i]);
            k_recursive_per_level.push(prover.recursive_ks[i]);
        }
        for i in 0..=r {
            let rate = prover.log_inv_rates[i];
            let proximity_loss = UDR_PROXIMITY_LOSS;
            let per_q = udr_per_query_bits(rate, log_msg_cols_per_level[i], proximity_loss);
            let queries = ((target_security_bits as f64) / per_q).ceil() as usize;
            let log_a = paper_thm_1_4_log_a(rate, log_msg_cols_per_level[i], proximity_loss);
            let eps_pg = ANALYSIS_LOG_Q - log_a;
            let fold_grinding_bits =
                ((target_security_bits as f64) - eps_pg).ceil().max(0.0) as usize;
            let eps_query = queries as f64 * per_q;
            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: log_msg_cols_per_level[i],
                log_num_interleaved: log_num_interleaved_per_level[i],
                k_recursive: k_recursive_per_level[i],
                regime: SoundnessRegime::Udr,
                eta: None,
                proximity_loss: Some(proximity_loss),
                queries,
                grinding_bits: 0,
                fold_grinding_bits,
                ood_samples: 0,
                target_security_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: None,
            });
        }
        let total_recursive: usize = prover.recursive_ks.iter().sum();
        let yr_log_n = log_n - initial_k - total_recursive;
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits,
            analysis_version: "no_row_union_over_ben_sasson_2025_cor_1_4".into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig { yr_log_n },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn derive_profile(m: usize, profile: LigeritoProfile) -> Result<Self, String> {
        let johnson_eta = match profile {
            LigeritoProfile::Fast => 0.04,
            LigeritoProfile::Slim => 0.02,
            LigeritoProfile::Secure => 0.0, // unused in the UDR branch
        };
        let target_bits = profile.security_bits();
        let log_inv_rate = profile.log_inv_rate();
        let query_grind: usize = match profile {
            LigeritoProfile::Slim => 16,
            LigeritoProfile::Fast | LigeritoProfile::Secure => 0,
        };
        let log_n = m
            .checked_sub(crate::pcs::LOG_PACKING)
            .ok_or_else(|| format!("m ({m}) < LOG_PACKING (7)"))?;
        let initial_k = 6usize;

        let per_query_bits_feas = |rate: usize| -> f64 {
            match profile {
                LigeritoProfile::Secure => udr_per_query_bits_asymptotic(rate),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, johnson_eta)
                }
            }
        };

        let t_feas = target_bits as f64 + 5.0;
        let queries_feas = |rate: usize| -> usize {
            ((t_feas - query_grind as f64).max(1.0) / per_query_bits_feas(rate)).ceil() as usize
        };
        let shape = derive_ladder_shape(log_n, initial_k, log_inv_rate, &queries_feas)?;
        let n_levels = shape.log_inv_rates.len();

        let t = target_bits as f64;

        let mut levels = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            let rate = shape.log_inv_rates[i];
            let cols = shape.log_msg_cols[i];
            let ilv = shape.log_num_interleaved[i];
            let per_q = match profile {
                LigeritoProfile::Secure => udr_per_query_bits(rate, cols, UDR_PROXIMITY_LOSS),
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    paper_per_query_bits(rate, johnson_eta)
                }
            };
            let queries = ((t - query_grind as f64).max(1.0) / per_q).ceil() as usize;
            if queries > (1usize << (cols + rate)) {
                return Err(format!(
                    "L{i}: {queries} queries exceed block length 2^{}",
                    cols + rate
                ));
            }
            let eps_query = queries as f64 * per_q;

            let (regime, eta, proximity_loss, eps_pg, ood_samples, eps_ood) = match profile {
                LigeritoProfile::Secure => {
                    let eps_pg =
                        ANALYSIS_LOG_Q - paper_thm_1_4_log_a(rate, cols, UDR_PROXIMITY_LOSS);
                    (
                        SoundnessRegime::Udr,
                        None,
                        Some(UDR_PROXIMITY_LOSS),
                        eps_pg,
                        0usize,
                        None,
                    )
                }
                LigeritoProfile::Fast | LigeritoProfile::Slim => {
                    let eps_pg = ANALYSIS_LOG_Q - paper_johnson_log_a(rate, johnson_eta, cols, ilv);
                    let mu = cols + ilv;
                    let ood_samples = if i == 0 {
                        0 // bound by the opening's own evaluation claim
                    } else {
                        (1..=8usize)
                            .find(|&s| paper_ood_bits(rate, johnson_eta, mu, s) >= t)
                            .ok_or_else(|| {
                                format!("L{i}: no OOD sample count reaches {t:.1} bits")
                            })?
                    };
                    let eps_ood = paper_ood_bits(rate, johnson_eta, mu, ood_samples);
                    (
                        SoundnessRegime::JohnsonOod,
                        Some(johnson_eta),
                        None,
                        eps_pg,
                        ood_samples,
                        Some(round1(eps_ood)),
                    )
                }
            };
            let fold_grinding_bits = (t - eps_pg).ceil().max(0.0) as usize;

            levels.push(LigeritoLevelConfig {
                log_inv_rate: rate,
                log_msg_cols: cols,
                log_num_interleaved: ilv,
                k_recursive: shape.k_recursive[i],
                regime,
                eta,
                proximity_loss,
                queries,
                grinding_bits: query_grind,
                fold_grinding_bits,
                ood_samples,
                target_security_bits: target_bits,
                expected_eps_pg_bits: round1(eps_pg),
                expected_eps_query_bits: round1(eps_query),
                expected_eps_ood_bits: eps_ood,
            });
        }

        let analysis_version = match profile {
            LigeritoProfile::Secure => "no_row_union_over_ben_sasson_2025_cor_1_4",
            LigeritoProfile::Fast | LigeritoProfile::Slim => {
                "johnson_ood_row_union_over_bchks25_thm_4_6"
            }
        };
        let cfg = Self {
            m,
            log_n,
            initial_k,
            target_security_bits: target_bits,
            analysis_version: analysis_version.into(),
            field: "f128".into(),
            hash: "sha256".into(),
            grinding_step: GrindingStep::PostCommitPreQueries,
            levels,
            final_block: FinalBlockConfig {
                yr_log_n: shape.yr_log_n,
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(s).map_err(|e| format!("toml parse: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("toml serialize: {e}"))
    }

    pub fn to_prover_verifier_configs(&self) -> Result<(ProverConfig, VerifierConfig), String> {
        self.validate()?;
        let merkle_hash = self.merkle_hash()?;
        let log_inv_rates: Vec<usize> = self.levels.iter().map(|lv| lv.log_inv_rate).collect();
        let recursive_ks: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.k_recursive)
            .collect();
        let recursive_log_msg_cols: Vec<usize> = self
            .levels
            .iter()
            .skip(1)
            .map(|lv| lv.log_msg_cols)
            .collect();
        let queries: Vec<usize> = self.levels.iter().map(|lv| lv.queries).collect();
        let grinding_bits: Vec<usize> = self.levels.iter().map(|lv| lv.grinding_bits).collect();
        let fold_grinding_bits: Vec<usize> =
            self.levels.iter().map(|lv| lv.fold_grinding_bits).collect();
        let ood_samples: Vec<usize> = self.levels.iter().map(|lv| lv.ood_samples).collect();
        let prover = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
            merkle_hash,
        };
        let verifier = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: recursive_ks.len(),
            initial_log_msg_cols: self.levels[0].log_msg_cols,
            initial_log_num_interleaved: self.initial_k,
            initial_k: self.initial_k,
            recursive_log_msg_cols,
            recursive_ks,
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
            merkle_hash,
        };
        Ok((prover, verifier))
    }

    pub fn merkle_hash(&self) -> Result<HashKind, String> {
        HashKind::parse(&self.hash).map_err(|e| format!("security config `hash`: {e}"))
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecursiveProof {
    pub opened_rows: Vec<Vec<F128>>,
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalProof {
    pub yr: Vec<F128>,
    pub opened_rows: Vec<Vec<F128>>,
    pub merkle_proof: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LigeritoProof {
    pub initial_root: Hash,
    pub initial_proof: RecursiveProof,
    pub recursive_roots: Vec<Hash>,
    pub recursive_proofs: Vec<RecursiveProof>,
    pub final_proof: FinalProof,
    pub sumcheck_transcript: Vec<SumcheckMessage>,
    #[serde(default)]
    pub grinding_nonces: Vec<u64>,
    #[serde(default)]
    pub ood_values: Vec<F128>,
    #[serde(default)]
    pub fold_grinding_nonces: Vec<u64>,
}

impl LigeritoProof {
    pub fn size_bytes(&self) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        let level_bytes = |p: &RecursiveProof| -> usize {
            p.opened_rows.iter().map(|r| r.len() * ELEM).sum::<usize>() + p.merkle_proof.len() * 32
        };
        let mut total = 32;
        total += self.recursive_roots.len() * 32;
        total += level_bytes(&self.initial_proof);
        for p in &self.recursive_proofs {
            total += level_bytes(p);
        }
        total += self.final_proof.yr.len() * ELEM
            + self
                .final_proof
                .opened_rows
                .iter()
                .map(|r| r.len() * ELEM)
                .sum::<usize>()
            + self.final_proof.merkle_proof.len() * 32;
        total += self.sumcheck_transcript.len() * 2 * ELEM;
        total += self.ood_values.len() * ELEM;
        total += (self.grinding_nonces.len() + self.fold_grinding_nonces.len()) * 8;
        total
    }

    pub fn print_size_breakdown(&self) {
        const ELEM: usize = core::mem::size_of::<F128>();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        let roots_b = 32 * (1 + self.recursive_roots.len());
        let init_opened: usize = self
            .initial_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let init_merkle: usize = self.initial_proof.merkle_proof.len() * 32;
        eprintln!(
            "  L0 (initial): opened={} ({}q × {}lanes × {}B)  merkle={}",
            kb(init_opened),
            self.initial_proof.opened_rows.len(),
            self.initial_proof
                .opened_rows
                .first()
                .map_or(0, |r| r.len()),
            ELEM,
            kb(init_merkle),
        );
        let mut total_opened = init_opened;
        let mut total_merkle = init_merkle;
        for (i, rp) in self.recursive_proofs.iter().enumerate() {
            let opened: usize = rp.opened_rows.iter().map(|r| r.len() * ELEM).sum();
            let merkle: usize = rp.merkle_proof.len() * 32;
            eprintln!(
                "  L{} (recursive): opened={} ({}q × {}lanes × {}B)  merkle={}",
                i + 1,
                kb(opened),
                rp.opened_rows.len(),
                rp.opened_rows.first().map_or(0, |r| r.len()),
                ELEM,
                kb(merkle),
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let final_opened: usize = self
            .final_proof
            .opened_rows
            .iter()
            .map(|r| r.len() * ELEM)
            .sum();
        let final_merkle: usize = self.final_proof.merkle_proof.len() * 32;
        let yr_b = self.final_proof.yr.len() * ELEM;
        eprintln!(
            "  L{} (final):  opened={} ({}q × {}lanes × {}B)  merkle={}  yr={} ({}×{}B)",
            self.recursive_proofs.len() + 1,
            kb(final_opened),
            self.final_proof.opened_rows.len(),
            self.final_proof.opened_rows.first().map_or(0, |r| r.len()),
            ELEM,
            kb(final_merkle),
            kb(yr_b),
            self.final_proof.yr.len(),
            ELEM,
        );
        total_opened += final_opened;
        total_merkle += final_merkle;
        let tx_b = self.sumcheck_transcript.len() * 2 * ELEM;
        eprintln!(
            "  TOTALS: roots={}  opened={}  merkle={}  yr={}  transcript={} ({}×2×{}B)  GRAND={}",
            kb(roots_b),
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(tx_b),
            self.sumcheck_transcript.len(),
            ELEM,
            kb(self.size_bytes()),
        );
    }
}


pub(crate) fn partial_eval_lsb(evals: &[F128], rs: &[F128]) -> Vec<F128> {
    let mut cur = evals.to_vec();
    for &r in rs {
        let half = cur.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            let e0 = cur[2 * i];
            let e1 = cur[2 * i + 1];
            next.push(e0 + r * (e0 + e1));
        }
        cur = next;
    }
    cur
}

#[cfg(test)]
pub(crate) fn eval_mle_lsb(evals: &[F128], point: &[F128]) -> F128 {
    let folded = partial_eval_lsb(evals, point);
    debug_assert_eq!(folded.len(), 1);
    folded[0]
}


#[inline]
fn next_s(s: F128, s_at_root: F128) -> F128 {
    s * s + s_at_root * s
}

pub(crate) fn eval_sk_at_vks(log_n: usize) -> Vec<F128> {
    let mut sks_vks = vec![F128::ZERO; log_n + 1];
    sks_vks[0] = F128::ONE;
    if log_n == 0 {
        return sks_vks;
    }
    let mut layer: Vec<F128> = (1..=log_n).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut cur_len = log_n;
    for i in 0..log_n {
        for j in 0..cur_len {
            let sk_at_vk = next_s(layer[j], sks_vks[i]);
            if j == 0 {
                sks_vks[i + 1] = sk_at_vk;
            } else {
                layer[j - 1] = sk_at_vk;
            }
        }
        cur_len -= 1;
    }
    sks_vks
}

/// Serial Montgomery batch inversion for the short deterministic denominator
/// vectors used by Ligerito basis induction (one field inversion plus three
/// products per element instead of one inversion per element). A zero input
/// stays zero, matching the element-wise
/// `if v.is_zero() { ZERO } else { v.inv() }` it replaces; every nonzero
/// output is the identical canonical inverse at the same index. Ported from
/// the Mac track (jacklightChen a0163b69, flock-challenge).
/// `FLOCK_NO_LIG_BATCH_INV=1` restores the element-wise form.
/// Mechanism from rival submission 9fc8de41-ba12-4559-8c56-2b88f91652ca (ported here).
fn batch_inverse_or_zero(values: &[F128]) -> Vec<F128> {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_BATCH_INV").is_none());
    if values.len() < 2 || !*ON {
        return values
            .iter()
            .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
            .collect();
    }

    let mut inverses = vec![F128::ZERO; values.len()];
    let Some(first_nonzero) = values.iter().position(|v| !v.is_zero()) else {
        return inverses;
    };
    let mut product = values[first_nonzero];
    for index in first_nonzero + 1..values.len() {
        let value = values[index];
        if value.is_zero() {
            continue;
        }
        inverses[index] = product;
        product *= value;
    }
    let mut suffix_inverse = product.inv();
    for index in (first_nonzero + 1..values.len()).rev() {
        let value = values[index];
        if value.is_zero() {
            continue;
        }
        inverses[index] *= suffix_inverse;
        suffix_inverse *= value;
    }
    inverses[first_nonzero] = suffix_inverse;
    inverses
}

fn evaluate_scaled_basis_inplace(
    sks_at_x: &mut [F128],
    basis: &mut [F128],
    sks_vks: &[F128],
    inv_sks_vks: &[F128],
    x: F128,
    alpha: F128,
) {
    let log_n = basis.len().trailing_zeros() as usize;
    debug_assert_eq!(basis.len(), 1 << log_n);
    debug_assert!(sks_at_x.len() >= log_n);
    debug_assert!(inv_sks_vks.len() > log_n);

    if log_n > 0 {
        sks_at_x[0] = x;
        for i in 1..log_n {
            sks_at_x[i] = next_s(sks_at_x[i - 1], sks_vks[i - 1]);
        }
        for i in 0..log_n {
            sks_at_x[i] *= inv_sks_vks[i];
        }
    }

    basis[0] = alpha;
    let wide = open_basis_x4_enabled();
    for k in 0..log_n {
        let s_at_x = sks_at_x[k];
        let current_len = 1 << k;
        if wide {
            let (current, rest) = basis[..2 * current_len].split_at_mut(current_len);
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            unsafe {
                eq_expand_block_x4(rest, current, s_at_x);
            }
            #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
            for (dst, &src) in rest.iter_mut().zip(current.iter()) {
                *dst = s_at_x * src;
            }
        } else {
            for i in 0..current_len {
                basis[i + current_len] = s_at_x * basis[i];
            }
        }
    }
}

fn open_basis_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_BASIS_X4").is_none());
    *ON
}


pub(crate) fn induce_sumcheck_enforced_sum(
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> F128 {
    assert_eq!(opened_rows.len(), queries.len());
    let eq = build_eq_table(v_challenges);
    let n_queries = queries.len();
    let alpha_weights: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        build_eq_table(alpha).into_iter().take(n_queries).collect()
    };
    let mut sum = F128::ZERO;
    for (i, row) in opened_rows.iter().enumerate() {
        debug_assert_eq!(row.len(), eq.len());
        let dot: F128 = row
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        sum += alpha_weights[i] * dot;
    }
    sum
}

#[inline]
pub(crate) fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (n - 1).ilog2() as usize + 1
    }
}

pub(crate) fn induce_sumcheck_evaluate_at_residual(
    log_msg_cols: usize,
    sks_vks: &[F128],
    queries: &[usize],
    alpha: &[F128],
    ris_for_basis: &[F128],
    yr_log_n: usize,
) -> Vec<F128> {
    use crate::lincheck::build_eq_table;
    use rayon::prelude::*;
    assert_eq!(ris_for_basis.len() + yr_log_n, log_msg_cols);
    let n_queries = queries.len();
    let yr_len = 1usize << yr_log_n;

    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let inv_sks_vks: Vec<F128> = sks_vks
        .iter()
        .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
        .collect();

    let prefix_len = ris_for_basis.len();

    struct PerQuery {
        prefix_prod: F128,
        suffix_w: Vec<F128>, // length = yr_log_n
    }
    let compute_query = |&q: &usize| -> PerQuery {
        let q_field = F128::new(q as u64, 0);
        let mut sks_at_x = Vec::with_capacity(log_msg_cols.max(1));
        if log_msg_cols > 0 {
            sks_at_x.push(q_field);
            for k in 1..log_msg_cols {
                sks_at_x.push(next_s(sks_at_x[k - 1], sks_vks[k - 1]));
            }
            for k in 0..log_msg_cols {
                sks_at_x[k] *= inv_sks_vks[k];
            }
        }
        let mut prefix_prod = F128::ONE;
        for k in 0..prefix_len {
            prefix_prod *= F128::ONE + ris_for_basis[k] * (F128::ONE + sks_at_x[k]);
        }
        let suffix_w = if log_msg_cols > prefix_len {
            sks_at_x[prefix_len..].to_vec()
        } else {
            Vec::new()
        };
        PerQuery {
            prefix_prod,
            suffix_w,
        }
    };
    const PAR_FLOOR: usize = 1024;
    let per_query: Vec<PerQuery> = if n_queries > PAR_FLOOR {
        queries.par_iter().map(compute_query).collect()
    } else {
        queries.iter().map(compute_query).collect()
    };

    let compute_y = |y: usize| -> F128 {
        let mut sum = F128::ZERO;
        for i in 0..n_queries {
            let pq = &per_query[i];
            let mut suffix_prod = F128::ONE;
            for j in 0..yr_log_n {
                let p_j = if (y >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                suffix_prod *= F128::ONE + p_j * (F128::ONE + pq.suffix_w[j]);
            }
            sum += alpha_pows[i] * pq.prefix_prod * suffix_prod;
        }
        sum
    };
    if yr_len > PAR_FLOOR {
        (0..yr_len).into_par_iter().map(compute_y).collect()
    } else {
        (0..yr_len).map(compute_y).collect()
    }
}

pub(crate) fn induce_sumcheck_poly(
    log_msg_cols: usize,
    sks_vks: &[F128],
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    use rayon::prelude::*;
    let n = 1usize << log_msg_cols;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);
    debug_assert_eq!(
        v_challenges.len(),
        opened_rows
            .first()
            .map(|r| r.len().trailing_zeros() as usize)
            .unwrap_or(0)
    );

    let eq = build_eq_table(v_challenges); // length 2^v_challenges.len() = num_interleaved

    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let inv_sks_vks: Vec<F128> = batch_inverse_or_zero(sks_vks);

    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (n_queries + n_threads - 1) / n_threads.max(1);

    let mut partials: Vec<(Vec<F128>, F128)> = (0..n_threads)
        .into_par_iter()
        .map(|t| {
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n_queries);
            if start >= end {
                return (vec![F128::ZERO; n], F128::ZERO);
            }
            let mut accum_basis = vec![F128::ZERO; n];
            let mut local_basis = vec![F128::ZERO; n];
            let mut sks_at_x = vec![F128::ZERO; log_msg_cols.max(1)];
            let mut local_sum = F128::ZERO;

            for i in start..end {
                let row = &opened_rows[i];
                let q = queries[i];
                let ap = alpha_pows[i];

                let dot: F128 = row
                    .iter()
                    .zip(eq.iter())
                    .map(|(&r, &e)| r * e)
                    .fold(F128::ZERO, |a, v| a + v);
                local_sum += dot * ap;

                let q_field = F128::new(q as u64, 0);
                evaluate_scaled_basis_inplace(
                    &mut sks_at_x,
                    &mut local_basis,
                    sks_vks,
                    &inv_sks_vks,
                    q_field,
                    ap,
                );
                for (acc, &v) in accum_basis.iter_mut().zip(local_basis.iter()) {
                    *acc += v;
                }
            }
            (accum_basis, local_sum)
        })
        .collect();

    debug_assert!(!partials.is_empty());
    let (mut basis_poly, mut enforced_sum) = partials.remove(0);
    for (_, ls) in partials.iter() {
        enforced_sum += *ls;
    }
    const REDUCE_PAR_FLOOR: usize = 1 << 12;
    if induce_sched_enabled() && n >= REDUCE_PAR_FLOOR && partials.len() > 1 {
        let reduce_floor = if open_fill_enabled() {
            1usize << 9
        } else {
            REDUCE_PAR_FLOOR
        };
        let chunk = (n / rayon::current_num_threads().max(1)).max(reduce_floor);
        basis_poly
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let base = ci * chunk;
                let len = out.len();
                for (lb, _) in partials.iter() {
                    for (acc, &v) in out.iter_mut().zip(lb[base..base + len].iter()) {
                        *acc += v;
                    }
                }
            });
    } else {
        for (lb, _) in partials.iter() {
            for (acc, &v) in basis_poly.iter_mut().zip(lb.iter()) {
                *acc += v;
            }
        }
    }

    (basis_poly, enforced_sum)
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_avx512(top: &mut [F128], bot: &mut [F128], t: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;

    unsafe {
        let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
        let tb_x64 = ghash_shift64_x4(tb);
        let lanes = top.len() & !3;
        let mut i = 0;
        while i < lanes {
            let va = _mm512_loadu_si512(top.as_ptr().add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(bot.as_ptr().add(i) as *const __m512i);
            let vs = _mm512_xor_si512(va, vb);
            _mm512_storeu_si512(top.as_mut_ptr().add(i) as *mut __m512i, vs);
            let nb = _mm512_xor_si512(vb, ghash_mul_x4_split(vs, tb, tb_x64));
            _mm512_storeu_si512(bot.as_mut_ptr().add(i) as *mut __m512i, nb);
            i += 4;
        }
        while i < top.len() {
            let s = top[i] + bot[i];
            top[i] = s;
            bot[i] = t * s + bot[i];
            i += 1;
        }
    }
}

#[inline(always)]
fn transpose_butterfly(top_h: &mut [F128], bot: &mut [F128], t: F128) {
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    unsafe {
        transpose_butterfly_avx512(top_h, bot, t)
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
        for (a_ref, b_ref) in top_h.iter_mut().zip(bot.iter_mut()) {
            let a = *a_ref;
            let b = *b_ref;
            let s = a + b;
            *a_ref = s;
            *b_ref = t * s + b;
        }
    }
}

#[inline]
fn tnt_pass_b_zero_selected(disabled: bool) -> bool {
    !disabled
}

fn tnt_pass_b_zero_enabled() -> bool {
    static DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_TNT_ZERO_TWIDDLE").is_some());
    tnt_pass_b_zero_selected(*DISABLED)
}

#[inline(never)]
fn transpose_pass_b_block_zero_xor(
    cols: &mut [&mut [F128]],
    stride: usize,
    _layer: usize,
    _ntt: &AdditiveNttF128,
) {
    for u in 0..stride {
        let (lo, hi) = cols.split_at_mut(u + stride);
        let top = &mut *lo[u];
        let bot = &*hi[0];
        debug_assert_eq!(top.len(), bot.len());
        #[cfg(target_feature = "avx512f")]
        {
            use core::arch::x86_64::*;
            unsafe {
                let lanes = top.len() & !3;
                let mut i = 0;
                while i < lanes {
                    let a = _mm512_loadu_si512(top.as_ptr().add(i).cast::<__m512i>());
                    let b = _mm512_loadu_si512(bot.as_ptr().add(i).cast::<__m512i>());
                    _mm512_storeu_si512(
                        top.as_mut_ptr().add(i).cast::<__m512i>(),
                        _mm512_xor_si512(a, b),
                    );
                    i += 4;
                }
                while i < top.len() {
                    top[i] += bot[i];
                    i += 1;
                }
            }
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            for (a, &b) in top.iter_mut().zip(bot.iter()) {
                *a += b;
            }
        }
    }
}

#[inline(never)]
fn transpose_pass_b_block_zero_general(
    cols: &mut [&mut [F128]],
    stride: usize,
    layer: usize,
    ntt: &AdditiveNttF128,
) {
    let t = ntt.twiddle(layer, 0);
    for u in 0..stride {
        let (lo, hi) = cols.split_at_mut(u + stride);
        transpose_butterfly(lo[u], hi[0], t);
    }
}

fn tntt_block_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_TNTT_BLOCK").is_none());
    *ON
}

#[allow(dead_code)] // Full-transform entry kept beside the `_dead` variant.
fn transpose_forward_ntt_dense_layers(ntt: &AdditiveNttF128, data: &mut [F128], top: usize) {
    transpose_forward_ntt_dense_layers_dead(ntt, data, top, 0);
}

fn transpose_forward_ntt_dense_layers_dead(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
    dead_top: usize,
) {
    if tntt_block_enabled() {
        transpose_forward_ntt_dense_layers_blocked_dead(ntt, data, top, dead_top);
    } else {
        transpose_forward_ntt_dense_layers_per_layer(ntt, data, top);
    }
}

fn transpose_forward_ntt_dense_layers_per_layer(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
) {
    use rayon::prelude::*;
    let log_d = data.len().trailing_zeros() as usize;
    debug_assert!(top <= log_d);
    let n_threads = rayon::current_num_threads().max(1);
    for layer in (0..top).rev() {
        let num_blocks = 1usize << layer;
        let block_size = 1usize << (log_d - layer);
        let bsh = block_size >> 1;
        if num_blocks >= n_threads {
            data.par_chunks_mut(block_size)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let t = ntt.twiddle(layer, block);
                    let (top_h, bot) = chunk.split_at_mut(bsh);
                    transpose_butterfly(top_h, bot, t);
                });
        } else {
            for block in 0..num_blocks {
                let t = ntt.twiddle(layer, block);
                let chunk = &mut data[block * block_size..(block + 1) * block_size];
                let (top_h, bot) = chunk.split_at_mut(bsh);
                const SEG: usize = 4096;
                top_h
                    .par_chunks_mut(SEG)
                    .zip(bot.par_chunks_mut(SEG))
                    .for_each(|(a, b)| transpose_butterfly(a, b, t));
            }
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn transpose_forward_ntt_dense_layers_blocked(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
) {
    transpose_forward_ntt_dense_layers_blocked_dead(ntt, data, top, 0);
}

fn transpose_top_only(top: &mut [F128], bot: &[F128]) {
    debug_assert_eq!(top.len(), bot.len());
    #[cfg(target_feature = "avx512f")]
    {
        use core::arch::x86_64::*;
        unsafe {
            let dst = top.as_mut_ptr();
            let src = bot.as_ptr();
            let n = top.len();
            let pair_lanes = n & !7;
            let mut i = 0;
            while i < pair_lanes {
                let a0 = _mm512_loadu_si512(dst.add(i).cast::<__m512i>());
                let a1 = _mm512_loadu_si512(dst.add(i + 4).cast::<__m512i>());
                let b0 = _mm512_loadu_si512(src.add(i).cast::<__m512i>());
                let b1 = _mm512_loadu_si512(src.add(i + 4).cast::<__m512i>());
                _mm512_storeu_si512(dst.add(i).cast::<__m512i>(), _mm512_xor_si512(a0, b0));
                _mm512_storeu_si512(
                    dst.add(i + 4).cast::<__m512i>(),
                    _mm512_xor_si512(a1, b1),
                );
                i += 8;
            }
            let lanes = n & !3;
            while i < lanes {
                let a = _mm512_loadu_si512(dst.add(i).cast::<__m512i>());
                let b = _mm512_loadu_si512(src.add(i).cast::<__m512i>());
                _mm512_storeu_si512(dst.add(i).cast::<__m512i>(), _mm512_xor_si512(a, b));
                i += 4;
            }
            while i < n {
                top[i] += bot[i];
                i += 1;
            }
        }
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        for (a, &b) in top.iter_mut().zip(bot.iter()) {
            *a += b;
        }
    }
}

/// Lane opus-pcs: CHUNK_LOG of the blocked transpose-NTT sites (17 = dense
/// blocked Pass A/B split and tile, 18 = fused sparse densify split). The
/// incumbent constant is 16 at both. `FLOCK_OPUS_TNTT_C17=<n>` /
/// `FLOCK_OPUS_TNTT_C18=<n>` (10..=20, read once) select another value for a
/// same-binary A/B; any value only changes how the same butterflies are
/// partitioned into tasks, never the retained output (tests:
/// `opus_tntt_chunk_log_preserves_outputs`). Tests may override per thread.
fn opus_tntt_chunk_log(site: u32) -> usize {
    if let Some(v) = opus_tntt_test_override() {
        return v;
    }
    fn read(name: &str) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| (10..=20).contains(v))
            .unwrap_or(16)
    }
    static C17: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| read("FLOCK_OPUS_TNTT_C17"));
    static C18: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| read("FLOCK_OPUS_TNTT_C18"));
    if site == 17 { *C17 } else { *C18 }
}

#[cfg(test)]
thread_local! {
    static OPUS_TNTT_C_TEST: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn opus_tntt_test_override() -> Option<usize> {
    OPUS_TNTT_C_TEST.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn opus_tntt_test_override() -> Option<usize> {
    None
}

fn transpose_forward_ntt_dense_layers_blocked_dead(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    top: usize,
    dead_top: usize,
) {
    use rayon::prelude::*;
    let log_d = data.len().trailing_zeros() as usize;
    debug_assert!(top <= log_d);
    if top == 0 {
        return;
    }
    let block_zero_xor = tnt_pass_b_zero_enabled();
    if log_d <= 10 {
        for layer in (0..top).rev() {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let bsh = block_size >> 1;
            if layer < dead_top {
                for jb in 0..num_blocks {
                    let (top_h, bot) =
                        data[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                    transpose_top_only(top_h, bot);
                }
                continue;
            }
            let jb_start = if block_zero_xor {
                let (top_h, bot) = data[..block_size].split_at_mut(bsh);
                transpose_top_only(top_h, bot);
                1
            } else {
                0
            };
            for jb in jb_start..num_blocks {
                let t = ntt.twiddle(layer, jb);
                let (top_h, bot) = data[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                transpose_butterfly(top_h, bot, t);
            }
        }
        return;
    }
    let n_threads = rayon::current_num_threads().max(1);
    // Lane opus-pcs: site-17 chunk log, default 16 (incumbent); see
    // `opus_tntt_chunk_log`. Every value only re-partitions the same layers.
    let chunk_log = opus_tntt_chunk_log(17);

    let split = log_d
        .saturating_sub(chunk_log)
        .max(ceil_log2(n_threads))
        .min(top);

    if top > split {
        let chunk_len = 1usize << (log_d - split);
        data.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(c, chunk)| {
                for layer in (split..top).rev() {
                    let nb = 1usize << (layer - split); // blocks inside this chunk
                    let block_size = 1usize << (log_d - layer);
                    let bsh = block_size >> 1;
                    for jb in 0..nb {
                        let t = ntt.twiddle(layer, (c << (layer - split)) + jb);
                        let (top_h, bot) =
                            chunk[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                        transpose_butterfly(top_h, bot, t);
                    }
                }
            });
    }

    if split > 0 {
        let nseg = 1usize << split;
        let seg = 1usize << (log_d - split);
        let mut tile = ((1usize << chunk_log) / nseg).max(1).min(seg);
        if open_fill_enabled() {
            let cap = seg >> ceil_log2(n_threads);
            if cap >= 64 {
                tile = tile.min(cap);
            }
        }
        let ntiles = seg / tile;
        if ntiles == 1 {
            for b in 0..split {
                let layer = split - 1 - b;
                let num_blocks = 1usize << layer;
                let block_size = 1usize << (log_d - layer);
                let bsh = block_size >> 1;
                if layer < dead_top {
                    for jb in 0..num_blocks {
                        let (top_h, bot) =
                            data[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                        transpose_top_only(top_h, bot);
                    }
                    continue;
                }
                let jb_start = if block_zero_xor {
                    let (top_h, bot) = data[..block_size].split_at_mut(bsh);
                    transpose_top_only(top_h, bot);
                    1
                } else {
                    0
                };
                for jb in jb_start..num_blocks {
                    let t = ntt.twiddle(layer, jb);
                    let (top_h, bot) =
                        data[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                    transpose_butterfly(top_h, bot, t);
                }
            }
            return;
        }
        let mut tiles: Vec<Vec<&mut [F128]>> =
            (0..ntiles).map(|_| Vec::with_capacity(nseg)).collect();
        for part in data.chunks_mut(seg) {
            for (ti, col) in part.chunks_mut(tile).enumerate() {
                tiles[ti].push(col);
            }
        }
        type BlockZeroFn = fn(&mut [&mut [F128]], usize, usize, &AdditiveNttF128);
        let block_zero: BlockZeroFn = if block_zero_xor {
            transpose_pass_b_block_zero_xor
        } else {
            transpose_pass_b_block_zero_general
        };
        tiles.into_par_iter().for_each(|mut cols| {
            for b in 0..split {
                let layer = split - 1 - b;
                let stride = 1usize << b;
                if layer < dead_top {
                    let mut j = 0;
                    while j < nseg {
                        for u in j..j + stride {
                            let (lo, hi) = cols.split_at_mut(u + stride);
                            transpose_top_only(lo[u], hi[0]);
                        }
                        j += stride << 1;
                    }
                    continue;
                }
                block_zero(&mut cols, stride, layer, ntt);
                let mut j = stride << 1;
                while j < nseg {
                    let t = ntt.twiddle(layer, j >> (b + 1));
                    for u in j..j + stride {
                        let (lo, hi) = cols.split_at_mut(u + stride);
                        transpose_butterfly(lo[u], hi[0], t);
                    }
                    j += stride << 1;
                }
            }
        });
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn transpose_forward_ntt(ntt: &AdditiveNttF128, data: &mut [F128], log_d: usize) {
    transpose_forward_ntt_dead(ntt, data, log_d, 0);
}

fn transpose_forward_ntt_dead(
    ntt: &AdditiveNttF128,
    data: &mut [F128],
    log_d: usize,
    dead_top: usize,
) {
    debug_assert_eq!(data.len(), 1usize << log_d);
    debug_assert!(log_d <= ntt.log_domain_size());
    transpose_forward_ntt_dense_layers_dead(ntt, data, log_d, dead_top.min(log_d));
}

pub(crate) fn induce_sumcheck_poly_via_ntt(
    log_msg_cols: usize,
    log_inv_rate: usize,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let n = 1usize << log_msg_cols;
    let log_block = log_msg_cols + log_inv_rate;
    let block_len = 1usize << log_block;
    let n_queries = queries.len();
    assert_eq!(opened_rows.len(), n_queries);

    let eq = build_eq_table(v_challenges);
    let alpha_pows: Vec<F128> = if n_queries == 0 {
        Vec::new()
    } else {
        let table = build_eq_table(alpha);
        debug_assert!(table.len() >= n_queries);
        table.into_iter().take(n_queries).collect()
    };

    let mut enforced_sum = F128::ZERO;
    for i in 0..n_queries {
        let dot: F128 = opened_rows[i]
            .iter()
            .zip(eq.iter())
            .map(|(&r, &e)| r * e)
            .fold(F128::ZERO, |a, v| a + v);
        enforced_sum += dot * alpha_pows[i];
    }

    let mut coeffs = if log_block == 0 {
        let mut c = vec![F128::ZERO; block_len];
        for i in 0..n_queries {
            c[queries[i]] += alpha_pows[i];
        }
        c
    } else {
        let ntt = AdditiveNttF128::standard(log_block);
        transpose_forward_ntt_sparse_dead(
            &ntt,
            queries,
            &alpha_pows,
            log_block,
            lig_tntt_dead_top(log_inv_rate),
        )
    };
    coeffs.truncate(n);
    (coeffs, enforced_sum)
}

struct SparseDualL0 {
    ntt: AdditiveNttF128,
    log_d: usize,
    log_inv_rate: usize,
    depth: usize,
    cache_len: usize,
    queries: Vec<usize>,
    alpha_pows: Vec<F128>,
    w0_pows: Vec<F128>,
    w2_pows: Vec<F128>,
    folded_k: usize,
    inverse_local_blocks: Vec<[F128; 32]>,
    rows: Vec<[F128; 16]>,
    round0_msg: SumcheckMessage,
    snapshots: bool,
    lane_x4: bool,
}

const SPARSE_DUAL_MAX_DEPTH: usize = 4;

#[inline(always)]
fn sparse_dual_one_plus_scale(
    ntt: &AdditiveNttF128,
    log_d: usize,
    s: usize,
    query: usize,
) -> F128 {
    let mut val = ntt.twiddle(log_d - 1 - s, query >> (s + 1));
    if (query >> s) & 1 == 0 {
        val += F128::ONE;
    }
    val
}

fn lig_sparse_dual_snapshot_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_LIG_SPARSE_DUAL_SNAPSHOT").is_none()
    });
    *ON
}

fn lig_sparse_dual_lane_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_LIG_SPARSE_DUAL_LANE_X4").is_none()
    });
    *ON
}

#[inline]
fn sparse_dual_lane_fold(row: &[F128], weights: &[F128], x4: bool) -> F128 {
    debug_assert_eq!(row.len(), weights.len());
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    if x4 {
        if row.len().is_multiple_of(4) {
            return unsafe { sparse_dual_lane_fold_avx512(row, weights) };
        }
        if row.len() == 2 {
            return unsafe { sparse_dual_lane_fold_pair_avx512(row, weights) };
        }
    }
    let _ = x4;
    row.iter()
        .zip(weights)
        .map(|(&v, &w)| v * w)
        .fold(F128::ZERO, |x, y| x + y)
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_lane_fold_pair_avx512(row: &[F128], weights: &[F128]) -> F128 {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    unsafe {
        let r256 = _mm256_loadu_si256(row.as_ptr() as *const __m256i);
        let w256 = _mm256_loadu_si256(weights.as_ptr() as *const __m256i);
        let mut acc = WideGhashX4::zero();
        acc.mul_acc(_mm512_zextsi256_si512(r256), _mm512_zextsi256_si512(w256));
        acc.fold().reduce()
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_lane_fold_avx512(row: &[F128], weights: &[F128]) -> F128 {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    unsafe {
        let mut acc = WideGhashX4::zero();
        let r_ptr = row.as_ptr();
        let w_ptr = weights.as_ptr();
        if row.len() == 8 {
            acc.mul_acc(
                _mm512_loadu_si512(r_ptr as *const __m512i),
                _mm512_loadu_si512(w_ptr as *const __m512i),
            );
            acc.mul_acc(
                _mm512_loadu_si512(r_ptr.add(4) as *const __m512i),
                _mm512_loadu_si512(w_ptr.add(4) as *const __m512i),
            );
        } else {
            let mut i = 0usize;
            while i < row.len() {
                acc.mul_acc(
                    _mm512_loadu_si512(r_ptr.add(i) as *const __m512i),
                    _mm512_loadu_si512(w_ptr.add(i) as *const __m512i),
                );
                i += 4;
            }
        }
        acc.fold().reduce()
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_lane_fold_block_avx512(
    block: &[F128],
    weights: &[F128],
    out: &mut [F128],
) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;
    let num_interleaved = weights.len();
    let cache_len = out.len();
    debug_assert_eq!(block.len(), cache_len * num_interleaved);
    debug_assert!(num_interleaved.is_multiple_of(4));
    unsafe {
        let b_ptr = block.as_ptr();
        let w_ptr = weights.as_ptr();
        if num_interleaved == 8 {
            let vw0 = _mm512_loadu_si512(w_ptr as *const __m512i);
            let vw1 = _mm512_loadu_si512(w_ptr.add(4) as *const __m512i);
            let mut j = 0usize;
            while j + 4 <= cache_len {
                let r0 = b_ptr.add(j * 8);
                let r1 = b_ptr.add((j + 1) * 8);
                let r2 = b_ptr.add((j + 2) * 8);
                let r3 = b_ptr.add((j + 3) * 8);
                let mut a0 = WideGhashX4::from_mul(_mm512_loadu_si512(r0 as *const __m512i), vw0);
                let mut a1 = WideGhashX4::from_mul(_mm512_loadu_si512(r1 as *const __m512i), vw0);
                let mut a2 = WideGhashX4::from_mul(_mm512_loadu_si512(r2 as *const __m512i), vw0);
                let mut a3 = WideGhashX4::from_mul(_mm512_loadu_si512(r3 as *const __m512i), vw0);
                a0.mul_acc(_mm512_loadu_si512(r0.add(4) as *const __m512i), vw1);
                a1.mul_acc(_mm512_loadu_si512(r1.add(4) as *const __m512i), vw1);
                a2.mul_acc(_mm512_loadu_si512(r2.add(4) as *const __m512i), vw1);
                a3.mul_acc(_mm512_loadu_si512(r3.add(4) as *const __m512i), vw1);
                _mm512_storeu_si512(
                    out.as_mut_ptr().add(j) as *mut __m512i,
                    WideGhashX4::fold4_and_reduce(a0, a1, a2, a3),
                );
                j += 4;
            }
            while j < cache_len {
                let r = b_ptr.add(j * 8);
                let mut a = WideGhashX4::from_mul(_mm512_loadu_si512(r as *const __m512i), vw0);
                a.mul_acc(_mm512_loadu_si512(r.add(4) as *const __m512i), vw1);
                *out.get_unchecked_mut(j) = a.fold().reduce();
                j += 1;
            }
        } else {
            let mut j = 0usize;
            while j + 4 <= cache_len {
                let r0 = b_ptr.add(j * num_interleaved);
                let r1 = b_ptr.add((j + 1) * num_interleaved);
                let r2 = b_ptr.add((j + 2) * num_interleaved);
                let r3 = b_ptr.add((j + 3) * num_interleaved);
                let vw0 = _mm512_loadu_si512(w_ptr as *const __m512i);
                let mut a0 = WideGhashX4::from_mul(_mm512_loadu_si512(r0 as *const __m512i), vw0);
                let mut a1 = WideGhashX4::from_mul(_mm512_loadu_si512(r1 as *const __m512i), vw0);
                let mut a2 = WideGhashX4::from_mul(_mm512_loadu_si512(r2 as *const __m512i), vw0);
                let mut a3 = WideGhashX4::from_mul(_mm512_loadu_si512(r3 as *const __m512i), vw0);
                let mut i = 4usize;
                while i + 8 <= num_interleaved {
                    let vw_a = _mm512_loadu_si512(w_ptr.add(i) as *const __m512i);
                    let vw_b = _mm512_loadu_si512(w_ptr.add(i + 4) as *const __m512i);
                    a0.mul_acc2(
                        _mm512_loadu_si512(r0.add(i) as *const __m512i),
                        vw_a,
                        _mm512_loadu_si512(r0.add(i + 4) as *const __m512i),
                        vw_b,
                    );
                    a1.mul_acc2(
                        _mm512_loadu_si512(r1.add(i) as *const __m512i),
                        vw_a,
                        _mm512_loadu_si512(r1.add(i + 4) as *const __m512i),
                        vw_b,
                    );
                    a2.mul_acc2(
                        _mm512_loadu_si512(r2.add(i) as *const __m512i),
                        vw_a,
                        _mm512_loadu_si512(r2.add(i + 4) as *const __m512i),
                        vw_b,
                    );
                    a3.mul_acc2(
                        _mm512_loadu_si512(r3.add(i) as *const __m512i),
                        vw_a,
                        _mm512_loadu_si512(r3.add(i + 4) as *const __m512i),
                        vw_b,
                    );
                    i += 8;
                }
                while i < num_interleaved {
                    let vw = _mm512_loadu_si512(w_ptr.add(i) as *const __m512i);
                    a0.mul_acc(_mm512_loadu_si512(r0.add(i) as *const __m512i), vw);
                    a1.mul_acc(_mm512_loadu_si512(r1.add(i) as *const __m512i), vw);
                    a2.mul_acc(_mm512_loadu_si512(r2.add(i) as *const __m512i), vw);
                    a3.mul_acc(_mm512_loadu_si512(r3.add(i) as *const __m512i), vw);
                    i += 4;
                }
                _mm512_storeu_si512(
                    out.as_mut_ptr().add(j) as *mut __m512i,
                    WideGhashX4::fold4_and_reduce(a0, a1, a2, a3),
                );
                j += 4;
            }
            while j < cache_len {
                let r = b_ptr.add(j * num_interleaved);
                let mut a = WideGhashX4::zero();
                let mut i = 0usize;
                while i < num_interleaved {
                    let vw = _mm512_loadu_si512(w_ptr.add(i) as *const __m512i);
                    a.mul_acc(_mm512_loadu_si512(r.add(i) as *const __m512i), vw);
                    i += 4;
                }
                *out.get_unchecked_mut(j) = a.fold().reduce();
                j += 1;
            }
        }
    }
}

#[inline]
fn sparse_dual_lane_fold_block(
    block: &[F128],
    weights: &[F128],
    out: &mut [F128],
    x4: bool,
) {
    let num_interleaved = weights.len();
    debug_assert_eq!(block.len(), out.len() * num_interleaved);
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    if x4 && num_interleaved.is_multiple_of(4) {
        unsafe { sparse_dual_lane_fold_block_avx512(block, weights, out) };
        return;
    }
    for (j, value) in out.iter_mut().enumerate() {
        let row = &block[j * num_interleaved..(j + 1) * num_interleaved];
        *value = sparse_dual_lane_fold(row, weights, x4);
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_round_dots_half2_pre_avx512(
    local: &[F128],
    vw: core::arch::x86_64::__m512i,
    vw_x64: core::arch::x86_64::__m512i,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_mul_x4_split;
    use core::arch::x86_64::*;
    unsafe {
        let vl = _mm512_loadu_si512(local.as_ptr() as *const __m512i);
        let pl = ghash_mul_x4_split(vl, vw, vw_x64);
        let lo256 = _mm512_castsi512_si256(pl);
        let hi256 = _mm512_extracti64x4_epi64::<1>(pl);
        let f0_xmm = _mm_xor_si128(
            _mm256_castsi256_si128(lo256),
            _mm256_extracti128_si256::<1>(lo256),
        );
        let sum256 = _mm256_xor_si256(lo256, hi256);
        let pair_xmm = _mm_xor_si128(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256::<1>(sum256),
        );
        (
            F128 {
                lo: _mm_extract_epi64::<0>(f0_xmm) as u64,
                hi: _mm_extract_epi64::<1>(f0_xmm) as u64,
            },
            F128 {
                lo: _mm_extract_epi64::<0>(pair_xmm) as u64,
                hi: _mm_extract_epi64::<1>(pair_xmm) as u64,
            },
        )
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_round_dots_half2_avx512(
    local: &[F128],
    weights: &[F128],
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::ghash_shift64_x4;
    use core::arch::x86_64::*;
    unsafe {
        let w256 = _mm256_loadu_si256(weights.as_ptr() as *const __m256i);
        let vw = _mm512_broadcast_i64x4(w256);
        let vw_x64 = ghash_shift64_x4(vw);
        sparse_dual_round_dots_half2_pre_avx512(local, vw, vw_x64)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn sparse_dual_round_dots_avx512(
    local: &[F128],
    weights: &[F128],
    half: usize,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{WideGhashX4, ghash_reduce_acc_x4};
    use core::arch::x86_64::*;
    unsafe {
        let local0_ptr = local.as_ptr();
        let local1_ptr = local0_ptr.add(half);
        let w_ptr = weights.as_ptr();
        let vw0 = _mm512_loadu_si512(w_ptr as *const __m512i);
        let vl0_0 = _mm512_loadu_si512(local0_ptr as *const __m512i);
        let vl1_0 = _mm512_loadu_si512(local1_ptr as *const __m512i);
        let mut acc_f0 = WideGhashX4::from_mul(vl0_0, vw0);
        let mut acc_f1 = WideGhashX4::from_mul(vl1_0, vw0);
        let mut i = 4usize;
        while i < half {
            let vw = _mm512_loadu_si512(w_ptr.add(i) as *const __m512i);
            let vl0 = _mm512_loadu_si512(local0_ptr.add(i) as *const __m512i);
            let vl1 = _mm512_loadu_si512(local1_ptr.add(i) as *const __m512i);
            acc_f0.mul_acc(vl0, vw);
            acc_f1.mul_acc(vl1, vw);
            i += 4;
        }
        let lo = _mm512_xor_si512(
            _mm512_shuffle_i64x2::<0x44>(acc_f0.lo, acc_f1.lo),
            _mm512_shuffle_i64x2::<0xEE>(acc_f0.lo, acc_f1.lo),
        );
        let mid = _mm512_xor_si512(
            _mm512_shuffle_i64x2::<0x44>(acc_f0.mid, acc_f1.mid),
            _mm512_shuffle_i64x2::<0xEE>(acc_f0.mid, acc_f1.mid),
        );
        let hi = _mm512_xor_si512(
            _mm512_shuffle_i64x2::<0x44>(acc_f0.hi, acc_f1.hi),
            _mm512_shuffle_i64x2::<0xEE>(acc_f0.hi, acc_f1.hi),
        );
        let pl = ghash_reduce_acc_x4(lo, mid, hi);
        let lo256 = _mm512_castsi512_si256(pl);
        let hi256 = _mm512_extracti64x4_epi64::<1>(pl);
        let f0_xmm = _mm_xor_si128(
            _mm256_castsi256_si128(lo256),
            _mm256_extracti128_si256::<1>(lo256),
        );
        let sum256 = _mm256_xor_si256(lo256, hi256);
        let pair_xmm = _mm_xor_si128(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256::<1>(sum256),
        );
        (
            F128 {
                lo: _mm_extract_epi64::<0>(f0_xmm) as u64,
                hi: _mm_extract_epi64::<1>(f0_xmm) as u64,
            },
            F128 {
                lo: _mm_extract_epi64::<0>(pair_xmm) as u64,
                hi: _mm_extract_epi64::<1>(pair_xmm) as u64,
            },
        )
    }
}

#[inline(always)]
fn inverse_ntt_butterfly_local(top: &mut [F128], bot: &mut [F128], t: F128) {
    debug_assert_eq!(top.len(), bot.len());
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    if top.len() >= 4 {
        use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
        use core::arch::x86_64::*;
        unsafe {
            let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
            let tb_x64 = ghash_shift64_x4(tb);
            let lanes = top.len() & !3;
            let mut j = 0usize;
            while j < lanes {
                let vy0 = _mm512_loadu_si512(top.as_ptr().add(j) as *const __m512i);
                let vy1 = _mm512_loadu_si512(bot.as_ptr().add(j) as *const __m512i);
                let vv = _mm512_xor_si512(vy0, vy1);
                let vu = _mm512_xor_si512(vy0, ghash_mul_x4_split(vv, tb, tb_x64));
                _mm512_storeu_si512(top.as_mut_ptr().add(j) as *mut __m512i, vu);
                _mm512_storeu_si512(bot.as_mut_ptr().add(j) as *mut __m512i, vv);
                j += 4;
            }
            while j < top.len() {
                let y0 = top[j];
                let v = y0 + bot[j];
                top[j] = y0 + v * t;
                bot[j] = v;
                j += 1;
            }
        }
        return;
    }
    for (y0_ref, y1_ref) in top.iter_mut().zip(bot.iter_mut()) {
        let y0 = *y0_ref;
        let v = y0 + *y1_ref;
        *y0_ref = y0 + v * t;
        *y1_ref = v;
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn inverse_ntt_layer_half1_avx512(
    ntt: &AdditiveNttF128,
    layer: usize,
    global_base: usize,
    data: &mut [F128],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(data.len().is_multiple_of(8));
    unsafe {
        let idx_top = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let idx_bot = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let idx_zip0 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let idx_zip1 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let ptr = data.as_mut_ptr();
        let base_block = global_base >> 1;
        let mut off = 0usize;
        while off < data.len() {
            let v0 = _mm512_loadu_si512(ptr.add(off) as *const __m512i);
            let v1 = _mm512_loadu_si512(ptr.add(off + 4) as *const __m512i);
            let y0 = _mm512_permutex2var_epi64(v0, idx_top, v1);
            let y1 = _mm512_permutex2var_epi64(v0, idx_bot, v1);
            let b = base_block + (off >> 1);
            let vt = f128x4_set(
                ntt.twiddle(layer, b),
                ntt.twiddle(layer, b + 1),
                ntt.twiddle(layer, b + 2),
                ntt.twiddle(layer, b + 3),
            );
            let vv = _mm512_xor_si512(y0, y1);
            let vu = _mm512_xor_si512(y0, ghash_mul_x4(vv, vt));
            let out0 = _mm512_permutex2var_epi64(vu, idx_zip0, vv);
            let out1 = _mm512_permutex2var_epi64(vu, idx_zip1, vv);
            _mm512_storeu_si512(ptr.add(off) as *mut __m512i, out0);
            _mm512_storeu_si512(ptr.add(off + 4) as *mut __m512i, out1);
            off += 8;
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn inverse_ntt_layer_half2_avx512(
    ntt: &AdditiveNttF128,
    layer: usize,
    global_base: usize,
    data: &mut [F128],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(data.len().is_multiple_of(8));
    unsafe {
        let idx_top = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let idx_bot = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        let ptr = data.as_mut_ptr();
        let base_block = global_base >> 2;
        let mut off = 0usize;
        while off < data.len() {
            let v0 = _mm512_loadu_si512(ptr.add(off) as *const __m512i);
            let v1 = _mm512_loadu_si512(ptr.add(off + 4) as *const __m512i);
            let y0 = _mm512_permutex2var_epi64(v0, idx_top, v1);
            let y1 = _mm512_permutex2var_epi64(v0, idx_bot, v1);
            let b = base_block + (off >> 2);
            let t0 = ntt.twiddle(layer, b);
            let t1 = ntt.twiddle(layer, b + 1);
            let vt = f128x4_set(t0, t0, t1, t1);
            let vv = _mm512_xor_si512(y0, y1);
            let vu = _mm512_xor_si512(y0, ghash_mul_x4(vv, vt));
            let out0 = _mm512_permutex2var_epi64(vu, idx_top, vv);
            let out1 = _mm512_permutex2var_epi64(vu, idx_bot, vv);
            _mm512_storeu_si512(ptr.add(off) as *mut __m512i, out0);
            _mm512_storeu_si512(ptr.add(off + 4) as *mut __m512i, out1);
            off += 8;
        }
    }
}

#[inline]
fn sparse_dual_row_stride(depth: usize) -> usize {
    match depth {
        1 | 2 => 1,
        3 => 2,
        4 => 4,
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn inverse_ntt_final_layers_local_staged(
    ntt: &AdditiveNttF128,
    log_d: usize,
    global_base: usize,
    data: &mut [F128],
    qoff: usize,
    depth: usize,
    alpha: F128,
    qscales: &[F128; 4],
    out_rows: &mut [[F128; 16]],
) -> (crate::field::F256Unreduced, crate::field::F256Unreduced) {
    let log_s = data.len().trailing_zeros() as usize;
    debug_assert_eq!(data.len(), 1usize << log_s);
    debug_assert_eq!(global_base & (data.len() - 1), 0);
    debug_assert!(depth < log_s && depth <= SPARSE_DUAL_MAX_DEPTH);
    debug_assert!(qoff < data.len());
    out_rows[0][12..16].copy_from_slice(qscales);
    let mut u0 = crate::field::F256Unreduced::ZERO;
    let mut u2 = crate::field::F256Unreduced::ZERO;
    for (stage, layer) in ((log_d - log_s)..log_d).rev().enumerate() {
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        let mut handled = false;
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        if data.len() >= 8 && (half == 1 || half == 2) {
            unsafe {
                if half == 1 {
                    inverse_ntt_layer_half1_avx512(ntt, layer, global_base, data);
                } else {
                    inverse_ntt_layer_half2_avx512(ntt, layer, global_base, data);
                }
            }
            handled = true;
        }
        if !handled {
            for off in (0..data.len()).step_by(block_size) {
                let block = (global_base + off) / block_size;
                let twiddle = ntt.twiddle(layer, block);
                let (top, bot) = data[off..off + block_size].split_at_mut(half);
                inverse_ntt_butterfly_local(top, bot, twiddle);
            }
        }
        let len = 2usize << stage;
        let start = qoff & !(len - 1);
        match stage {
            0 => {
                let f0 = data[start];
                let f1 = data[start + 1];
                let a2 = alpha * (f0 + f1);
                u0 = alpha.mul_unreduced(f0);
                u2 = a2.mul_unreduced(qscales[0]);
            }
            1 => {
                out_rows[0][0..4].copy_from_slice(&data[start..start + 4]);
            }
            2 => {
                out_rows[0][4..12].copy_from_slice(&data[start..start + 8]);
            }
            3 => {
                out_rows[1].copy_from_slice(&data[start..start + 16]);
            }
            4 => {
                out_rows[2].copy_from_slice(&data[start..start + 16]);
                out_rows[3].copy_from_slice(&data[start + 16..start + 32]);
            }
            _ => unreachable!(),
        }
    }
    (u0, u2)
}

fn forward_ntt_local_until(
    ntt: &AdditiveNttF128,
    log_d: usize,
    global_base: usize,
    data: &mut [F128],
    end_layer: usize,
) {
    let log_s = data.len().trailing_zeros() as usize;
    let start_layer = log_d - log_s;
    debug_assert!((start_layer..=log_d).contains(&end_layer));
    debug_assert_eq!(global_base & (data.len() - 1), 0);
    for layer in start_layer..end_layer {
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        for off in (0..data.len()).step_by(block_size) {
            let block = (global_base + off) / block_size;
            let twiddle = ntt.twiddle(layer, block);
            for j in 0..half {
                let v = data[off + half + j];
                let u = data[off + j] + v * twiddle;
                data[off + j] = u;
                data[off + half + j] = v + u;
            }
        }
    }
}

fn inverse_ntt_final_layers_local(
    ntt: &AdditiveNttF128,
    log_d: usize,
    global_base: usize,
    data: &mut [F128],
) {
    let log_s = data.len().trailing_zeros() as usize;
    debug_assert_eq!(data.len(), 1usize << log_s);
    debug_assert_eq!(global_base & (data.len() - 1), 0);
    for layer in ((log_d - log_s)..log_d).rev() {
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        if data.len() >= 8 && (half == 1 || half == 2) {
            unsafe {
                if half == 1 {
                    inverse_ntt_layer_half1_avx512(ntt, layer, global_base, data);
                } else {
                    inverse_ntt_layer_half2_avx512(ntt, layer, global_base, data);
                }
            }
            continue;
        }
        for off in (0..data.len()).step_by(block_size) {
            let block = (global_base + off) / block_size;
            let twiddle = ntt.twiddle(layer, block);
            let (top, bot) = data[off..off + block_size].split_at_mut(half);
            inverse_ntt_butterfly_local(top, bot, twiddle);
        }
    }
}

/// Lane gl2: `FLOCK_NO_GL2_STREAM=1` keeps the sparse dual's per-query work
/// after query sampling (incumbent). Default: the alpha-independent part of
/// every query (lane fold of its codeword block, the staged inverse-NTT
/// snapshot rows, the scale factors and the stage-0 values) runs on helper
/// workers WHILE the owner is still drawing the later queries; after alpha
/// only the alpha-weighted sums and the move into sorted order remain.
/// Read once.
fn gl2_stream_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL2_STREAM").is_none());
    *ON
}

/// Shared state of one streamed sparse dual (see [`gl2_stream_enabled`]).
/// Slots are indexed by DRAW order (the order distinct queries were sampled).
/// Aligned so the `Arc` counts (bumped by spinners when they re-read the
/// side list) never share a line with the fields the owner reads per draw.
#[repr(align(128))]
struct Gl2DualStream {
    count: usize,
    log_d: usize,
    depth: usize,
    cache_len: usize,
    stride: usize,
    num_interleaved: usize,
    lane_x4: bool,
    ntt: AdditiveNttF128,
    lane_weights: Vec<F128>,
    codeword: Gl2Ptr<F128>,
    /// Draw slot d holds the d-th distinct query once `produced > d`. The
    /// owner publishes whole 64-byte lines (8 slots) at a time, so it never
    /// stores into a line a helper has read; helpers poll `produced` with a
    /// pause backoff so the owner's publishing store rarely meets sharers.
    qlines: std::cell::UnsafeCell<Vec<Gl2Line>>,
    produced: Gl2Pad,
    next: Gl2Pad,
    done: Gl2Pad,
    poll: usize,
    side: std::sync::Mutex<Option<crate::hot_pool::side::Task>>,
    /// Per draw slot: `stride` snapshot rows, then (cached[qoff], f0, f0+f1, qscale0).
    rows: std::cell::UnsafeCell<Vec<[F128; 16]>>,
    vals: std::cell::UnsafeCell<Vec<[F128; 4]>>,
    owner_sibling: Option<usize>,
}

// SAFETY: every draw slot of `rows` / `vals` is written by exactly one thread
// (the one that claimed its index from `next`) and read by the owner only
// after `done == count` (acquire); the codeword outlives the stream's use
// (the owner waits for `done == count` before returning).
unsafe impl Sync for Gl2DualStream {}
unsafe impl Send for Gl2DualStream {}

impl Gl2DualStream {
    #[allow(clippy::too_many_arguments)]
    fn start(
        depth: usize,
        log_d: usize,
        codeword: &[F128],
        num_interleaved: usize,
        lane_challenges: &[F128],
        count: usize,
    ) -> std::sync::Arc<Self> {
        assert!((1..=SPARSE_DUAL_MAX_DEPTH).contains(&depth));
        assert_eq!(num_interleaved, 1usize << lane_challenges.len());
        assert_eq!(codeword.len(), (1usize << log_d) * num_interleaved);
        let cache_len = 1usize << (depth + 1);
        let stride = sparse_dual_row_stride(depth);
        let owner = rayon::current_thread_index();
        let n = rayon::current_num_threads();
        let st = std::sync::Arc::new(Self {
            count,
            log_d,
            depth,
            cache_len,
            stride,
            num_interleaved,
            lane_x4: lig_sparse_dual_lane_x4_enabled(),
            ntt: AdditiveNttF128::standard(log_d),
            lane_weights: build_eq_table(lane_challenges),
            codeword: Gl2Ptr::of(codeword),
            qlines: std::cell::UnsafeCell::new(vec![Gl2Line([0; 8]); count.div_ceil(8)]),
            produced: Gl2Pad(std::sync::atomic::AtomicUsize::new(0)),
            next: Gl2Pad(std::sync::atomic::AtomicUsize::new(0)),
            done: Gl2Pad(std::sync::atomic::AtomicUsize::new(0)),
            poll: gl2_stream_knobs().1,
            side: std::sync::Mutex::new(None),
            rows: std::cell::UnsafeCell::new(crate::alloc_uninit_vec::<[F128; 16]>(count * stride)),
            vals: std::cell::UnsafeCell::new(crate::alloc_uninit_vec::<[F128; 4]>(count)),
            owner_sibling: owner.map(|o| (o + n / 2) % n.max(1)),
        });
        let helpers = n.saturating_sub(2).min(count).min(gl2_stream_knobs().0);
        if gl2_side_enabled() {
            // Joined by idle hot-pool spinners (not the owner's sibling: the
            // draws are a latency-bound BLAKE3 chain on the owner's core).
            let active = std::sync::Arc::new(Gl2Pad(std::sync::atomic::AtomicUsize::new(0)));
            let h = st.clone();
            let t: crate::hot_pool::side::Task = std::sync::Arc::new(move |sib: bool| {
                use std::sync::atomic::Ordering;
                // Loads only until there is work: a finished stream costs a
                // spinner one shared-line load per poll.
                if sib || h.next.0.load(Ordering::Relaxed) >= h.count {
                    return;
                }
                if active.0.fetch_add(1, Ordering::AcqRel) < helpers {
                    h.work(true);
                }
                active.0.fetch_sub(1, Ordering::AcqRel);
            });
            *st.side.lock().unwrap_or_else(|e| e.into_inner()) = Some(t.clone());
            crate::hot_pool::side::register(t);
        } else {
            for _ in 0..helpers {
                let h = st.clone();
                rayon::spawn(move || {
                    if h.owner_sibling.is_some() && rayon::current_thread_index() == h.owner_sibling {
                        return;
                    }
                    h.work(true);
                });
            }
        }
        st
    }

    /// Record distinct query number `d` (owner only, draw order); publish
    /// every completed 64-byte line. The owner keeps `d` itself: a counter
    /// in this struct would share a line with fields every helper reads.
    #[inline]
    fn push(&self, d: usize, q: usize) {
        use std::sync::atomic::Ordering;
        // SAFETY: unpublished lines are touched only by the owner; helpers
        // read a line only after its release below.
        unsafe {
            (*self.qlines.get()).as_mut_ptr().add(d / 8).cast::<usize>().add(d % 8).write(q);
            if (d + 1) % 8 == 0 || d + 1 == self.count {
                self.produced.0.store(d + 1, Ordering::Release);
            }
        }
    }

    /// Claim and process draw slots until all are claimed.
    fn work(&self, helper: bool) {
        use std::sync::atomic::Ordering;
        loop {
            let d = self.next.0.load(Ordering::Relaxed);
            if d >= self.count {
                return;
            }
            if self.produced.0.load(Ordering::Acquire) <= d {
                for _ in 0..self.poll {
                    std::hint::spin_loop();
                }
                continue;
            }
            if self.next.0.compare_exchange_weak(d, d + 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
                continue;
            }
            // SAFETY: slot d < produced (acquire above): its line is final.
            let q = unsafe { (*self.qlines.get()).as_ptr().add(d / 8).cast::<usize>().add(d % 8).read() };
            self.one(d, q);
            self.done.0.fetch_add(1, Ordering::AcqRel);
            if helper {
                crate::rstrace::evd("ds:q", d as u32);
            }
        }
    }

    fn one(&self, d: usize, query: usize) {
        let (cache_len, ni, log_d) = (self.cache_len, self.num_interleaved, self.log_d);
        // SAFETY: see the `Sync` impl.
        let cw = unsafe { self.codeword.get() };
        let base = query & !(cache_len - 1);
        let qoff = query - base;
        let block = &cw[base * ni..(base + cache_len) * ni];
        let mut cached = [F128::ZERO; 32];
        sparse_dual_lane_fold_block(block, &self.lane_weights, &mut cached[..cache_len], self.lane_x4);
        let c = cached[qoff];
        let mut qscales = [F128::ZERO; 4];
        for (s, slot) in qscales.iter_mut().enumerate().take((self.depth + 1).min(4)) {
            *slot = sparse_dual_one_plus_scale(&self.ntt, log_d, s, query);
        }
        // SAFETY: slot `d` is this thread's (claimed from `next`).
        let out_rows = unsafe {
            let rows = &mut *self.rows.get();
            std::slice::from_raw_parts_mut(rows.as_mut_ptr().add(d * self.stride), self.stride)
        };
        let (f0, f1) = inverse_ntt_final_layers_local_staged_raw(
            &self.ntt,
            log_d,
            base,
            &mut cached[..cache_len],
            qoff,
            self.depth,
            &qscales,
            out_rows,
        );
        unsafe { (&mut *self.vals.get())[d] = [c, f0, f0 + f1, qscales[0]] };
    }

    /// Owner, after the last draw: help with unclaimed slots, then wait.
    fn finish(&self) {
        use std::sync::atomic::Ordering;
        self.work(false);
        while self.done.0.load(Ordering::Acquire) < self.count {
            std::hint::spin_loop();
        }
    }

    /// The side-list entry, for one batched removal at the end of the open.
    fn take_side(&self) -> Option<crate::hot_pool::side::Task> {
        self.side.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

/// [`inverse_ntt_final_layers_local_staged`] without the alpha-dependent
/// stage-0 sums: returns the stage-0 pair `(f0, f1)` instead. Same layers,
/// same snapshot rows.
#[allow(clippy::too_many_arguments)]
fn inverse_ntt_final_layers_local_staged_raw(
    ntt: &AdditiveNttF128,
    log_d: usize,
    global_base: usize,
    data: &mut [F128],
    qoff: usize,
    depth: usize,
    qscales: &[F128; 4],
    out_rows: &mut [[F128; 16]],
) -> (F128, F128) {
    let log_s = data.len().trailing_zeros() as usize;
    debug_assert!(depth < log_s && depth <= SPARSE_DUAL_MAX_DEPTH);
    out_rows[0][12..16].copy_from_slice(qscales);
    let (mut f0, mut f1) = (F128::ZERO, F128::ZERO);
    for (stage, layer) in ((log_d - log_s)..log_d).rev().enumerate() {
        let block_size = 1usize << (log_d - layer);
        let half = block_size >> 1;
        let mut handled = false;
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        if data.len() >= 8 && (half == 1 || half == 2) {
            unsafe {
                if half == 1 {
                    inverse_ntt_layer_half1_avx512(ntt, layer, global_base, data);
                } else {
                    inverse_ntt_layer_half2_avx512(ntt, layer, global_base, data);
                }
            }
            handled = true;
        }
        if !handled {
            for off in (0..data.len()).step_by(block_size) {
                let block = (global_base + off) / block_size;
                let twiddle = ntt.twiddle(layer, block);
                let (top, bot) = data[off..off + block_size].split_at_mut(half);
                inverse_ntt_butterfly_local(top, bot, twiddle);
            }
        }
        let len = 2usize << stage;
        let start = qoff & !(len - 1);
        match stage {
            0 => {
                f0 = data[start];
                f1 = data[start + 1];
            }
            1 => out_rows[0][0..4].copy_from_slice(&data[start..start + 4]),
            2 => out_rows[0][4..12].copy_from_slice(&data[start..start + 8]),
            3 => out_rows[1].copy_from_slice(&data[start..start + 16]),
            4 => {
                out_rows[2].copy_from_slice(&data[start..start + 16]);
                out_rows[3].copy_from_slice(&data[start + 16..start + 32]);
            }
            _ => unreachable!(),
        }
    }
    (f0, f1)
}

/// Lane gl2: [`sample_distinct_queries`] that also publishes each new
/// distinct query to `stream` in draw order. Same draws, same result; also
/// returns, per sorted position, the query's draw slot.
fn sample_distinct_queries_streamed<Ch: Challenger>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
    stream: &Gl2DualStream,
) -> (Vec<usize>, Vec<u32>) {
    assert!(count <= block_len);
    assert_eq!(stream.count, count);
    thread_local! {
        static SEEN: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(count);
    let small = count <= 256 && block_len < u32::MAX as usize;
    if small {
        Gl2QSet::with(|seen| {
            crate::rstrace::evd("qs:begin", count as u32);
            while out.len() < count {
                let v = challenger.sample_f128();
                let q = (v.lo as usize) % block_len;
                if seen.insert(q) {
                    stream.push(out.len(), q);
                    out.push((q, out.len() as u32));
                }
            }
            crate::rstrace::evd("qs:drawn", count as u32);
        });
    } else {
        SEEN.with(|cell| {
            let mut seen = cell.borrow_mut();
            let words = block_len.div_ceil(64);
            if seen.len() < words {
                seen.resize(words, 0);
            }
            while out.len() < count {
                let v = challenger.sample_f128();
                let q = (v.lo as usize) % block_len;
                let (w, b) = (q >> 6, 1u64 << (q & 63));
                if seen[w] & b == 0 {
                    seen[w] |= b;
                    stream.push(out.len(), q);
                    out.push((q, out.len() as u32));
                }
            }
            for &(q, _) in &out {
                seen[q >> 6] = 0;
            }
        });
    }
    out.sort_unstable();
    crate::rstrace::evd("qs:sorted", count as u32);
    out.into_iter().unzip()
}

impl SparseDualL0 {
    /// Lane gl2: the [`Self::new`] result (snapshot path) from a finished
    /// [`Gl2DualStream`]: identical fields and enforced sum, since every
    /// alpha-weighted term is the same field product, summed in F256 before
    /// one reduction exactly as in `new`.
    #[allow(clippy::too_many_arguments)]
    fn from_stream(
        stream: &Gl2DualStream,
        log_inv_rate: usize,
        queries: Vec<usize>,
        perm: &[u32],
        alpha: &[F128],
    ) -> (Self, F128) {
        use rayon::prelude::*;
        let (depth, log_d, stride, n_queries) = (stream.depth, stream.log_d, stream.stride, stream.count);
        assert!(log_inv_rate + depth <= log_d);
        assert_eq!(queries.len(), n_queries);
        let alpha_pows: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            let mut table = build_eq_table(alpha);
            table.truncate(queries.len());
            table
        };
        assert_eq!(alpha_pows.len(), n_queries);
        stream.finish();
        crate::rstrace::evd("ds:done", 0);
        // SAFETY: `finish` observed `done == count` (acquire): every slot is written.
        let (src_rows, vals) = unsafe { (&*stream.rows.get(), &*stream.vals.get()) };
        let mut rows = crate::alloc_uninit_vec::<[F128; 16]>(n_queries * stride);
        let mut w2_pows = crate::alloc_uninit_vec::<F128>(n_queries);
        const CH: usize = 16;
        let zero = crate::field::F256Unreduced::ZERO;
        let (enf_acc, u0_acc, u2_acc) = rows
            .par_chunks_mut(stride * CH)
            .zip(w2_pows.par_chunks_mut(CH))
            .enumerate()
            .map(|(c, (rows_c, w2_c))| {
                let (mut e, mut u0, mut u2) = (zero, zero, zero);
                for (j, w2) in w2_c.iter_mut().enumerate() {
                    let i = c * CH + j;
                    let d = perm[i] as usize;
                    rows_c[j * stride..(j + 1) * stride]
                        .copy_from_slice(&src_rows[d * stride..(d + 1) * stride]);
                    let [cq, f0, s, qs0] = vals[d];
                    let a = alpha_pows[i];
                    e ^= a.mul_unreduced(cq);
                    u0 ^= a.mul_unreduced(f0);
                    u2 ^= (a * s).mul_unreduced(qs0);
                    *w2 = a * qs0;
                }
                (e, u0, u2)
            })
            .reduce(|| (zero, zero, zero), |a, b| (a.0 ^ b.0, a.1 ^ b.1, a.2 ^ b.2));
        let enforced_sum = enf_acc.reduce();
        let round0_msg = SumcheckMessage {
            u_0: u0_acc.reduce(),
            u_2: u2_acc.reduce(),
        };
        (
            Self {
                ntt: AdditiveNttF128::standard(log_d),
                log_d,
                log_inv_rate,
                depth,
                cache_len: stream.cache_len,
                queries,
                w0_pows: alpha_pows.clone(),
                alpha_pows,
                w2_pows,
                folded_k: 0,
                inverse_local_blocks: Vec::new(),
                rows,
                round0_msg,
                snapshots: true,
                lane_x4: stream.lane_x4,
            },
            enforced_sum,
        )
    }

    fn new(
        depth: usize,
        log_d: usize,
        log_inv_rate: usize,
        l0_codeword: &[F128],
        num_interleaved: usize,
        lane_challenges: &[F128],
        queries: &[usize],
        alpha: &[F128],
    ) -> (Self, F128) {
        use rayon::prelude::*;
        assert!((1..=SPARSE_DUAL_MAX_DEPTH).contains(&depth));
        assert!(log_inv_rate + depth <= log_d);
        assert_eq!(num_interleaved, 1usize << lane_challenges.len());
        assert_eq!(l0_codeword.len(), (1usize << log_d) * num_interleaved);
        let lane_weights = build_eq_table(lane_challenges);
        let alpha_pows: Vec<F128> = if queries.is_empty() {
            Vec::new()
        } else {
            let mut table = build_eq_table(alpha);
            table.truncate(queries.len());
            table
        };
        assert_eq!(alpha_pows.len(), queries.len());

        let cache_len = 1usize << (depth + 1);
        let ntt = AdditiveNttF128::standard(log_d);
        let snapshots = lig_sparse_dual_snapshot_enabled();
        let lane_x4 = lig_sparse_dual_lane_x4_enabled();
        let n_queries = queries.len();
        let stride = sparse_dual_row_stride(depth);
        let mut inverse_local_blocks = if snapshots {
            Vec::new()
        } else {
            crate::alloc_uninit_vec::<[F128; 32]>(n_queries)
        };
        let mut rows = if snapshots {
            crate::alloc_uninit_vec::<[F128; 16]>(n_queries * stride)
        } else {
            Vec::new()
        };
        let w0_pows = if snapshots {
            alpha_pows.clone()
        } else {
            Vec::new()
        };
        let mut w2_pows = if snapshots {
            crate::alloc_uninit_vec::<F128>(n_queries)
        } else {
            Vec::new()
        };
        let use_par = crate::serial_par_enabled() && n_queries * cache_len * num_interleaved >= 16384;
        let (enforced_sum, round0_msg) = if snapshots {
            let (enf_acc, u0_acc, u2_acc) = if use_par {
                queries
                    .par_iter()
                    .zip(&alpha_pows)
                    .zip(w2_pows.par_iter_mut())
                    .zip(rows.par_chunks_exact_mut(stride))
                    .with_min_len(lf1_dual_minlen())
                    .map(|(((&query, &alpha), w2_slot), out_rows)| {
                        let base = query & !(cache_len - 1);
                        let qoff = query - base;
                        let block =
                            &l0_codeword[base * num_interleaved..(base + cache_len) * num_interleaved];
                        let mut cached = [F128::ZERO; 32];
                        sparse_dual_lane_fold_block(
                            block,
                            &lane_weights,
                            &mut cached[..cache_len],
                            lane_x4,
                        );
                        let enforced = alpha.mul_unreduced(cached[qoff]);
                        let mut qscales = [F128::ZERO; 4];
                        for (s, slot) in qscales.iter_mut().enumerate().take((depth + 1).min(4)) {
                            *slot = sparse_dual_one_plus_scale(&ntt, log_d, s, query);
                        }
                        *w2_slot = alpha * qscales[0];
                        let (u0, u2) = inverse_ntt_final_layers_local_staged(
                            &ntt,
                            log_d,
                            base,
                            &mut cached[..cache_len],
                            qoff,
                            depth,
                            alpha,
                            &qscales,
                            out_rows,
                        );
                        (enforced, u0, u2)
                    })
                    .reduce(
                        || {
                            (
                                crate::field::F256Unreduced::ZERO,
                                crate::field::F256Unreduced::ZERO,
                                crate::field::F256Unreduced::ZERO,
                            )
                        },
                        |a, b| (a.0 ^ b.0, a.1 ^ b.1, a.2 ^ b.2),
                    )
            } else {
                let mut enf_acc = crate::field::F256Unreduced::ZERO;
                let mut u0_acc = crate::field::F256Unreduced::ZERO;
                let mut u2_acc = crate::field::F256Unreduced::ZERO;
                for (((&query, &alpha), w2_slot), out_rows) in queries
                    .iter()
                    .zip(&alpha_pows)
                    .zip(w2_pows.iter_mut())
                    .zip(rows.chunks_exact_mut(stride))
                {
                    let base = query & !(cache_len - 1);
                    let qoff = query - base;
                    let block =
                        &l0_codeword[base * num_interleaved..(base + cache_len) * num_interleaved];
                    let mut cached = [F128::ZERO; 32];
                    sparse_dual_lane_fold_block(
                        block,
                        &lane_weights,
                        &mut cached[..cache_len],
                        lane_x4,
                    );
                    enf_acc ^= alpha.mul_unreduced(cached[qoff]);
                    let mut qscales = [F128::ZERO; 4];
                    for (s, slot) in qscales.iter_mut().enumerate().take((depth + 1).min(4)) {
                        *slot = sparse_dual_one_plus_scale(&ntt, log_d, s, query);
                    }
                    *w2_slot = alpha * qscales[0];
                    let (u0, u2) = inverse_ntt_final_layers_local_staged(
                        &ntt,
                        log_d,
                        base,
                        &mut cached[..cache_len],
                        qoff,
                        depth,
                        alpha,
                        &qscales,
                        out_rows,
                    );
                    u0_acc ^= u0;
                    u2_acc ^= u2;
                }
                (enf_acc, u0_acc, u2_acc)
            };
            (
                enf_acc.reduce(),
                SumcheckMessage {
                    u_0: u0_acc.reduce(),
                    u_2: u2_acc.reduce(),
                },
            )
        } else {
            let enf = queries
                .par_iter()
                .zip(&alpha_pows)
                .zip(inverse_local_blocks.par_iter_mut())
                .map(|((&query, &alpha), cached)| {
                    let base = query & !(cache_len - 1);
                    let block =
                        &l0_codeword[base * num_interleaved..(base + cache_len) * num_interleaved];
                    sparse_dual_lane_fold_block(
                        block,
                        &lane_weights,
                        &mut cached[..cache_len],
                        lane_x4,
                    );
                    let enforced = alpha * cached[query - base];
                    inverse_ntt_final_layers_local(&ntt, log_d, base, &mut cached[..cache_len]);
                    enforced
                })
                .reduce(|| F128::ZERO, |a, b| a + b);
            (
                enf,
                SumcheckMessage {
                    u_0: F128::ZERO,
                    u_2: F128::ZERO,
                },
            )
        };

        (
            Self {
                ntt,
                log_d,
                log_inv_rate,
                depth,
                cache_len,
                queries: queries.to_vec(),
                alpha_pows,
                w0_pows,
                w2_pows,
                folded_k: 0,
                inverse_local_blocks,
                rows,
                round0_msg,
                snapshots,
                lane_x4,
            },
            enforced_sum,
        )
    }

    fn round_msg_query_unreduced(
        &self,
        index: usize,
        fold_challenges: &[F128],
        fold_weights: &[F128],
        log_s: usize,
        block_len: usize,
        stride: usize,
    ) -> (crate::field::F256Unreduced, crate::field::F256Unreduced) {
        let ntt = &self.ntt;
        let query = self.queries[index];
        let alpha = self.alpha_pows[index];
        let cache_base = query & !(self.cache_len - 1);
        let global_base = query & !(block_len - 1);
        let cache_off = global_base - cache_base;
        let mut residue: [F128; 32];
        let local: &[F128] = if self.snapshots {
            let base_row = index * stride;
            match log_s {
                2 => &self.rows[base_row][0..4],
                3 => &self.rows[base_row][4..12],
                4 => &self.rows[base_row + 1],
                5 => unsafe {
                    core::slice::from_raw_parts(self.rows[base_row + 2].as_ptr(), 32)
                },
                _ => unreachable!(),
            }
        } else {
            residue = self.inverse_local_blocks[index];
            forward_ntt_local_until(
                ntt,
                self.log_d,
                cache_base,
                &mut residue[..self.cache_len],
                self.log_d - log_s,
            );
            &residue[cache_off..cache_off + block_len]
        };

        let (w0, w2) = if self.snapshots {
            let scales = &self.rows[index * stride][12..16];
            let mut d0 = F128::ONE + fold_challenges[0] * scales[0];
            for (s, &r_s) in fold_challenges.iter().enumerate().skip(1) {
                d0 *= F128::ONE + r_s * scales[s];
            }
            let top_scale = if log_s <= 4 {
                scales[log_s - 1]
            } else {
                sparse_dual_one_plus_scale(ntt, self.log_d, log_s - 1, query)
            };
            let w0 = alpha * d0;
            (w0, w0 * top_scale)
        } else {
            let mut d0 = F128::ONE;
            for (s, &r_s) in fold_challenges.iter().enumerate() {
                let one_plus_scale = sparse_dual_one_plus_scale(ntt, self.log_d, s, query);
                d0 *= F128::ONE + r_s * one_plus_scale;
            }
            let one_plus_top = sparse_dual_one_plus_scale(ntt, self.log_d, log_s - 1, query);
            let w0 = alpha * d0;
            (w0, w0 * one_plus_top)
        };

        let half = block_len >> 1;
        if self.lane_x4 && half == 1 {
            let f0 = local[0];
            let pair_sum = f0 + local[1];
            return (w0.mul_unreduced(f0), w2.mul_unreduced(pair_sum));
        }
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        if self.lane_x4 && half == 2 {
            let (f0, pair_sum) =
                unsafe { sparse_dual_round_dots_half2_avx512(local, fold_weights) };
            return (w0.mul_unreduced(f0), w2.mul_unreduced(pair_sum));
        }
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        if self.lane_x4 && half.is_multiple_of(4) {
            let (f0, pair_sum) =
                unsafe { sparse_dual_round_dots_avx512(local, fold_weights, half) };
            return (w0.mul_unreduced(f0), w2.mul_unreduced(pair_sum));
        }
        let f0 = local[..half]
            .iter()
            .zip(fold_weights)
            .map(|(&v, &w)| v * w)
            .fold(F128::ZERO, |x, y| x + y);
        let f1 = local[half..]
            .iter()
            .zip(fold_weights)
            .map(|(&v, &w)| v * w)
            .fold(F128::ZERO, |x, y| x + y);
        let pair_sum = f0 + f1;

        (w0.mul_unreduced(f0), w2.mul_unreduced(pair_sum))
    }

    fn round_msg_impl(&self, fold_challenges: &[F128], chunked: bool) -> SumcheckMessage {
        use rayon::prelude::*;
        assert!(fold_challenges.len() <= self.depth);
        if self.snapshots && fold_challenges.is_empty() {
            return self.round0_msg;
        }
        let log_s = fold_challenges.len() + 1;
        let block_len = 1usize << log_s;
        let stride = sparse_dual_row_stride(self.depth);
        let mut weights = [F128::ZERO; 16];
        weights[0] = F128::ONE;
        let mut weights_len = 1usize;
        for &challenge in fold_challenges {
            for i in 0..weights_len {
                let value = weights[i];
                let hi = value * challenge;
                weights[weights_len + i] = hi;
                weights[i] = value + hi;
            }
            weights_len <<= 1;
        }
        let fold_weights = &weights[..weights_len];
        let (u0_acc, u2_acc) = if chunked {
            (0..self.queries.len())
                .into_par_iter()
                .with_min_len(32)
                .map(|index| {
                    self.round_msg_query_unreduced(
                        index,
                        fold_challenges,
                        fold_weights,
                        log_s,
                        block_len,
                        stride,
                    )
                })
                .reduce(
                    || {
                        (
                            crate::field::F256Unreduced::ZERO,
                            crate::field::F256Unreduced::ZERO,
                        )
                    },
                    |a, b| (a.0 ^ b.0, a.1 ^ b.1),
                )
        } else {
            let mut u0 = crate::field::F256Unreduced::ZERO;
            let mut u2 = crate::field::F256Unreduced::ZERO;
            for index in 0..self.queries.len() {
                let (du0, du2) = self.round_msg_query_unreduced(
                    index,
                    fold_challenges,
                    fold_weights,
                    log_s,
                    block_len,
                    stride,
                );
                u0 ^= du0;
                u2 ^= du2;
            }
            (u0, u2)
        };
        SumcheckMessage {
            u_0: u0_acc.reduce(),
            u_2: u2_acc.reduce(),
        }
    }

    fn round_msg(&self, fold_challenges: &[F128]) -> SumcheckMessage {
        self.round_msg_impl(fold_challenges, false)
    }

    fn round_msg_step(&mut self, fold_challenges: &[F128]) -> SumcheckMessage {
        assert!(fold_challenges.len() <= self.depth);
        if !self.snapshots
            || fold_challenges.is_empty()
            || fold_challenges.len() != self.folded_k + 1
        {
            return self.round_msg(fold_challenges);
        }
        let k = fold_challenges.len();
        let r_last = fold_challenges[k - 1];
        let log_s = k + 1;
        let block_len = 1usize << log_s;
        let half = block_len >> 1;
        let stride = sparse_dual_row_stride(self.depth);
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        unsafe {
            use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
            use core::arch::x86_64::*;
            let rv = _mm512_broadcast_i32x4(_mm_set_epi64x(r_last.hi as i64, r_last.lo as i64));
            let rv64 = ghash_shift64_x4(rv);
            let lanes = self.w0_pows.len() & !3;
            let w0_ptr = self.w0_pows.as_mut_ptr();
            let w2_ptr = self.w2_pows.as_ptr();
            let mut i = 0usize;
            while i < lanes {
                let vw0 = _mm512_loadu_si512(w0_ptr.add(i) as *const __m512i);
                let vw2 = _mm512_loadu_si512(w2_ptr.add(i) as *const __m512i);
                let p = ghash_mul_x4_split(vw2, rv, rv64);
                _mm512_storeu_si512(w0_ptr.add(i) as *mut __m512i, _mm512_xor_si512(vw0, p));
                i += 4;
            }
            while i < self.w0_pows.len() {
                self.w0_pows[i] += r_last * self.w2_pows[i];
                i += 1;
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )))]
        for (w0, &w2) in self.w0_pows.iter_mut().zip(&self.w2_pows) {
            *w0 += r_last * w2;
        }
        self.folded_k = k;

        let mut weights = [F128::ZERO; 16];
        weights[0] = F128::ONE;
        let mut weights_len = 1usize;
        for &challenge in fold_challenges {
            for i in 0..weights_len {
                let value = weights[i];
                let hi = value * challenge;
                weights[weights_len + i] = hi;
                weights[i] = value + hi;
            }
            weights_len <<= 1;
        }
        let fold_weights = &weights[..weights_len];
        let mut u0 = crate::field::F256Unreduced::ZERO;
        let mut u2 = crate::field::F256Unreduced::ZERO;
        for index in 0..self.queries.len() {
            let base_row = index * stride;
            let local: &[F128] = match log_s {
                2 => &self.rows[base_row][0..4],
                3 => &self.rows[base_row][4..12],
                4 => &self.rows[base_row + 1],
                5 => unsafe {
                    core::slice::from_raw_parts(self.rows[base_row + 2].as_ptr(), 32)
                },
                _ => unreachable!(),
            };
            let w0 = self.w0_pows[index];
            let top_scale = if log_s <= 4 {
                self.rows[base_row][12 + log_s - 1]
            } else {
                sparse_dual_one_plus_scale(&self.ntt, self.log_d, log_s - 1, self.queries[index])
            };
            let w2 = w0 * top_scale;
            if k < self.depth {
                self.w2_pows[index] = w2;
            }
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            if self.lane_x4 && half == 2 {
                let (f0, pair_sum) =
                    unsafe { sparse_dual_round_dots_half2_avx512(local, fold_weights) };
                u0 ^= w0.mul_unreduced(f0);
                u2 ^= w2.mul_unreduced(pair_sum);
                continue;
            }
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            if self.lane_x4 && half.is_multiple_of(4) {
                let (f0, pair_sum) =
                    unsafe { sparse_dual_round_dots_avx512(local, fold_weights, half) };
                u0 ^= w0.mul_unreduced(f0);
                u2 ^= w2.mul_unreduced(pair_sum);
                continue;
            }
            let f0 = local[..half]
                .iter()
                .zip(fold_weights)
                .map(|(&v, &w)| v * w)
                .fold(F128::ZERO, |x, y| x + y);
            let f1 = local[half..]
                .iter()
                .zip(fold_weights)
                .map(|(&v, &w)| v * w)
                .fold(F128::ZERO, |x, y| x + y);
            u0 ^= w0.mul_unreduced(f0);
            u2 ^= w2.mul_unreduced(f0 + f1);
        }
        SumcheckMessage {
            u_0: u0.reduce(),
            u_2: u2.reduce(),
        }
    }

    #[cfg(test)]
    fn round_msg_chunked(&self, fold_challenges: &[F128]) -> SumcheckMessage {
        self.round_msg_impl(fold_challenges, true)
    }

    fn scale(&mut self, alpha: F128) {
        self.round0_msg.u_0 *= alpha;
        self.round0_msg.u_2 *= alpha;
        self.folded_k = 0;
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        {
            use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
            use core::arch::x86_64::*;
            unsafe {
                let ax = _mm512_broadcast_i32x4(_mm_set_epi64x(alpha.hi as i64, alpha.lo as i64));
                let ax64 = ghash_shift64_x4(ax);
                let lanes = self.alpha_pows.len() & !3;
                let ptr = self.alpha_pows.as_mut_ptr();
                let mut i = 0usize;
                while i < lanes {
                    let v = _mm512_loadu_si512(ptr.add(i) as *const __m512i);
                    let p = ghash_mul_x4_split(v, ax, ax64);
                    _mm512_storeu_si512(ptr.add(i) as *mut __m512i, p);
                    i += 4;
                }
                while i < self.alpha_pows.len() {
                    self.alpha_pows[i] *= alpha;
                    i += 1;
                }
                if self.snapshots {
                    self.w0_pows.copy_from_slice(&self.alpha_pows);
                    let w2_ptr = self.w2_pows.as_mut_ptr();
                    let mut j = 0usize;
                    while j < lanes {
                        let v = _mm512_loadu_si512(w2_ptr.add(j) as *const __m512i);
                        let p = ghash_mul_x4_split(v, ax, ax64);
                        _mm512_storeu_si512(w2_ptr.add(j) as *mut __m512i, p);
                        j += 4;
                    }
                    while j < self.w2_pows.len() {
                        self.w2_pows[j] *= alpha;
                        j += 1;
                    }
                }
            }
            return;
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )))]
        {
            for value in &mut self.alpha_pows {
                *value *= alpha;
            }
            if self.snapshots {
                self.w0_pows.copy_from_slice(&self.alpha_pows);
                for value in &mut self.w2_pows {
                    *value *= alpha;
                }
            }
        }
    }

    fn materialize_after_folds(&self, fold_challenges: &[F128]) -> Vec<F128> {
        let k = fold_challenges.len();
        assert!(k > 0 && k == self.depth && k <= SPARSE_DUAL_MAX_DEPTH);
        let full_ntt = &self.ntt;
        let n_queries = self.queries.len();
        let stride = sparse_dual_row_stride(self.depth);
        let reduced_log_d = self.log_d - k;
        if self.snapshots && self.folded_k == k && n_queries <= 256 {
            let mut pos_stack = [0usize; 256];
            for (dst, &query) in pos_stack[..n_queries].iter_mut().zip(&self.queries) {
                *dst = query >> k;
            }
            let mut folded = transpose_forward_ntt_sparse_dead(
                full_ntt,
                &pos_stack[..n_queries],
                &self.w0_pows,
                reduced_log_d,
                lig_tntt_dead_top(self.log_inv_rate),
            );
            folded.truncate(1usize << (reduced_log_d - self.log_inv_rate));
            return folded;
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        if self.snapshots && self.lane_x4 && n_queries <= 256 {
            use crate::field::gf2_128::x86_64::{
                f128x4_set, ghash_mul_x4, ghash_mul_x4_split, ghash_shift64_x4,
            };
            use core::arch::x86_64::*;
            let mut pos_stack = [0usize; 256];
            let mut val_stack = [F128::ZERO; 256];
            for (dst, &query) in pos_stack[..n_queries].iter_mut().zip(&self.queries) {
                *dst = query >> k;
            }
            unsafe {
                let v_one = _mm512_broadcast_i32x4(_mm_set_epi64x(0, 1));
                let mut vr = [_mm512_setzero_si512(); SPARSE_DUAL_MAX_DEPTH];
                let mut vr_x64 = [_mm512_setzero_si512(); SPARSE_DUAL_MAX_DEPTH];
                for (s, &r_s) in fold_challenges.iter().enumerate() {
                    let v = _mm512_broadcast_i32x4(_mm_set_epi64x(r_s.hi as i64, r_s.lo as i64));
                    vr[s] = v;
                    vr_x64[s] = ghash_shift64_x4(v);
                }
                let lanes = n_queries & !3;
                let a_ptr = self.alpha_pows.as_ptr();
                let v_ptr = val_stack.as_mut_ptr();
                let mut idx = 0usize;
                while idx < lanes {
                    let s0 = &self.rows.get_unchecked(idx * stride)[12..16];
                    let s1 = &self.rows.get_unchecked((idx + 1) * stride)[12..16];
                    let s2 = &self.rows.get_unchecked((idx + 2) * stride)[12..16];
                    let s3 = &self.rows.get_unchecked((idx + 3) * stride)[12..16];
                    let mut acc = _mm512_loadu_si512(a_ptr.add(idx) as *const __m512i);
                    for s in 0..k {
                        let vs = f128x4_set(s0[s], s1[s], s2[s], s3[s]);
                        let factor =
                            _mm512_xor_si512(v_one, ghash_mul_x4_split(vs, vr[s], vr_x64[s]));
                        acc = ghash_mul_x4(acc, factor);
                    }
                    _mm512_storeu_si512(v_ptr.add(idx) as *mut __m512i, acc);
                    idx += 4;
                }
                while idx < n_queries {
                    let scales = &self.rows[idx * stride][12..16];
                    let mut v = self.alpha_pows[idx];
                    for (s, &r_s) in fold_challenges.iter().enumerate() {
                        v *= F128::ONE + r_s * scales[s];
                    }
                    *v_ptr.add(idx) = v;
                    idx += 1;
                }
            }
            let mut folded = transpose_forward_ntt_sparse_dead(
                full_ntt,
                &pos_stack[..n_queries],
                &val_stack[..n_queries],
                reduced_log_d,
                lig_tntt_dead_top(self.log_inv_rate),
            );
            folded.truncate(1usize << (reduced_log_d - self.log_inv_rate));
            return folded;
        }
        let mut positions = Vec::with_capacity(n_queries);
        let mut values: Vec<F128> = Vec::with_capacity(n_queries);
        for (index, &query) in self.queries.iter().enumerate() {
            let mut local_value = self.alpha_pows[index];
            if self.snapshots {
                let scales = &self.rows[index * stride][12..16];
                for (s, &r_s) in fold_challenges.iter().enumerate() {
                    local_value *= F128::ONE + r_s * scales[s];
                }
            } else {
                for (s, &r_s) in fold_challenges.iter().enumerate() {
                    let one_plus_scale =
                        sparse_dual_one_plus_scale(full_ntt, self.log_d, s, query);
                    local_value *= F128::ONE + r_s * one_plus_scale;
                }
            }
            positions.push(query >> k);
            values.push(local_value);
        }
        let mut folded = transpose_forward_ntt_sparse_dead(
            full_ntt,
            &positions,
            &values,
            reduced_log_d,
            lig_tntt_dead_top(self.log_inv_rate),
        );
        folded.truncate(1usize << (reduced_log_d - self.log_inv_rate));
        folded
    }
}

fn eq_split_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_EQ_SPLIT").is_none());
    *ON
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
fn eq_split_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_EQ_SPLIT_X4").is_none());
    *ON
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn eq_expand_block_x4(out: &mut [F128], lo: &[F128], e: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    debug_assert_eq!(out.len(), lo.len());
    unsafe {
        let eb = _mm512_broadcast_i32x4(_mm_set_epi64x(e.hi as i64, e.lo as i64));
        let eb_x64 = ghash_shift64_x4(eb);
        let lanes = out.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let v = _mm512_loadu_si512(lo.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(
                out.as_mut_ptr().add(i) as *mut __m512i,
                ghash_mul_x4_split(v, eb, eb_x64),
            );
            i += 4;
        }
        while i < out.len() {
            out[i] = lo[i] * e;
            i += 1;
        }
    }
}

fn build_eq_table_split(point: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    const SPLIT_MIN_LOG: usize = 17;
    const SPLIT_MIN_LOG_FILL: usize = 13;
    let split_min = if open_fill_enabled() {
        SPLIT_MIN_LOG_FILL
    } else {
        SPLIT_MIN_LOG
    };
    let d = point.len();
    if !eq_split_enabled() || d < split_min {
        return build_eq_table(point);
    }
    let n_threads = rayon::current_num_threads().max(1);
    let log_blocks = (ceil_log2(n_threads) + 2).min(d - 1);
    let h = d - log_blocks;
    let lo = build_eq_table(&point[..h]);
    let hi = build_eq_table(&point[h..]);
    debug_assert_eq!(lo.len(), 1usize << h);
    debug_assert_eq!(hi.len(), 1usize << log_blocks);
    let mut out = crate::alloc_uninit_vec::<F128>(1usize << d);
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    let x4 = eq_split_x4_enabled();
    out.par_chunks_mut(1usize << h)
        .zip(hi.par_iter())
        .for_each(|(chunk, &e)| {
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            if x4 {
                unsafe { eq_expand_block_x4(chunk, &lo, e) };
                return;
            }
            for (o, &l) in chunk.iter_mut().zip(lo.iter()) {
                *o = l * e;
            }
        });
    out
}

pub(crate) fn induce_sched_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_INDUCE_SCHED").is_none());
    *ON
}

fn induce_lazy_sks_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_INDUCE_LAZY_SKS").is_none());
    *ON
}

fn induce_ntt_crossover_c() -> usize {
    if induce_sched_enabled() { 1 } else { 4 }
}

pub(crate) fn induce_sumcheck_poly_auto(
    log_msg_cols: usize,
    log_inv_rate: usize,
    sks_vks: Option<&[F128]>,
    opened_rows: &[Vec<F128>],
    v_challenges: &[F128],
    queries: &[usize],
    alpha: &[F128],
) -> (Vec<F128>, F128) {
    let log_block = log_msg_cols + log_inv_rate;
    let use_ntt = log_msg_cols >= 12
        && queries.len() > induce_ntt_crossover_c() * (1usize << log_inv_rate) * log_block.max(1);
    if use_ntt {
        induce_sumcheck_poly_via_ntt(
            log_msg_cols,
            log_inv_rate,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    } else {
        let computed;
        let sks_vks = match sks_vks {
            Some(table) => table,
            None => {
                computed = eval_sk_at_vks(log_msg_cols);
                &computed
            }
        };
        induce_sumcheck_poly(
            log_msg_cols,
            sks_vks,
            opened_rows,
            v_challenges,
            queries,
            alpha,
        )
    }
}

fn induce_fused_densify_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_INDUCE_FUSED_DENSIFY").is_none());
    *ON
}

#[cfg_attr(not(test), allow(dead_code))]
fn window_slots(nwin: usize, processed: Vec<(usize, Vec<F128>)>) -> Vec<Option<Vec<F128>>> {
    let mut slots: Vec<Option<Vec<F128>>> = (0..nwin).map(|_| None).collect();
    for (w, buf) in processed {
        debug_assert!(w < nwin);
        debug_assert!(slots[w].is_none(), "window {w} densified twice");
        slots[w] = Some(buf);
    }
    slots
}

#[cfg_attr(not(test), allow(dead_code))]
fn densify_windows_fused(n: usize, k: usize, slots: Vec<Option<Vec<F128>>>) -> Vec<F128> {
    use rayon::prelude::*;
    debug_assert_eq!(slots.len(), n >> k);
    let mut data: Vec<F128> = crate::alloc_uninit_vec(n);
    data.par_chunks_mut(1usize << k)
        .zip(slots.into_par_iter())
        .for_each(|(dst, src)| match src {
            Some(buf) => dst.copy_from_slice(&buf),
            None => dst.fill(F128::ZERO),
        });
    data
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_blocks_bsh1(p: *mut F128, t0: F128, t1: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    unsafe {
        let ta = _mm512_broadcast_i32x4(_mm_set_epi64x(t0.hi as i64, t0.lo as i64));
        let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t1.hi as i64, t1.lo as i64));
        let tv = _mm512_mask_blend_epi64(0xF0, ta, tb);
        let tv_x64 = ghash_shift64_x4(tv);
        let v = _mm512_loadu_si512(p as *const __m512i);
        let sw = _mm512_shuffle_i64x2::<0xB1>(v, v);
        let s = _mm512_xor_si512(v, sw);
        let nb = _mm512_xor_si512(v, ghash_mul_x4_split(s, tv, tv_x64));
        _mm512_storeu_si512(p as *mut __m512i, _mm512_mask_blend_epi64(0x33, nb, s));
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_blocks_bsh2(p: *mut F128, t: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    unsafe {
        let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t.hi as i64, t.lo as i64));
        let tb_x64 = ghash_shift64_x4(tb);
        let v = _mm512_loadu_si512(p as *const __m512i);
        let sw = _mm512_shuffle_i64x2::<0x4E>(v, v);
        let s = _mm512_xor_si512(v, sw);
        let nb = _mm512_xor_si512(v, ghash_mul_x4_split(s, tb, tb_x64));
        _mm512_storeu_si512(p as *mut __m512i, _mm512_mask_blend_epi64(0x0F, nb, s));
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_blocks_bsh1_pair(
    p: *mut F128,
    t0: F128,
    t1: F128,
    t2: F128,
    t3: F128,
) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    unsafe {
        let ta = _mm512_broadcast_i32x4(_mm_set_epi64x(t0.hi as i64, t0.lo as i64));
        let tb = _mm512_broadcast_i32x4(_mm_set_epi64x(t1.hi as i64, t1.lo as i64));
        let tc = _mm512_broadcast_i32x4(_mm_set_epi64x(t2.hi as i64, t2.lo as i64));
        let td = _mm512_broadcast_i32x4(_mm_set_epi64x(t3.hi as i64, t3.lo as i64));
        let tv = _mm512_mask_blend_epi64(0xF0, ta, tb);
        let tw = _mm512_mask_blend_epi64(0xF0, tc, td);
        let tv_x64 = ghash_shift64_x4(tv);
        let tw_x64 = ghash_shift64_x4(tw);
        let v0 = _mm512_loadu_si512(p as *const __m512i);
        let v1 = _mm512_loadu_si512(p.add(4) as *const __m512i);
        let s0 = _mm512_xor_si512(v0, _mm512_shuffle_i64x2::<0xB1>(v0, v0));
        let s1 = _mm512_xor_si512(v1, _mm512_shuffle_i64x2::<0xB1>(v1, v1));
        let m0 = ghash_mul_x4_split(s0, tv, tv_x64);
        let m1 = ghash_mul_x4_split(s1, tw, tw_x64);
        _mm512_storeu_si512(
            p as *mut __m512i,
            _mm512_mask_blend_epi64(0x33, _mm512_xor_si512(v0, m0), s0),
        );
        _mm512_storeu_si512(
            p.add(4) as *mut __m512i,
            _mm512_mask_blend_epi64(0x33, _mm512_xor_si512(v1, m1), s1),
        );
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[inline]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn transpose_butterfly_blocks_bsh2_pair(p: *mut F128, t0: F128, t1: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    unsafe {
        let tb0 = _mm512_broadcast_i32x4(_mm_set_epi64x(t0.hi as i64, t0.lo as i64));
        let tb1 = _mm512_broadcast_i32x4(_mm_set_epi64x(t1.hi as i64, t1.lo as i64));
        let tb0_x64 = ghash_shift64_x4(tb0);
        let tb1_x64 = ghash_shift64_x4(tb1);
        let v0 = _mm512_loadu_si512(p as *const __m512i);
        let v1 = _mm512_loadu_si512(p.add(4) as *const __m512i);
        let s0 = _mm512_xor_si512(v0, _mm512_shuffle_i64x2::<0x4E>(v0, v0));
        let s1 = _mm512_xor_si512(v1, _mm512_shuffle_i64x2::<0x4E>(v1, v1));
        let m0 = ghash_mul_x4_split(s0, tb0, tb0_x64);
        let m1 = ghash_mul_x4_split(s1, tb1, tb1_x64);
        _mm512_storeu_si512(
            p as *mut __m512i,
            _mm512_mask_blend_epi64(0x0F, _mm512_xor_si512(v0, m0), s0),
        );
        _mm512_storeu_si512(
            p.add(4) as *mut __m512i,
            _mm512_mask_blend_epi64(0x0F, _mm512_xor_si512(v1, m1), s1),
        );
    }
}

#[inline]
fn transpose_forward_ntt_window_dense_slice(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    buf: &mut [F128],
) {
    debug_assert_eq!(buf.len(), 1usize << k);
    for s in 0..k {
        let layer = log_d - 1 - s;
        let bsh = 1usize << s;
        let block_size = bsh << 1;
        let nblocks = (1usize << k) / block_size;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        {
            let block_base = w << (k - s - 1);
            if bsh == 1 {
                let mut jb = 0;
                while jb + 3 < nblocks {
                    let t0 = ntt.twiddle(layer, block_base + jb);
                    let t1 = ntt.twiddle(layer, block_base + jb + 1);
                    let t2 = ntt.twiddle(layer, block_base + jb + 2);
                    let t3 = ntt.twiddle(layer, block_base + jb + 3);
                    unsafe {
                        transpose_butterfly_blocks_bsh1_pair(
                            buf.as_mut_ptr().add(jb * 2),
                            t0,
                            t1,
                            t2,
                            t3,
                        )
                    };
                    jb += 4;
                }
                while jb + 1 < nblocks {
                    let t0 = ntt.twiddle(layer, block_base + jb);
                    let t1 = ntt.twiddle(layer, block_base + jb + 1);
                    unsafe {
                        transpose_butterfly_blocks_bsh1(buf.as_mut_ptr().add(jb * 2), t0, t1)
                    };
                    jb += 2;
                }
                if jb < nblocks {
                    let t = ntt.twiddle(layer, block_base + jb);
                    let base = jb * 2;
                    let (top_h, bot) = buf[base..base + 2].split_at_mut(1);
                    transpose_butterfly(top_h, bot, t);
                }
                continue;
            }
            if bsh == 2 {
                let mut jb = 0;
                while jb + 1 < nblocks {
                    let t0 = ntt.twiddle(layer, block_base + jb);
                    let t1 = ntt.twiddle(layer, block_base + jb + 1);
                    unsafe {
                        transpose_butterfly_blocks_bsh2_pair(
                            buf.as_mut_ptr().add(jb * 4),
                            t0,
                            t1,
                        )
                    };
                    jb += 2;
                }
                if jb < nblocks {
                    let t = ntt.twiddle(layer, block_base + jb);
                    unsafe { transpose_butterfly_blocks_bsh2(buf.as_mut_ptr().add(jb * 4), t) };
                }
                continue;
            }
        }
        for jb in 0..nblocks {
            let t = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
            let base = jb * block_size;
            let (top_h, bot) = buf[base..base + block_size].split_at_mut(bsh);
            transpose_butterfly(top_h, bot, t);
        }
    }
}

#[inline]
fn transpose_forward_ntt_window_dense(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    mut buf: Vec<F128>,
) -> (usize, Vec<F128>) {
    transpose_forward_ntt_window_dense_slice(ntt, log_d, k, w, &mut buf);
    (w, buf)
}

fn singleton_uninit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_SINGLETON_UNINIT").is_none());
    *ON
}

fn take_singleton_buf(k: usize) -> Vec<F128> {
    if singleton_uninit_enabled() {
        crate::alloc_uninit_vec::<F128>(1usize << k)
    } else {
        vec![F128::ZERO; 1usize << k]
    }
}

fn expand_singleton_into(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    p: usize,
    value: F128,
    out: &mut [F128],
) {
    assert!(k > 0 && k <= log_d);
    assert!(p < 1usize << k);
    assert_eq!(out.len(), 1usize << k);
    let mut scale0 = ntt.twiddle(log_d - 1, (w << (k - 1)) + (p >> 1));
    if p & 1 != 0 {
        scale0 += F128::ONE;
    }
    let v0 = value;
    let v1 = if value == F128::ONE {
        scale0
    } else {
        value * scale0
    };
    if k == 1 {
        out[0] = v0;
        out[1] = v1;
        return;
    }
    let mut scale1 = ntt.twiddle(log_d - 2, (w << (k - 2)) + (p >> 2));
    if (p >> 1) & 1 != 0 {
        scale1 += F128::ONE;
    }
    let v2 = if value == F128::ONE {
        scale1
    } else {
        v0 * scale1
    };
    let v3 = v1 * scale1;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    {
        use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4_split, ghash_shift64_x4};
        use core::arch::x86_64::*;
        unsafe {
            let v_lo = f128x4_set(v0, v1, v2, v3);
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, v_lo);
            if k == 2 {
                return;
            }
            let mut scale2 = ntt.twiddle(log_d - 3, (w << (k - 3)) + (p >> 3));
            if (p >> 2) & 1 != 0 {
                scale2 += F128::ONE;
            }
            let e2 = _mm512_broadcast_i32x4(_mm_set_epi64x(scale2.hi as i64, scale2.lo as i64));
            let e2_x64 = ghash_shift64_x4(e2);
            let v_hi = ghash_mul_x4_split(v_lo, e2, e2_x64);
            _mm512_storeu_si512(out.as_mut_ptr().add(4) as *mut __m512i, v_hi);
        }
        let mut len = 8usize;
        for s in 3..k {
            let layer = log_d - 1 - s;
            let jb = p >> (s + 1);
            let mut scale = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
            if (p >> s) & 1 != 0 {
                scale += F128::ONE;
            }
            let (current, rest) = out[..2 * len].split_at_mut(len);
            unsafe { eq_expand_block_x4(&mut rest[..len], current, scale) };
            len <<= 1;
        }
        return;
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    {
        out[0] = v0;
        out[1] = v1;
        out[2] = v2;
        out[3] = v3;
        let mut len = 4usize;
        for s in 2..k {
            let layer = log_d - 1 - s;
            let jb = p >> (s + 1);
            let mut scale = ntt.twiddle(layer, (w << (k - s - 1)) + jb);
            if (p >> s) & 1 != 0 {
                scale += F128::ONE;
            }
            let (current, rest) = out[..2 * len].split_at_mut(len);
            for (dst, &src) in rest[..len].iter_mut().zip(current.iter()) {
                *dst = src * scale;
            }
            len <<= 1;
        }
    }
}

#[allow(dead_code)] // Retained singleton expansion oracle.
fn transpose_forward_ntt_window_singleton(
    ntt: &AdditiveNttF128,
    log_d: usize,
    k: usize,
    w: usize,
    p: usize,
    value: F128,
) -> (usize, Vec<F128>) {
    let mut buf = take_singleton_buf(k);
    expand_singleton_into(ntt, log_d, k, w, p, value, &mut buf);
    (w, buf)
}

fn lig_induce_singleton_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_INDUCE_SINGLETON").is_none());
    *ON
}

const INDUCE_K_MAX_MEM: usize = 12;

fn induce_window_k(log_d: usize, n_queries: usize, threads: usize, singleton: bool) -> usize {
    if log_d < 12 || n_queries == 0 {
        return 0;
    }
    if !singleton {
        return 8usize.min(log_d);
    }
    let k_par = log_d.saturating_sub(ceil_log2(threads.max(1)) + 2);
    let hi = INDUCE_K_MAX_MEM.min(k_par).min(log_d).max(8);
    let cost = |k: usize| -> u128 {
        (n_queries as u128) * ((1u128 << k) - 1) + (((log_d - k) as u128) << (log_d - 1))
    };
    (8..=hi).min_by_key(|&k| cost(k)).unwrap_or(8)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InduceSingletonStats {
    singleton_hits: usize,
    multi_hits: usize,
    collision_fallbacks: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
fn transpose_forward_ntt_sparse(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
) -> Vec<F128> {
    transpose_forward_ntt_sparse_inner(ntt, positions, values, log_d, None, None).0
}

fn lig_tntt_dead_top(log_inv_rate: usize) -> usize {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_TNTT_DEAD_BOT").is_none());
    if *ON { log_inv_rate } else { 0 }
}

fn transpose_forward_ntt_sparse_dead(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    dead_top: usize,
) -> Vec<F128> {
    transpose_forward_ntt_sparse_inner_dead(ntt, positions, values, log_d, None, None, dead_top).0
}

#[cfg_attr(not(test), allow(dead_code))]
fn transpose_forward_ntt_sparse_inner(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    singleton_override: Option<bool>,
    k_override: Option<usize>,
) -> (Vec<F128>, InduceSingletonStats) {
    transpose_forward_ntt_sparse_inner_dead(
        ntt,
        positions,
        values,
        log_d,
        singleton_override,
        k_override,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn transpose_forward_ntt_sparse_inner_dead(
    ntt: &AdditiveNttF128,
    positions: &[usize],
    values: &[F128],
    log_d: usize,
    singleton_override: Option<bool>,
    k_override: Option<usize>,
    dead_top: usize,
) -> (Vec<F128>, InduceSingletonStats) {
    use rayon::prelude::*;
    use std::collections::HashMap;
    let n = 1usize << log_d;
    assert_eq!(positions.len(), values.len());
    let fill = open_fill_enabled();
    let singleton_on = fill && singleton_override.unwrap_or_else(lig_induce_singleton_enabled);
    let sparse_trie = singleton_on && sparse_window::enabled();
    let k = if log_d < 12 || positions.is_empty() {
        0
    } else {
        match k_override {
            Some(v) => v.clamp(1, log_d),
            None => induce_window_k(
                log_d,
                positions.len(),
                rayon::current_num_threads(),
                singleton_on,
            ),
        }
    };

    if k == 0 {
        let mut data = vec![F128::ZERO; n];
        for (&p, &v) in positions.iter().zip(values) {
            data[p] += v;
        }
        if log_d > 0 {
            transpose_forward_ntt_dead(ntt, &mut data, log_d, dead_top);
        }
        return (data, InduceSingletonStats::default());
    }

    let wmask = (1usize << k) - 1;
    let mut win_vec: Vec<(usize, Vec<F128>)> = Vec::new();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    if fill {
        order.extend(0..positions.len() as u32);
        if sparse_trie {
            if !positions.windows(2).all(|w| w[0] <= w[1]) {
                order.sort_by_key(|&i| positions[i as usize]);
            }
        } else if !positions.windows(2).all(|w| (w[0] >> k) <= (w[1] >> k)) {
            order.sort_by_key(|&i| positions[i as usize] >> k);
        }
        let mut s = 0usize;
        while s < order.len() {
            let w = positions[order[s] as usize] >> k;
            let mut e = s + 1;
            while e < order.len() && positions[order[e] as usize] >> k == w {
                e += 1;
            }
            runs.push((s, e));
            s = e;
        }
    } else {
        let mut windows: HashMap<usize, Vec<F128>> = HashMap::new();
        for (&p, &v) in positions.iter().zip(values) {
            let buf = windows
                .entry(p >> k)
                .or_insert_with(|| vec![F128::ZERO; 1 << k]);
            buf[p & wmask] += v;
        }
        win_vec = windows.into_iter().collect();
    }

    let multi_max = k / 2;
    let singleton_stats = if singleton_on && fill {
        InduceSingletonStats {
            singleton_hits: runs.iter().filter(|(s, e)| e - s == 1).count(),
            multi_hits: runs
                .iter()
                .filter(|(s, e)| e - s > 1 && e - s <= multi_max)
                .count(),
            collision_fallbacks: runs.iter().filter(|(s, e)| e - s > multi_max).count(),
        }
    } else {
        InduceSingletonStats::default()
    };

    let top = log_d - k;
    let mut dense_top = top;
    let nwins = if fill { runs.len() } else { win_vec.len() };
    let ot = open_timing();
    let mf0 = if ot { minor_faults() } else { 0 };
    let _tw = std::time::Instant::now();
    let (mut data, win_ms, alloc_ms, dens_ms) = if fill && induce_fused_densify_enabled() {
        let nwin = n >> k;
        let mut win_run: Vec<(u32, u32)> = vec![(0, 0); nwin];
        for &(rs, re) in &runs {
            let w = positions[order[rs] as usize] >> k;
            win_run[w] = (rs as u32, re as u32);
        }
        let alloc_ms = _tw.elapsed().as_secs_f64() * 1e3;
        let _td = std::time::Instant::now();
        let mut data: Vec<F128> = crate::alloc_uninit_vec(n);
        let n_threads = rayon::current_num_threads().max(1);
        // Lane opus-pcs: site-18 chunk log, default 16 (incumbent).
        let chunk_log = opus_tntt_chunk_log(18);
        let split = log_d
            .saturating_sub(chunk_log)
            .max(ceil_log2(n_threads))
            .min(top);
        let fuse_pass_a = tntt_block_enabled() && top > split;
        let wins_per_chunk_log = if fuse_pass_a { top - split } else { 0 };
        let wins_per_chunk = 1usize << wins_per_chunk_log;
        let chunk_len = 1usize << (k + wins_per_chunk_log);
        if fuse_pass_a {
            dense_top = split;
        }
        data.par_chunks_mut(chunk_len)
            .zip(win_run.par_chunks(wins_per_chunk))
            .enumerate()
            .for_each_init(Vec::<F128>::new, |scratch, (c, (chunk, wr_chunk))| {
                let w_base = c << wins_per_chunk_log;
                for (wi, (dst, &(rs, re))) in chunk
                    .chunks_exact_mut(1usize << k)
                    .zip(wr_chunk.iter())
                    .enumerate()
                {
                    let w = w_base + wi;
                    let rs = rs as usize;
                    let re = re as usize;
                    let nnz = re - rs;
                    if nnz == 0 {
                        dst.fill(F128::ZERO);
                        continue;
                    }
                    let first = order[rs] as usize;
                    if sparse_trie {
                        sparse_window::expand(
                            ntt, log_d, k, w, positions, values, &order[rs..re], dst,
                        );
                    } else if singleton_on && nnz <= multi_max {
                        expand_singleton_into(
                            ntt,
                            log_d,
                            k,
                            w,
                            positions[first] & wmask,
                            values[first],
                            dst,
                        );
                        if nnz > 1 {
                            if scratch.len() < 1usize << k {
                                *scratch = take_singleton_buf(k);
                            }
                            let scratch = &mut scratch[..1usize << k];
                            for &i in &order[rs + 1..re] {
                                let i = i as usize;
                                expand_singleton_into(
                                    ntt,
                                    log_d,
                                    k,
                                    w,
                                    positions[i] & wmask,
                                    values[i],
                                    scratch,
                                );
                                transpose_top_only(dst, scratch);
                            }
                        }
                    } else {
                        dst.fill(F128::ZERO);
                        for &i in &order[rs..re] {
                            let i = i as usize;
                            dst[positions[i] & wmask] += values[i];
                        }
                        transpose_forward_ntt_window_dense_slice(ntt, log_d, k, w, dst);
                    }
                }
                if fuse_pass_a {
                    for layer in (split..top).rev() {
                        let nb = 1usize << (layer - split);
                        let block_size = 1usize << (log_d - layer);
                        let bsh = block_size >> 1;
                        for jb in 0..nb {
                            let t = ntt.twiddle(layer, (c << (layer - split)) + jb);
                            let (top_h, bot) =
                                chunk[jb * block_size..(jb + 1) * block_size].split_at_mut(bsh);
                            transpose_butterfly(top_h, bot, t);
                        }
                    }
                }
            });
        let dens_ms = _td.elapsed().as_secs_f64() * 1e3;
        (data, 0.0, alloc_ms, dens_ms)
    } else {
        let processed: Vec<(usize, Vec<F128>)> = if fill {
            runs.par_iter()
                .map_init(Vec::<F128>::new, |scratch, &(rs, re)| {
                    let first = order[rs] as usize;
                    let w = positions[first] >> k;
                    let nnz = re - rs;
                    if sparse_trie {
                        let mut buf = take_singleton_buf(k);
                        sparse_window::expand(
                            ntt, log_d, k, w, positions, values, &order[rs..re], &mut buf,
                        );
                        return (w, buf);
                    } else if singleton_on && nnz <= multi_max {
                        let mut buf = take_singleton_buf(k);
                        expand_singleton_into(
                            ntt,
                            log_d,
                            k,
                            w,
                            positions[first] & wmask,
                            values[first],
                            &mut buf,
                        );
                        if nnz > 1 {
                            if scratch.len() < 1usize << k {
                                *scratch = take_singleton_buf(k);
                            }
                            let scratch = &mut scratch[..1usize << k];
                            for &i in &order[rs + 1..re] {
                                let i = i as usize;
                                expand_singleton_into(
                                    ntt,
                                    log_d,
                                    k,
                                    w,
                                    positions[i] & wmask,
                                    values[i],
                                    scratch,
                                );
                                transpose_top_only(&mut buf, scratch);
                            }
                        }
                        return (w, buf);
                    }
                    let mut buf = vec![F128::ZERO; 1 << k];
                    for &i in &order[rs..re] {
                        let i = i as usize;
                        buf[positions[i] & wmask] += values[i];
                    }
                    transpose_forward_ntt_window_dense(ntt, log_d, k, w, buf)
                })
                .collect()
        } else {
            win_vec
                .into_par_iter()
                .map(|(w, buf)| transpose_forward_ntt_window_dense(ntt, log_d, k, w, buf))
                .collect()
        };
        let win_ms = _tw.elapsed().as_secs_f64() * 1e3;

        let _ta = std::time::Instant::now();
        let (data, alloc_ms, dens_ms) = if induce_fused_densify_enabled() {
            let slots = window_slots(n >> k, processed);
            let alloc_ms = _ta.elapsed().as_secs_f64() * 1e3;
            let _td = std::time::Instant::now();
            let data = densify_windows_fused(n, k, slots);
            (data, alloc_ms, _td.elapsed().as_secs_f64() * 1e3)
        } else {
            let mut data = vec![F128::ZERO; n];
            let alloc_ms = _ta.elapsed().as_secs_f64() * 1e3;
            let _td = std::time::Instant::now();
            for (w, buf) in processed {
                data[(w << k)..((w + 1) << k)].copy_from_slice(&buf);
            }
            (data, alloc_ms, _td.elapsed().as_secs_f64() * 1e3)
        };
        (data, win_ms, alloc_ms, dens_ms)
    };

    let _ts = std::time::Instant::now();
    transpose_forward_ntt_dense_layers_dead(ntt, &mut data, dense_top, dead_top.min(dense_top));
    if ot {
        eprintln!(
            "      [sparse-ntt] log_d={log_d} k={k} wins={nwin} win-phase {win_ms:.2} ms  alloc(zeroed 2^{log_d}) {alloc_ms:.2} ms  densify {dens_ms:.2} ms  dense({dl} layers) {ds:.2} ms  minflt +{mf}",
            nwin = nwins,
            win_ms = win_ms,
            dl = dense_top,
            ds = _ts.elapsed().as_secs_f64() * 1e3,
            mf = minor_faults() - mf0,
        );
    }
    (data, singleton_stats)
}


pub(crate) struct LigeroWitness {
    pub mat: Vec<F128>,
    pub tree: Vec<Hash>,
    pub block_len: usize,
    pub num_interleaved: usize,
}

impl Drop for LigeroWitness {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.mat));
        crate::pcs::commit::give_tree(std::mem::take(&mut self.tree));
    }
}

impl Drop for SumcheckProver {
    fn drop(&mut self) {
        if let FoldBuf::Owned(v) = std::mem::take(&mut self.f) {
            crate::scratch::give_f128(v);
        }
        if let FoldBuf::Owned(v) = std::mem::take(&mut self.combined_basis) {
            crate::scratch::give_f128(v);
        }
    }
}

impl LigeroWitness {
    #[inline]
    pub fn row(&self, pos: usize) -> &[F128] {
        let start = pos * self.num_interleaved;
        &self.mat[start..start + self.num_interleaved]
    }

    #[inline]
    pub fn root(&self) -> Hash {
        self.tree[self.tree.len() - 1]
    }
}

#[cfg(unix)]
fn minor_faults() -> u64 {
    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i64,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        ru_nswap: i64,
        ru_inblock: i64,
        ru_oublock: i64,
        ru_msgsnd: i64,
        ru_msgrcv: i64,
        ru_nsignals: i64,
        ru_nvcsw: i64,
        ru_nivcsw: i64,
    }
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut ru = std::mem::MaybeUninit::<Rusage>::uninit();
    let rc = unsafe { getrusage(0, ru.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    unsafe { ru.assume_init() }.ru_minflt as u64
}
#[cfg(not(unix))]
fn minor_faults() -> u64 {
    0
}

pub(crate) fn ligero_commit(
    poly: &[F128],
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    ntt: &AdditiveNttF128,
    kind: HashKind,
) -> LigeroWitness {
    let msg_cols = 1usize << log_msg_cols;
    let num_interleaved = 1usize << log_num_interleaved;
    let block_len = msg_cols << log_inv_rate;
    let log_block_len = log_msg_cols + log_inv_rate;
    assert_eq!(poly.len(), num_interleaved * msg_cols);
    assert!(log_block_len <= ntt.log_domain_size());

    let codeword_len = block_len * num_interleaved;
    let ot = open_timing();
    let t_alloc = std::time::Instant::now();
    let mut mat = crate::scratch::take_f128(codeword_len);
    let mat_alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;
    let leaf_size_bytes = num_interleaved * core::mem::size_of::<F128>();

    if crate::pcs::commit::lig_fused_commit_enabled() && !crate::gpu::merkle::available() {
        crate::rstrace::ev("lc:start", log_block_len as u32);
        let faults_before = if ot { minor_faults() } else { 0 };
        let mat_cap = mat.capacity();
        let t_encode = std::time::Instant::now();
        let mut tree = crate::pcs::commit::take_tree(2 * block_len - 1);
        let folded = crate::pcs::commit::fused_encode_leaves_subtree(
            ntt,
            poly,
            &mut mat,
            num_interleaved,
            &mut tree,
            block_len,
            leaf_size_bytes,
            kind,
        );
        let fused_ms = t_encode.elapsed().as_secs_f64() * 1e3;
        let t_upper = std::time::Instant::now();
        crate::rstrace::ev("lc:fused", log_block_len as u32);
        crate::pcs::commit::build_upper_levels(&mut tree, block_len, block_len >> folded, kind);
        crate::rstrace::ev("lc:end", log_block_len as u32);
        if ot {
            eprintln!(
                "[open-timing] ligero_commit: leaves=2^{log_block_len} leaf={leaf_size_bytes}B \
                 ({:.1} MiB) mat-alloc {mat_alloc_ms:.2} ms (cap {} F128, tree cap {}) FUSED \
                 encode+leaves+subtree({folded}) {fused_ms:.2} ms upper {:.2} ms minflt +{}",
                (codeword_len * 16) as f64 / (1024.0 * 1024.0),
                mat_cap,
                tree.capacity(),
                t_upper.elapsed().as_secs_f64() * 1e3,
                minor_faults().saturating_sub(faults_before),
            );
        }
        return LigeroWitness {
            mat,
            tree,
            block_len,
            num_interleaved,
        };
    }

    let t_encode = std::time::Instant::now();
    ntt.rs_encode_interleaved(poly, &mut mat, num_interleaved);
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1e3;
    let data_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            mat.as_ptr() as *const u8,
            mat.len() * core::mem::size_of::<F128>(),
        )
    };
    debug_assert_eq!(data_bytes.len(), block_len * leaf_size_bytes);
    let t_merkle = std::time::Instant::now();
    let mut gpu_busy_ms = 0.0f64;
    let gpu_tree = if kind == HashKind::Blake3
        && block_len >= gpu_open_merkle_min_leaves()
        && merkle::blake3_leaf_size_is_batchable(leaf_size_bytes)
        && crate::gpu::merkle::available()
    {
        gpu_merkle_tree_for_open(data_bytes, block_len, leaf_size_bytes, &mut gpu_busy_ms)
    } else {
        None
    };
    let on_gpu = gpu_tree.is_some();
    let mut tree_alloc_ms = 0.0f64;
    let mut leaves_ms = 0.0f64;
    let tree = gpu_tree.unwrap_or_else(|| {
        let t_ta = std::time::Instant::now();
        let mut tree = crate::pcs::commit::take_tree(2 * block_len - 1);
        tree_alloc_ms = t_ta.elapsed().as_secs_f64() * 1e3;
        if ot {
            let t_l = std::time::Instant::now();
            merkle::hash_leaves(data_bytes, leaf_size_bytes, &mut tree[..block_len], kind);
            leaves_ms = t_l.elapsed().as_secs_f64() * 1e3;
            crate::pcs::commit::build_upper_levels(&mut tree, block_len, block_len, kind);
        } else {
            merkle::fill_merkle_tree(&mut tree, data_bytes, block_len, kind);
        }
        tree
    });
    if ot {
        eprintln!(
            "[open-timing] ligero_commit: leaves=2^{log_block_len} leaf={leaf_size_bytes}B \
             ({:.1} MiB) mat-alloc {mat_alloc_ms:.2} ms encode {encode_ms:.2} ms merkle({}) {:.2} ms \
             [tree-alloc {tree_alloc_ms:.2} leaves {leaves_ms:.2}] (gpu busy {gpu_busy_ms:.2} ms)",
            (codeword_len * 16) as f64 / (1024.0 * 1024.0),
            if on_gpu { "gpu" } else { "cpu" },
            t_merkle.elapsed().as_secs_f64() * 1e3,
        );
    }

    LigeroWitness {
        mat,
        tree,
        block_len,
        num_interleaved,
    }
}

fn gpu_open_merkle_min_leaves() -> usize {
    static MIN: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_NO_GPU_OPEN_MERKLE").is_some() {
            return usize::MAX;
        }
        match std::env::var("FLOCK_GPU_OPEN_MERKLE_MIN_LOG2") {
            Ok(s) => s.parse::<u32>().map(|l| 1usize << l).unwrap_or(usize::MAX),
            Err(_) if std::env::var_os("FLOCK_GPU_OPEN_MERKLE").is_some() => 1usize << 18,
            Err(_) => usize::MAX,
        }
    });
    *MIN
}

const GPU_OPEN_STOP_NODES: usize = 1 << 10;

const GPU_OPEN_TREE_PAD_NODES: usize = 512;

fn gpu_merkle_tree_for_open(
    data_bytes: &[u8],
    block_len: usize,
    leaf_size: usize,
    busy_ms: &mut f64,
) -> Option<Vec<Hash>> {
    let total_nodes = 2 * block_len - 1;
    let mut tree: Vec<Hash> = crate::alloc_uninit_vec(total_nodes + GPU_OPEN_TREE_PAD_NODES);
    let mut session = unsafe {
        crate::gpu::merkle::begin(
            data_bytes,
            leaf_size,
            tree.as_mut_ptr(),
            tree.len(),
            GPU_OPEN_STOP_NODES,
        )
    }?;
    if !session.commit_leaves(0, block_len) || !session.commit_parent_levels() {
        session.finish();
        return None;
    }
    *busy_ms = session.finish()? * 1e3;
    let from_nodes = if GPU_OPEN_STOP_NODES <= block_len / 2 {
        GPU_OPEN_STOP_NODES
    } else {
        block_len
    };
    crate::pcs::commit::build_upper_levels(&mut tree, block_len, from_nodes, HashKind::Blake3);
    tree.truncate(total_nodes);
    Some(tree)
}


#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckMessage {
    pub u_0: F128,
    pub u_2: F128,
}

#[derive(Clone, Copy, Debug)]
struct RoundQuad {
    c: F128, // u_0
    b: F128, // u_1 (X coeff) — derived from T_r and u_2
    a: F128, // u_2 (X² coeff)
}

impl RoundQuad {
    #[inline]
    fn from_msg(msg: SumcheckMessage, t_r: F128) -> Self {
        Self {
            c: msg.u_0,
            b: t_r + msg.u_2,
            a: msg.u_2,
        }
    }
    #[inline]
    fn eval(&self, r: F128) -> F128 {
        self.c + r * self.b + r * r * self.a
    }
    #[inline]
    fn fold(p1: &Self, p2: &Self, alpha: F128) -> Self {
        Self {
            c: p1.c + alpha * p2.c,
            b: p1.b + alpha * p2.b,
            a: p1.a + alpha * p2.a,
        }
    }
}

fn round_msg_lsb(f: &[F128], b: &[F128]) -> SumcheckMessage {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    {
        const PAR_THRESHOLD: usize = 4096;
        if n < PAR_THRESHOLD {
            let (u_0, u_2) = unsafe { msg_reduce_avx512(f, b) };
            return SumcheckMessage { u_0, u_2 };
        }
        const CHUNK: usize = 1024;
        let (u_0, u_2) = f
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(fc, bc)| {
                unsafe { msg_reduce_avx512(fc, bc) }
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );
        return SumcheckMessage { u_0, u_2 };
    }

    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
        const PAR_THRESHOLD: usize = 4096;
        let half = n / 2;
        if half < PAR_THRESHOLD {
            let mut u_0 = F128::ZERO;
            let mut u_2 = F128::ZERO;
            for j in 0..half {
                let f0 = f[2 * j];
                let f1 = f[2 * j + 1];
                let b0 = b[2 * j];
                let b1 = b[2 * j + 1];
                u_0 += f0 * b0;
                u_2 += (f0 + f1) * (b0 + b1);
            }
            return SumcheckMessage { u_0, u_2 };
        }

        let (u_0, u_2) = (0..half)
            .into_par_iter()
            .with_min_len(PAR_THRESHOLD / 4)
            .map(|j| {
                let f0 = f[2 * j];
                let f1 = f[2 * j + 1];
                let b0 = b[2 * j];
                let b1 = b[2 * j + 1];
                (f0 * b0, (f0 + f1) * (b0 + b1))
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            );
        SumcheckMessage { u_0, u_2 }
    }
}

fn round_msg_and_eval_lsb(f: &[F128], b: &[F128]) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);

    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    if open_ood_x4_enabled() {
        const PAR_THRESHOLD: usize = 4096;
        if n < PAR_THRESHOLD {
            let (u_0, u_2, y) = unsafe { msg_reduce_eval_avx512(f, b) };
            return (SumcheckMessage { u_0, u_2 }, y);
        }
        const CHUNK: usize = 1024;
        let (u_0, u_2, y) = f
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(fc, bc)| {
                unsafe { msg_reduce_eval_avx512(fc, bc) }
            })
            .reduce(
                || (F128::ZERO, F128::ZERO, F128::ZERO),
                |(a0, a2, ay), (c0, c2, cy)| (a0 + c0, a2 + c2, ay + cy),
            );
        return (SumcheckMessage { u_0, u_2 }, y);
    }

    const PAR_THRESHOLD: usize = 4096;
    let half = n / 2;
    let term = |j: usize| -> (F128, F128, F128) {
        let f0 = f[2 * j];
        let f1 = f[2 * j + 1];
        let b0 = b[2 * j];
        let b1 = b[2 * j + 1];
        let e0 = f0 * b0;
        (e0, (f0 + f1) * (b0 + b1), e0 + f1 * b1)
    };
    if half < PAR_THRESHOLD {
        let (mut u_0, mut u_2, mut y) = (F128::ZERO, F128::ZERO, F128::ZERO);
        for j in 0..half {
            let (a0, a2, ay) = term(j);
            u_0 += a0;
            u_2 += a2;
            y += ay;
        }
        return (SumcheckMessage { u_0, u_2 }, y);
    }

    let (u_0, u_2, y) = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(term)
        .reduce(
            || (F128::ZERO, F128::ZERO, F128::ZERO),
            |(a0, a2, ay), (b0, b2, by)| (a0 + b0, a2 + b2, ay + by),
        );
    (SumcheckMessage { u_0, u_2 }, y)
}

const LAZY_OOD_EQ_SPLIT_LOW_LOG: usize = 11;

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn factorized_eq_round0_avx512(f: &[F128], weights: &[F128]) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    unsafe {
        debug_assert_eq!(f.len(), 2 * weights.len());
        let mut a_acc = WideGhashX4::zero();
        let mut s_acc = WideGhashX4::zero();
        let lanes = weights.len() & !3;
        let mut j = 0usize;
        while j < lanes {
            let f0 = _mm512_loadu_si512(f.as_ptr().add(2 * j) as *const __m512i);
            let f1 = _mm512_loadu_si512(f.as_ptr().add(2 * j + 4) as *const __m512i);
            let w = _mm512_loadu_si512(weights.as_ptr().add(j) as *const __m512i);
            let even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let f0_sum = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1_sum = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let sum = _mm512_shuffle_i32x4::<0x88>(f0_sum, f1_sum);
            a_acc.mul_acc(even, w);
            s_acc.mul_acc(sum, w);
            j += 4;
        }
        let mut a = a_acc.fold().reduce();
        let mut s = s_acc.fold().reduce();
        while j < weights.len() {
            a += f[2 * j] * weights[j];
            s += (f[2 * j] + f[2 * j + 1]) * weights[j];
            j += 1;
        }
        (a, s)
    }
}

#[inline]
fn factorized_eq_round0(f: &[F128], weights: &[F128]) -> (F128, F128) {
    assert_eq!(f.len(), 2 * weights.len());
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    unsafe {
        factorized_eq_round0_avx512(f, weights)
    }
    #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
    {
        let mut a = F128::ZERO;
        let mut s = F128::ZERO;
        for (pair, &weight) in f.chunks_exact(2).zip(weights) {
            a += pair[0] * weight;
            s += (pair[0] + pair[1]) * weight;
        }
        (a, s)
    }
}

fn round_msg_and_eval_lsb_factorized_eq(
    f: &[F128],
    eq_lo: &[F128],
    eq_hi: &[F128],
    z_0: F128,
) -> (SumcheckMessage, F128) {
    use rayon::prelude::*;

    assert!(eq_lo.len().is_power_of_two() && eq_lo.len() >= 2);
    assert!(eq_hi.len().is_power_of_two());
    let tail_len = eq_lo
        .len()
        .checked_mul(eq_hi.len())
        .expect("factorized OOD tail overflow");
    assert_eq!(f.len(), 2 * tail_len);

    let (a, s) = f
        .par_chunks(2 * eq_lo.len())
        .zip(eq_hi.par_iter())
        .map(|(chunk, &hi)| {
            let (a, s) = factorized_eq_round0(chunk, eq_lo);
            (a * hi, s * hi)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, s0), (a1, s1)| (a0 + a1, s0 + s1),
        );
    (
        SumcheckMessage {
            u_0: (F128::ONE + z_0) * a,
            u_2: s,
        },
        a + z_0 * s,
    )
}

#[cfg(test)]
fn partial_eval_lsb_one(evals: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let n = evals.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let one_plus_r = F128::ONE + r;

    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        for j in 0..half {
            let v0 = evals[2 * j];
            let v1 = evals[2 * j + 1];
            evals[j] = v0 * one_plus_r + v1 * r;
        }
        evals.truncate(half);
        return;
    }

    let folded: Vec<F128> = (0..half)
        .into_par_iter()
        .with_min_len(PAR_THRESHOLD / 4)
        .map(|j| evals[2 * j] * one_plus_r + evals[2 * j + 1] * r)
        .collect();
    *evals = folded;
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn msg_reduce_avx512(fc: &[F128], bc: &[F128]) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    let lanes = len & !7;
    unsafe {
        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();

        let mut k = 0;
        while k < lanes {
            let f0 = _mm512_loadu_si512(fc.as_ptr().add(k) as *const __m512i);
            let f1 = _mm512_loadu_si512(fc.as_ptr().add(k + 4) as *const __m512i);
            let b0 = _mm512_loadu_si512(bc.as_ptr().add(k) as *const __m512i);
            let b1 = _mm512_loadu_si512(bc.as_ptr().add(k + 4) as *const __m512i);

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            k += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();

        while k + 1 < len {
            let f0 = fc[k];
            let f1 = fc[k + 1];
            let b0 = bc[k];
            let b1 = bc[k + 1];
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            k += 2;
        }

        (u0, u2)
    }
}

/// Lane opus-pcs: the vector body of [`msg_reduce_avx512`] over one
/// eight-aligned span, accumulating into caller-held raw Karatsuba streams
/// (no fold/reduce). Feeding consecutive spans of a block in order and then
/// folding once reproduces `msg_reduce_avx512` on the whole block exactly:
/// same products, same accumulator sequence.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
#[inline]
#[allow(dead_code)] // Consumed only by the ranked AVX-512 GFNI route.
unsafe fn msg_accumulate_avx512(
    fc: &[F128],
    bc: &[F128],
    u0_acc: &mut crate::field::gf2_128::x86_64::WideGhashX4,
    u2_acc: &mut crate::field::gf2_128::x86_64::WideGhashX4,
) {
    use core::arch::x86_64::*;
    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(8));
    unsafe {
        let mut k = 0;
        while k < len {
            let f0 = _mm512_loadu_si512(fc.as_ptr().add(k) as *const __m512i);
            let f1 = _mm512_loadu_si512(fc.as_ptr().add(k + 4) as *const __m512i);
            let b0 = _mm512_loadu_si512(bc.as_ptr().add(k) as *const __m512i);
            let b1 = _mm512_loadu_si512(bc.as_ptr().add(k + 4) as *const __m512i);

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            k += 8;
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn msg_reduce_eval_avx512(fc: &[F128], bc: &[F128]) -> (F128, F128, F128) {
    use crate::field::gf2_128::x86_64::WideGhashX4;
    use core::arch::x86_64::*;

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    let lanes = len & !7;
    unsafe {
        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();
        let mut y_acc = WideGhashX4::zero();

        let mut k = 0;
        while k < lanes {
            let f0 = _mm512_loadu_si512(fc.as_ptr().add(k) as *const __m512i);
            let f1 = _mm512_loadu_si512(fc.as_ptr().add(k + 4) as *const __m512i);
            let b0 = _mm512_loadu_si512(bc.as_ptr().add(k) as *const __m512i);
            let b1 = _mm512_loadu_si512(bc.as_ptr().add(k + 4) as *const __m512i);

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0s = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1s = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0s, f1s);
            let b0s = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1s = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0s, b1s);
            u2_acc.mul_acc(f_sum, b_sum);

            y_acc.mul_acc(f0, b0);
            y_acc.mul_acc(f1, b1);

            k += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();
        let mut y = y_acc.fold().reduce();

        while k + 1 < len {
            let f0 = fc[k];
            let f1 = fc[k + 1];
            let b0 = bc[k];
            let b1 = bc[k + 1];
            let e0 = f0 * b0;
            u0 += e0;
            u2 += (f0 + f1) * (b0 + b1);
            y += e0 + f1 * b1;
            k += 2;
        }

        (u0, u2, y)
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn glue_block_x4(acc: &mut [F128], src: &[F128], alpha: F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, ghash_shift64_x4};
    use core::arch::x86_64::*;
    debug_assert_eq!(acc.len(), src.len());
    unsafe {
        let ab = _mm512_broadcast_i32x4(_mm_set_epi64x(alpha.hi as i64, alpha.lo as i64));
        let ab_x64 = ghash_shift64_x4(ab);
        let lanes = acc.len() & !3;
        let mut i = 0usize;
        while i < lanes {
            let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
            let v = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
            _mm512_storeu_si512(
                acc.as_mut_ptr().add(i) as *mut __m512i,
                _mm512_xor_si512(a, ghash_mul_x4_split(v, ab, ab_x64)),
            );
            i += 4;
        }
        while i < acc.len() {
            acc[i] += alpha * src[i];
            i += 1;
        }
    }
}

#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
fn open_ood_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_OOD_X4").is_none());
    *ON
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
unsafe fn fold_and_msg_chunk_x86<const CORR: bool>(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
    r_x4: core::arch::x86_64::__m512i,
    r_x64: core::arch::x86_64::__m512i,
    stream: bool,
    deferred: Option<(&[F128], F128)>,
    ood: Option<(&[F128], F128)>,
) -> (F128, F128) {
    use crate::field::gf2_128::x86_64::{ghash_mul_x4_split, WideGhashX4};
    use core::arch::x86_64::*;

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(2));
    if CORR {
        if let Some((basis, _)) = deferred {
            debug_assert!(basis.len() >= 2 * (base + len));
        }
        if let Some((eq_lo, _)) = ood {
            debug_assert!(eq_lo.len() >= len);
        }
    }

    unsafe {
        let mut u0_acc = WideGhashX4::zero();
        let mut u2_acc = WideGhashX4::zero();
        let f_ptr = f.as_ptr();
        let b_ptr = b.as_ptr();
        let fc_ptr = fc.as_mut_ptr();
        let bc_ptr = bc.as_mut_ptr();
        let dst_aligned =
            (fc_ptr as usize).is_multiple_of(16) && (bc_ptr as usize).is_multiple_of(16);

        let deferred_vec = if CORR {
            use crate::field::gf2_128::x86_64::ghash_shift64_x4;
            deferred.map(|(basis, alpha)| {
                let ax = _mm512_broadcast_i32x4(_mm_set_epi64x(alpha.hi as i64, alpha.lo as i64));
                (basis.as_ptr(), ax, ghash_shift64_x4(ax))
            })
        } else {
            None
        };
        let ood_vec = if CORR {
            use crate::field::gf2_128::x86_64::ghash_shift64_x4;
            ood.map(|(eq_lo, scale)| {
                let sx = _mm512_broadcast_i32x4(_mm_set_epi64x(scale.hi as i64, scale.lo as i64));
                (eq_lo.as_ptr(), sx, ghash_shift64_x4(sx))
            })
        } else {
            None
        };

        let fold4 = |ptr: *const F128, source: usize| -> __m512i {
            let lo = _mm512_loadu_si512(ptr.add(source) as *const __m512i);
            let hi = _mm512_loadu_si512(ptr.add(source + 4) as *const __m512i);
            let even = _mm512_shuffle_i32x4::<0x88>(lo, hi);
            let odd = _mm512_shuffle_i32x4::<0xDD>(lo, hi);
            _mm512_xor_si512(
                even,
                ghash_mul_x4_split(_mm512_xor_si512(even, odd), r_x4, r_x64),
            )
        };
        let store4 = |value: __m512i, ptr: *mut F128| {
            if stream && dst_aligned {
                _mm_stream_si128(
                    ptr.cast::<__m128i>(),
                    _mm512_extracti32x4_epi32::<0>(value),
                );
                _mm_stream_si128(
                    ptr.add(1).cast::<__m128i>(),
                    _mm512_extracti32x4_epi32::<1>(value),
                );
                _mm_stream_si128(
                    ptr.add(2).cast::<__m128i>(),
                    _mm512_extracti32x4_epi32::<2>(value),
                );
                _mm_stream_si128(
                    ptr.add(3).cast::<__m128i>(),
                    _mm512_extracti32x4_epi32::<3>(value),
                );
            } else {
                _mm512_storeu_si512(ptr.cast::<__m512i>(), value);
            }
        };

        let defer_one = if CORR {
            deferred.and_then(|(basis, alpha)| {
                (alpha == F128::ONE && open_defer_one_xor_enabled()).then_some(basis.as_ptr())
            })
        } else {
            None
        };
        let fold4_sum = |p: *const F128, q: *const F128, source: usize| -> __m512i {
            let lo = _mm512_xor_si512(
                _mm512_loadu_si512(p.add(source) as *const __m512i),
                _mm512_loadu_si512(q.add(source) as *const __m512i),
            );
            let hi = _mm512_xor_si512(
                _mm512_loadu_si512(p.add(source + 4) as *const __m512i),
                _mm512_loadu_si512(q.add(source + 4) as *const __m512i),
            );
            let even = _mm512_shuffle_i32x4::<0x88>(lo, hi);
            let odd = _mm512_shuffle_i32x4::<0xDD>(lo, hi);
            _mm512_xor_si512(
                even,
                ghash_mul_x4_split(_mm512_xor_si512(even, odd), r_x4, r_x64),
            )
        };

        let mut t = 0usize;
        while t + 8 <= len {
            let f0 = fold4(f_ptr, 2 * (base + t));
            let f1 = fold4(f_ptr, 2 * (base + t + 4));
            let (mut b0, mut b1) = match defer_one {
                Some(dptr) => (
                    fold4_sum(b_ptr, dptr, 2 * (base + t)),
                    fold4_sum(b_ptr, dptr, 2 * (base + t + 4)),
                ),
                None => (fold4(b_ptr, 2 * (base + t)), fold4(b_ptr, 2 * (base + t + 4))),
            };
            if CORR {
                if let (None, Some((dptr, ax, ax64))) = (defer_one, deferred_vec) {
                    b0 = _mm512_xor_si512(
                        b0,
                        ghash_mul_x4_split(fold4(dptr, 2 * (base + t)), ax, ax64),
                    );
                    b1 = _mm512_xor_si512(
                        b1,
                        ghash_mul_x4_split(fold4(dptr, 2 * (base + t + 4)), ax, ax64),
                    );
                }
                if let Some((eptr, sx, sx64)) = ood_vec {
                    let e0 = _mm512_loadu_si512(eptr.add(t) as *const __m512i);
                    let e1 = _mm512_loadu_si512(eptr.add(t + 4) as *const __m512i);
                    b0 = _mm512_xor_si512(b0, ghash_mul_x4_split(e0, sx, sx64));
                    b1 = _mm512_xor_si512(b1, ghash_mul_x4_split(e1, sx, sx64));
                }
            }

            let f_even = _mm512_shuffle_i32x4::<0x88>(f0, f1);
            let b_even = _mm512_shuffle_i32x4::<0x88>(b0, b1);
            u0_acc.mul_acc(f_even, b_even);

            let f0_sum = _mm512_xor_si512(f0, _mm512_shuffle_i32x4::<0xB1>(f0, f0));
            let f1_sum = _mm512_xor_si512(f1, _mm512_shuffle_i32x4::<0xB1>(f1, f1));
            let f_sum = _mm512_shuffle_i32x4::<0x88>(f0_sum, f1_sum);
            let b0_sum = _mm512_xor_si512(b0, _mm512_shuffle_i32x4::<0xB1>(b0, b0));
            let b1_sum = _mm512_xor_si512(b1, _mm512_shuffle_i32x4::<0xB1>(b1, b1));
            let b_sum = _mm512_shuffle_i32x4::<0x88>(b0_sum, b1_sum);
            u2_acc.mul_acc(f_sum, b_sum);

            store4(f0, fc_ptr.add(t));
            store4(f1, fc_ptr.add(t + 4));
            store4(b0, bc_ptr.add(t));
            store4(b1, bc_ptr.add(t + 4));
            t += 8;
        }

        let mut u0 = u0_acc.fold().reduce();
        let mut u2 = u2_acc.fold().reduce();
        while t + 1 < len {
            let source = 2 * (base + t);
            let f0 = *f_ptr.add(source) + r * (*f_ptr.add(source) + *f_ptr.add(source + 1));
            let f1 = *f_ptr.add(source + 2) + r * (*f_ptr.add(source + 2) + *f_ptr.add(source + 3));
            let mut b0 = *b_ptr.add(source) + r * (*b_ptr.add(source) + *b_ptr.add(source + 1));
            let mut b1 = *b_ptr.add(source + 2) + r * (*b_ptr.add(source + 2) + *b_ptr.add(source + 3));
            if CORR {
                if let Some((basis, alpha)) = deferred {
                    let dptr = basis.as_ptr();
                    let d0 = *dptr.add(source) + r * (*dptr.add(source) + *dptr.add(source + 1));
                    let d1 =
                        *dptr.add(source + 2) + r * (*dptr.add(source + 2) + *dptr.add(source + 3));
                    b0 += alpha * d0;
                    b1 += alpha * d1;
                }
                if let Some((eq_lo, scale)) = ood {
                    b0 += scale * *eq_lo.as_ptr().add(t);
                    b1 += scale * *eq_lo.as_ptr().add(t + 1);
                }
            }
            *fc_ptr.add(t) = f0;
            *fc_ptr.add(t + 1) = f1;
            *bc_ptr.add(t) = b0;
            *bc_ptr.add(t + 1) = b1;
            u0 += f0 * b0;
            u2 += (f0 + f1) * (b0 + b1);
            t += 2;
        }
        if stream && dst_aligned {
            _mm_sfence();
        }
        (u0, u2)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn open_nt_min_half_x86() -> usize {
    static MIN: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if std::env::var_os("FLOCK_OPEN_NT_17").is_some() {
            1usize << 17
        } else {
            1usize << 21
        }
    });
    *MIN
}

fn open_defer_one_xor_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPEN_DEFER_ONE_XOR").is_none());
    *ON
}

fn fold_and_msg_lsb(
    f: &[F128],
    b: &[F128],
    r: F128,
    arena: Option<&mut FoldArena>,
) -> (FoldBuf, FoldBuf, SumcheckMessage) {
    fold_and_msg_lsb_inner(f, b, r, arena, None, None)
}

fn fold_and_msg_lsb_inner(
    f: &[F128],
    b: &[F128],
    r: F128,
    arena: Option<&mut FoldArena>,
    lazy_ood: Option<(&[F128], &[F128], F128)>,
    deferred_basis: Option<(&[F128], F128)>,
) -> (FoldBuf, FoldBuf, SumcheckMessage) {
    use rayon::prelude::*;
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    debug_assert_eq!(b.len(), n);
    let half = n / 2;
    if let Some((eq_lo, eq_hi, _)) = lazy_ood {
        assert!(eq_lo.len().is_power_of_two() && eq_lo.len() >= 2);
        assert!(eq_hi.len().is_power_of_two());
        assert_eq!(half, eq_lo.len() * eq_hi.len());
    }
    if let Some((basis, _)) = deferred_basis {
        assert_eq!(basis.len(), n, "deferred basis fold shape changed");
    }
    const PAR_THRESHOLD: usize = 4096;
    if half < PAR_THRESHOLD {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        if half >= 2 && lazy_ood.is_none_or(|(eq_lo, eq_hi, _)| eq_hi.len() == 1 && eq_lo.len() >= half) {
            let mut nf = crate::alloc_uninit_f128_vec(half);
            let mut nb = crate::alloc_uninit_f128_vec(half);
            let (u_0, u_2) = unsafe {
                use crate::field::gf2_128::x86_64::ghash_shift64_x4;
                use core::arch::x86_64::*;
                let r_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
                let r_x64 = ghash_shift64_x4(r_x4);
                if lazy_ood.is_none() && deferred_basis.is_none() {
                    fold_and_msg_chunk_x86::<false>(
                        f, b, 0, &mut nf, &mut nb, r, r_x4, r_x64, false, None, None,
                    )
                } else {
                    let ood = lazy_ood.map(|(eq_lo, eq_hi, gamma)| (eq_lo, gamma * eq_hi[0]));
                    fold_and_msg_chunk_x86::<true>(
                        f,
                        b,
                        0,
                        &mut nf,
                        &mut nb,
                        r,
                        r_x4,
                        r_x64,
                        false,
                        deferred_basis,
                        ood,
                    )
                }
            };
            return (
                FoldBuf::Owned(nf),
                FoldBuf::Owned(nb),
                SumcheckMessage { u_0, u_2 },
            );
        }
        let mut nf = Vec::with_capacity(half);
        let mut nb = Vec::with_capacity(half);
        let defer_one = deferred_basis
            .filter(|&(_, alpha)| alpha == F128::ONE && open_defer_one_xor_enabled())
            .map(|(basis, _)| basis);
        for j in 0..half {
            let f0 = f[2 * j];
            let f1 = f[2 * j + 1];
            let b0 = b[2 * j];
            let b1 = b[2 * j + 1];
            nf.push(f0 + r * (f0 + f1));
            if let Some(basis) = defer_one {
                let s0 = b0 + basis[2 * j];
                let s1 = b1 + basis[2 * j + 1];
                nb.push(s0 + r * (s0 + s1));
                continue;
            }
            let mut folded_b = b0 + r * (b0 + b1);
            if let Some((basis, alpha)) = deferred_basis {
                let d0 = basis[2 * j];
                let d1 = basis[2 * j + 1];
                folded_b += alpha * (d0 + r * (d0 + d1));
            }
            nb.push(folded_b);
        }
        if let Some((eq_lo, eq_hi, gamma)) = lazy_ood {
            for (j, value) in nb.iter_mut().enumerate() {
                *value += gamma * eq_hi[j / eq_lo.len()] * eq_lo[j % eq_lo.len()];
            }
        }
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let (u_0, u_2) = unsafe { msg_reduce_avx512(&nf, &nb) };
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        let (u_0, u_2) = {
            let mut u_0 = F128::ZERO;
            let mut u_2 = F128::ZERO;
            let mut k = 0;
            while k + 1 < half {
                let f0 = nf[k];
                let f1 = nf[k + 1];
                let b0 = nb[k];
                let b1 = nb[k + 1];
                u_0 += f0 * b0;
                u_2 += (f0 + f1) * (b0 + b1);
                k += 2;
            }
            (u_0, u_2)
        };
        return (
            FoldBuf::Owned(nf),
            FoldBuf::Owned(nb),
            SumcheckMessage { u_0, u_2 },
        );
    }

    const CHUNK: usize = 2048;
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_nt = lazy_ood.is_none()
        && deferred_basis.is_none()
        && half >= (1usize << 21)
        && !crate::sl2_env_set!("FLOCK_NO_OPEN_NT");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let use_nt = lazy_ood.is_none()
        && deferred_basis.is_none()
        && half >= open_nt_min_half_x86()
        && !crate::sl2_env_set!("FLOCK_NO_OPEN_NT");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let (r_x4, r_x64) = unsafe {
        use crate::field::gf2_128::x86_64::ghash_shift64_x4;
        use core::arch::x86_64::*;
        let r_x4 = _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
        (r_x4, ghash_shift64_x4(r_x4))
    };

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_soa =
        lazy_ood.is_none() && deferred_basis.is_none() && cfg!(target_feature = "sha3") && {
            static SOA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                std::env::var_os("FLOCK_NO_OPEN_SUMCHECK_OPT").is_none()
            });
            *SOA
        };
    let (mut nf, mut nb) = match arena.and_then(|a| a.carve_pair(half)) {
        Some(pair) => pair,
        None => {
            #[cfg(target_arch = "x86_64")]
            {
                (
                    FoldBuf::Owned(crate::scratch::take_f128(half)),
                    FoldBuf::Owned(crate::scratch::take_f128(half)),
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                (
                    FoldBuf::Owned(crate::alloc_uninit_f128_vec(half)),
                    FoldBuf::Owned(crate::alloc_uninit_f128_vec(half)),
                )
            }
        }
    };
    let (nf_s, nb_s): (&mut [F128], &mut [F128]) = (&mut nf, &mut nb);
    if let Some((eq_lo, _, _)) = lazy_ood {
        assert_eq!(eq_lo.len(), CHUNK, "ranked lazy OOD chunk width changed");
    }
    let (u_0, u_2) = nf_s
        .par_chunks_mut(CHUNK)
        .zip(nb_s.par_chunks_mut(CHUNK))
        .enumerate()
        .map(|(ci, (fc, bc))| {
            let base = ci * CHUNK;
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            {
                if lazy_ood.is_none() && deferred_basis.is_none() {
                    return unsafe {
                        fold_and_msg_chunk_x86::<false>(
                            f, b, base, fc, bc, r, r_x4, r_x64, use_nt, None, None,
                        )
                    };
                }
                return unsafe {
                    let ood = lazy_ood.map(|(eq_lo, eq_hi, gamma)| (eq_lo, gamma * eq_hi[ci]));
                    fold_and_msg_chunk_x86::<true>(
                        f,
                        b,
                        base,
                        fc,
                        bc,
                        r,
                        r_x4,
                        r_x64,
                        false,
                        deferred_basis,
                        ood,
                    )
                };
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                if use_soa {
                    return unsafe {
                        if use_nt {
                            fold_and_msg_chunk_nt_neon_soa::<true>(f, b, base, fc, bc, r)
                        } else {
                            fold_and_msg_chunk_nt_neon_soa::<false>(f, b, base, fc, bc, r)
                        }
                    };
                }
                if use_nt {
                    return unsafe { fold_and_msg_chunk_nt_neon(f, b, base, fc, bc, r) };
                }

            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            {
                let len = fc.len();
                crate::field::f128_slice::fold_pairs(f, base, fc, r);
                if let Some((basis, alpha)) = deferred_basis {
                    crate::field::f128_slice::fold_pairs_with_scaled_addend(
                        b, basis, base, bc, r, alpha,
                    );
                } else {
                    crate::field::f128_slice::fold_pairs(b, base, bc, r);
                }
                if let Some((eq_lo, eq_hi, gamma)) = lazy_ood {
                    let scale = gamma * eq_hi[ci];
                    crate::field::f128_slice::add_scaled(bc, &eq_lo[..len], scale);
                }
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    let (u0, u2) = unsafe { msg_reduce_avx512(fc, bc) };
                    (u0, u2)
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    let mut k = 0;
                    while k + 1 < len {
                        let f0 = fc[k];
                        let f1 = fc[k + 1];
                        let b0 = bc[k];
                        let b1 = bc[k + 1];
                        u0 += f0 * b0;
                        u2 += (f0 + f1) * (b0 + b1);
                        k += 2;
                    }
                    (u0, u2)
                }
            }
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(a0, a2), (c0, c2)| (a0 + c0, a2 + c2),
        );
    (nf, nb, SumcheckMessage { u_0, u_2 })
}

#[inline]
fn eval_lookahead(coeffs: &[F128; 6], challenge: F128) -> SumcheckMessage {
    let challenge_sq = challenge * challenge;
    SumcheckMessage {
        u_0: coeffs[0] + coeffs[1] * challenge + coeffs[2] * challenge_sq,
        u_2: coeffs[3] + coeffs[4] * challenge + coeffs[5] * challenge_sq,
    }
}

#[inline]
fn eval_quadratic_tensor(coefficients: &[F128], challenges: &[F128]) -> F128 {
    debug_assert_eq!(coefficients.len(), 3usize.pow(challenges.len() as u32));
    coefficients
        .iter()
        .enumerate()
        .fold(F128::ZERO, |sum, (mut index, &coefficient)| {
            let mut weight = F128::ONE;
            for &challenge in challenges.iter().rev() {
                weight *= match index % 3 {
                    0 => F128::ONE,
                    1 => challenge,
                    2 => challenge * challenge,
                    _ => unreachable!(),
                };
                index /= 3;
            }
            sum + coefficient * weight
        })
}

#[inline]
fn eval_fold4_lookahead2(
    coefficients: &super::Fold4Lookahead2,
    r0: F128,
    r1: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..9], &[r0, r1]),
        u_2: eval_quadratic_tensor(&coefficients[9..], &[r0, r1]),
    }
}

#[inline]
fn eval_fold4_lookahead3(
    coefficients: &super::Fold4Lookahead3,
    r0: F128,
    r1: F128,
    r2: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..27], &[r0, r1, r2]),
        u_2: eval_quadratic_tensor(&coefficients[27..], &[r0, r1, r2]),
    }
}

#[cfg(target_arch = "x86_64")]
fn mdf4_pf_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_MDF4_PF").is_none());
    *ON
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn direct_fold4_b_gfni_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD4_B_GFNI").is_none());
    *ON
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )
))]
#[inline]
fn direct_fold4_b_gfni_shape(claims_len: usize, has_ordinary: bool, block_len: usize) -> bool {
    claims_len == 2 && !has_ordinary && block_len.is_multiple_of(64)
}

#[inline]
#[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
fn eval_fold8_lookahead4(
    coefficients: &super::Fold8Lookahead4,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..81], &[r0, r1, r2, r3]),
        u_2: eval_quadratic_tensor(&coefficients[81..], &[r0, r1, r2, r3]),
    }
}

#[inline]
#[allow(dead_code)] // Reserved for the rollback DirectFold8 lookahead path.
fn eval_fold8_lookahead5(
    coefficients: &super::Fold8Lookahead5,
    r0: F128,
    r1: F128,
    r2: F128,
    r3: F128,
    r4: F128,
) -> SumcheckMessage {
    SumcheckMessage {
        u_0: eval_quadratic_tensor(&coefficients[..243], &[r0, r1, r2, r3, r4]),
        u_2: eval_quadratic_tensor(&coefficients[243..], &[r0, r1, r2, r3, r4]),
    }
}

fn materialize_direct_fold4(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold4Factors],
    challenges: [F128; 4],
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(16));
    let [r0, r1, r2, r3] = challenges;

    let fold_weight: [F128; 16] = std::array::from_fn(|bank| {
        let mut weight = F128::ONE;
        for (bit, &challenge) in challenges.iter().enumerate() {
            weight *= if (bank >> bit) & 1 == 0 {
                F128::ONE + challenge
            } else {
                challenge
            };
        }
        weight
    });
    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold4_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 16;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && out_len == block_len * claim.eq_hi.len()
    }));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let b_gfni_on = direct_fold4_b_gfni_enabled()
        && direct_fold4_b_gfni_shape(claims.len(), has_ordinary, block_len);
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let b_gfni_on = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_rows: Vec<(Vec<u64>, Vec<u64>)> = if b_gfni_on {
        claims
            .par_iter()
            .map(|claim| {
                (
                    claim.eq_lo.iter().map(|x| x.lo).collect(),
                    claim.eq_lo.iter().map(|x| x.hi).collect(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_mats: Vec<super::ring_switch::GfniDirectFoldMap> = if b_gfni_on {
        direct_tables
            .par_iter()
            .map(|table| super::ring_switch::build_gfni_direct_fold_map_from_table(table))
            .collect()
    } else {
        Vec::new()
    };

    let table_len = super::ring_switch::FOLD_TABLE_TOTAL;
    const SUB: usize = 256;
    let deferred_reduce = super::fold_deferred_reduce_enabled();
    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let mid_len = if has_ordinary || !deferred_reduce {
        4 * SUB
    } else {
        0
    };
    #[cfg(target_arch = "x86_64")]
    let pf_on = mdf4_pf_enabled();
    #[cfg(target_arch = "x86_64")]
    let n_blocks = out_len / block_len;
    let (u_0, u_2) = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            || {
                (
                    vec![F128::ZERO; if b_gfni_on { 0 } else { table_len }],
                    vec![F128::ZERO; mid_len],
                )
            },
            |(scratch, mid), (block, (b_out, f_out))| {
                let start = 16 * block * block_len;
                let f_in = &packed_witness[start..start + 16 * block_len];
                #[cfg(target_arch = "x86_64")]
                let (pf_base, pf_span) = if pf_on && block + 1 < n_blocks {
                    let p = unsafe { packed_witness.as_ptr().add(start + 16 * block_len) };
                    (
                        p.cast::<u8>(),
                        16 * block_len * core::mem::size_of::<F128>(),
                    )
                } else {
                    (core::ptr::null::<u8>(), 0usize)
                };
                #[cfg(target_arch = "x86_64")]
                let mut pf_at = 0usize;
                if deferred_reduce {
                    crate::field::f128_slice::fold16_banked(f_in, f_out, &fold_weight);
                } else {
                    let mut slot = 0usize;
                    while slot < block_len {
                        let n = SUB.min(block_len - slot);
                        let mid = &mut mid[..4 * n];
                        crate::field::f128_slice::fold4_nested(
                            &f_in[16 * slot..16 * (slot + n)],
                            mid,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(
                            mid,
                            &mut f_out[slot..slot + n],
                            r2,
                            r3,
                        );
                        slot += n;
                    }
                }
                if has_ordinary {
                    let b_in = &ordinary_basis[start..start + 16 * block_len];
                    let mut slot = 0usize;
                    while slot < block_len {
                        let n = SUB.min(block_len - slot);
                        let mid = &mut mid[..4 * n];
                        crate::field::f128_slice::fold4_nested(
                            &b_in[16 * slot..16 * (slot + n)],
                            mid,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(
                            mid,
                            &mut b_out[slot..slot + n],
                            r2,
                            r3,
                        );
                        slot += n;
                    }
                }
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512vbmi",
                    target_feature = "vpclmulqdq",
                    target_feature = "gfni"
                ))]
                if b_gfni_on {
                    use crate::zerocheck::multilinear::kernels::x86_64::gfni_fold64_four_maps_staged;
                    use core::arch::x86_64::_mm512_setzero_si512;

                    let (claim0, claim1) = (&claims[0], &claims[1]);
                    let mut mats0_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats0_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats1_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats1_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    unsafe {
                        super::ring_switch::compose_block_mats_gfni_into(
                            &direct_gfni_mats[0],
                            claim0.eq_hi[block],
                            &mut *mats0_lo.as_mut_ptr(),
                            &mut *mats0_hi.as_mut_ptr(),
                        );
                        super::ring_switch::compose_block_mats_gfni_into(
                            &direct_gfni_mats[1],
                            claim1.eq_hi[block],
                            &mut *mats1_lo.as_mut_ptr(),
                            &mut *mats1_hi.as_mut_ptr(),
                        );
                    }
                    let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
                    let mut planes = unsafe { [_mm512_setzero_si512(); 16] };
                    for slot in (0..block_len).step_by(64) {
                        unsafe {
                            gfni_fold64_four_maps_staged(
                                rows0.0.as_ptr().add(slot).cast::<u8>(),
                                &*mats0_lo.as_ptr(),
                                rows0.1.as_ptr().add(slot).cast::<u8>(),
                                &*mats0_hi.as_ptr(),
                                rows1.0.as_ptr().add(slot).cast::<u8>(),
                                &*mats1_lo.as_ptr(),
                                rows1.1.as_ptr().add(slot).cast::<u8>(),
                                &*mats1_hi.as_ptr(),
                                b_out.as_mut_ptr().add(slot),
                                planes.as_mut_ptr().cast::<core::arch::x86_64::__m512i>(),
                            );
                        }
                    }
                }
                if !b_gfni_on {
                    let table = &mut scratch[..table_len];
                    let mut claims_iter = claims.iter().zip(direct_tables.iter());
                    let (first, first_table) = claims_iter
                        .next()
                        .expect("materialize_direct_fold4: claims non-empty");
                    super::ring_switch::compose_block_table(first_table, first.eq_hi[block], table);
                    let mut s = 0usize;
                    while s + 3 < block_len {
                        #[cfg(target_arch = "x86_64")]
                        if !pf_base.is_null() && pf_at < pf_span {
                            unsafe {
                                core::arch::x86_64::_mm_prefetch(
                                    pf_base.add(pf_at).cast::<i8>(),
                                    core::arch::x86_64::_MM_HINT_T1,
                                );
                            }
                            pf_at += 64;
                        }
                        b_out[s] = super::ring_switch::fold_one_slot(first.eq_lo[s], table);
                        b_out[s + 1] = super::ring_switch::fold_one_slot(first.eq_lo[s + 1], table);
                        b_out[s + 2] = super::ring_switch::fold_one_slot(first.eq_lo[s + 2], table);
                        b_out[s + 3] = super::ring_switch::fold_one_slot(first.eq_lo[s + 3], table);
                        s += 4;
                    }
                    while s < block_len {
                        b_out[s] = super::ring_switch::fold_one_slot(first.eq_lo[s], table);
                        s += 1;
                    }
                    for (claim, direct_table) in claims_iter {
                        super::ring_switch::compose_block_table(
                            direct_table,
                            claim.eq_hi[block],
                            table,
                        );
                        let mut s = 0usize;
                        while s + 3 < block_len {
                            #[cfg(target_arch = "x86_64")]
                            if !pf_base.is_null() && pf_at < pf_span {
                                unsafe {
                                    core::arch::x86_64::_mm_prefetch(
                                        pf_base.add(pf_at).cast::<i8>(),
                                        core::arch::x86_64::_MM_HINT_T1,
                                    );
                                }
                                pf_at += 64;
                            }
                            b_out[s] += super::ring_switch::fold_one_slot(claim.eq_lo[s], table);
                            b_out[s + 1] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 1], table);
                            b_out[s + 2] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 2], table);
                            b_out[s + 3] +=
                                super::ring_switch::fold_one_slot(claim.eq_lo[s + 3], table);
                            s += 4;
                        }
                        while s < block_len {
                            b_out[s] += super::ring_switch::fold_one_slot(claim.eq_lo[s], table);
                            s += 1;
                        }
                    }
                }
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    let mut k = 0;
                    while k + 1 < f_out.len() {
                        let f0 = f_out[k];
                        let f1 = f_out[k + 1];
                        let b0 = b_out[k];
                        let b1 = b_out[k + 1];
                        u0 += f0 * b0;
                        u2 += (f0 + f1) * (b0 + b1);
                        k += 2;
                    }
                    (u0, u2)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    crate::scratch::give_f128(packed_witness);
    if has_ordinary {
        crate::scratch::give_f128(ordinary_basis);
    }
    (folded_f, folded_b, SumcheckMessage { u_0, u_2 })
}

#[allow(clippy::assign_op_pattern)] // Preserve the hand-scheduled indexed stores.
fn materialize_direct_ab_fold2(
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold2Factors],
    r0: F128,
    r1: F128,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    assert!(ordinary_basis.is_empty() || ordinary_basis.len() == packed_witness.len());
    let fold_weight = [
        (F128::ONE + r0) * (F128::ONE + r1),
        r0 * (F128::ONE + r1),
        (F128::ONE + r0) * r1,
        r0 * r1,
    ];
    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            super::ring_switch::build_direct_fold2_table(&claim.low_eq, &fold_weight, &claim.table)
        })
        .collect();

    let out_len = packed_witness.len() / 4;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(2));
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && out_len == block_len * claim.eq_hi.len()
    }));
    let table_len = super::ring_switch::FOLD_TABLE_TOTAL;
    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    let (u_0, u_2) = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            || {
                crate::scratch::LocalBuf::new(
                    if claims.len() == 2 {
                        table_len
                    } else {
                        claims.len() * table_len
                    },
                    crate::scratch::fold_buf_pool_enabled(),
                )
            },
            |scratch, (block, (b_out, f_out))| {
                if claims.len() != 2 {
                    for (claim_index, (claim, direct_table)) in
                        claims.iter().zip(direct_tables.iter()).enumerate()
                    {
                        super::ring_switch::compose_block_table(
                            direct_table,
                            claim.eq_hi[block],
                            &mut scratch[claim_index * table_len..(claim_index + 1) * table_len],
                        );
                    }
                }
                let start = 4 * block * block_len;
                let f_in = &packed_witness[start..start + 4 * block_len];
                let b_in = (!ordinary_basis.is_empty())
                    .then(|| &ordinary_basis[start..start + 4 * block_len]);
                crate::field::f128_slice::fold4_nested(f_in, f_out, r0, r1);
                let fold4 = |input: &[F128], slot: usize| {
                    let a0 = input[4 * slot];
                    let a1 = input[4 * slot + 1];
                    let a2 = input[4 * slot + 2];
                    let a3 = input[4 * slot + 3];
                    let low = a0 + r0 * (a0 + a1);
                    let high = a2 + r0 * (a2 + a3);
                    low + r1 * (low + high)
                };
                if let [only] = claims {
                    let table = &scratch[..table_len];
                    for pair in 0..(block_len / 2) {
                        let slot0 = 2 * pair;
                        let slot1 = slot0 + 1;
                        let b0 = super::ring_switch::fold_one_slot(only.eq_lo[slot0], table)
                            + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot0));
                        let b1 = super::ring_switch::fold_one_slot(only.eq_lo[slot1], table)
                            + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot1));
                        b_out[slot0] = b0;
                        b_out[slot1] = b1;
                    }
                } else if let [first, second] = claims {
                    debug_assert!(b_in.is_none());
                    let table = &mut scratch[..table_len];
                    super::ring_switch::compose_block_table(
                        &direct_tables[0],
                        first.eq_hi[block],
                        table,
                    );
                    let n_pairs = block_len / 2;
                    let mut pair = 0usize;
                    while pair + 1 < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        let s2 = s0 + 2;
                        let s3 = s0 + 3;
                        b_out[s0] = super::ring_switch::fold_one_slot(first.eq_lo[s0], table);
                        b_out[s1] = super::ring_switch::fold_one_slot(first.eq_lo[s1], table);
                        b_out[s2] = super::ring_switch::fold_one_slot(first.eq_lo[s2], table);
                        b_out[s3] = super::ring_switch::fold_one_slot(first.eq_lo[s3], table);
                        pair += 2;
                    }
                    if pair < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        b_out[s0] = super::ring_switch::fold_one_slot(first.eq_lo[s0], table);
                        b_out[s1] = super::ring_switch::fold_one_slot(first.eq_lo[s1], table);
                    }
                    super::ring_switch::compose_block_table(
                        &direct_tables[1],
                        second.eq_hi[block],
                        table,
                    );
                    pair = 0;
                    while pair + 1 < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        let s2 = s0 + 2;
                        let s3 = s0 + 3;
                        b_out[s0] =
                            b_out[s0] + super::ring_switch::fold_one_slot(second.eq_lo[s0], table);
                        b_out[s1] =
                            b_out[s1] + super::ring_switch::fold_one_slot(second.eq_lo[s1], table);
                        b_out[s2] =
                            b_out[s2] + super::ring_switch::fold_one_slot(second.eq_lo[s2], table);
                        b_out[s3] =
                            b_out[s3] + super::ring_switch::fold_one_slot(second.eq_lo[s3], table);
                        pair += 2;
                    }
                    if pair < n_pairs {
                        let s0 = 2 * pair;
                        let s1 = s0 + 1;
                        b_out[s0] =
                            b_out[s0] + super::ring_switch::fold_one_slot(second.eq_lo[s0], table);
                        b_out[s1] =
                            b_out[s1] + super::ring_switch::fold_one_slot(second.eq_lo[s1], table);
                    }
                } else {
                    for pair in 0..(block_len / 2) {
                        let slot0 = 2 * pair;
                        let slot1 = slot0 + 1;
                        let direct_at = |slot: usize| {
                            claims
                                .iter()
                                .enumerate()
                                .map(|(claim_index, claim)| {
                                    super::ring_switch::fold_one_slot(
                                        claim.eq_lo[slot],
                                        &scratch[claim_index * table_len
                                            ..(claim_index + 1) * table_len],
                                    )
                                })
                                .fold(F128::ZERO, |sum, value| sum + value)
                        };
                        let b0 =
                            direct_at(slot0) + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot0));
                        let b1 =
                            direct_at(slot1) + b_in.map_or(F128::ZERO, |basis| fold4(basis, slot1));
                        b_out[slot0] = b0;
                        b_out[slot1] = b1;
                    }
                }
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    let mut u0 = F128::ZERO;
                    let mut u2 = F128::ZERO;
                    let mut k = 0;
                    while k + 1 < f_out.len() {
                        let f0 = f_out[k];
                        let f1 = f_out[k + 1];
                        let b0 = b_out[k];
                        let b1 = b_out[k + 1];
                        u0 += f0 * b0;
                        u2 += (f0 + f1) * (b0 + b1);
                        k += 2;
                    }
                    (u0, u2)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    crate::scratch::give_f128(packed_witness);
    if !ordinary_basis.is_empty() {
        crate::scratch::give_f128(ordinary_basis);
    }
    (folded_f, folded_b, SumcheckMessage { u_0, u_2 })
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes")]
unsafe fn fold_and_msg_chunk_nt_neon(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    #[inline(always)]
    unsafe fn store_nt_pair(dst: *mut F128, v0: F128, v1: F128) {
        unsafe {
            let q0: core::arch::aarch64::uint8x16_t = core::mem::transmute(v0);
            let q1: core::arch::aarch64::uint8x16_t = core::mem::transmute(v1);
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{p}]",
                a = in(vreg) q0,
                b = in(vreg) q1,
                p = in(reg) dst,
                options(nostack, preserves_flags)
            );
        }
    }

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(2));
    let mut u0 = F128::ZERO;
    let mut u2 = F128::ZERO;
    unsafe {
        let mut src_f = f.as_ptr().add(2 * base);
        let mut src_b = b.as_ptr().add(2 * base);
        let mut dst_f = fc.as_mut_ptr();
        let mut dst_b = bc.as_mut_ptr();
        let mut remaining = len / 2;
        while remaining != 0 {
            let fe0 = src_f.read();
            let fo0 = src_f.add(1).read();
            let fe1 = src_f.add(2).read();
            let fo1 = src_f.add(3).read();
            let be0 = src_b.read();
            let bo0 = src_b.add(1).read();
            let be1 = src_b.add(2).read();
            let bo1 = src_b.add(3).read();

            let pf = ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: fe0.lo ^ fo0.lo,
                        hi: fe0.hi ^ fo0.hi,
                    },
                    F128 {
                        lo: fe1.lo ^ fo1.lo,
                        hi: fe1.hi ^ fo1.hi,
                    },
                ],
            );
            let pb = ghash_mul_vec2_neon(
                [r, r],
                [
                    F128 {
                        lo: be0.lo ^ bo0.lo,
                        hi: be0.hi ^ bo0.hi,
                    },
                    F128 {
                        lo: be1.lo ^ bo1.lo,
                        hi: be1.hi ^ bo1.hi,
                    },
                ],
            );
            let f0 = F128 {
                lo: fe0.lo ^ pf[0].lo,
                hi: fe0.hi ^ pf[0].hi,
            };
            let f1 = F128 {
                lo: fe1.lo ^ pf[1].lo,
                hi: fe1.hi ^ pf[1].hi,
            };
            let b0 = F128 {
                lo: be0.lo ^ pb[0].lo,
                hi: be0.hi ^ pb[0].hi,
            };
            let b1 = F128 {
                lo: be1.lo ^ pb[1].lo,
                hi: be1.hi ^ pb[1].hi,
            };

            store_nt_pair(dst_f, f0, f1);
            store_nt_pair(dst_b, b0, b1);

            let g = ghash_mul_vec2_neon([f0, f0 + f1], [b0, b0 + b1]);
            u0 += g[0];
            u2 += g[1];

            src_f = src_f.add(4);
            src_b = src_b.add(4);
            dst_f = dst_f.add(2);
            dst_b = dst_b.add(2);
            remaining -= 1;
        }
    }
    (u0, u2)
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "aes,sha3")]
unsafe fn fold_and_msg_chunk_nt_neon_soa<const NT: bool>(
    f: &[F128],
    b: &[F128],
    base: usize,
    fc: &mut [F128],
    bc: &mut [F128],
    r: F128,
) -> (F128, F128) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn pmull_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let d: uint64x2_t;
        unsafe {
            core::arch::asm!(
                "pmull {d:v}.1q, {a:v}.1d, {b:v}.1d",
                d = lateout(vreg) d,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        d
    }
    #[inline(always)]
    unsafe fn pmull_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        let d: uint64x2_t;
        unsafe {
            core::arch::asm!(
                "pmull2 {d:v}.1q, {a:v}.2d, {b:v}.2d",
                d = lateout(vreg) d,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        d
    }

    #[inline(always)]
    unsafe fn mul2_kara_barrett_aos(
        d_lo: uint64x2_t,
        d_hi: uint64x2_t,
        r_lo: uint64x2_t,
        r_hi: uint64x2_t,
        r_sum: uint64x2_t,
        c87: uint64x2_t,
    ) -> (uint64x2_t, uint64x2_t) {
        unsafe {
            let d_sum = veorq_u64(d_lo, d_hi);
            let p0_0 = pmull_lo(d_lo, r_lo);
            let p0_1 = pmull_hi(d_lo, r_lo);
            let p2_0 = pmull_lo(d_hi, r_hi);
            let p2_1 = pmull_hi(d_hi, r_hi);
            let pm_0 = pmull_lo(d_sum, r_sum);
            let pm_1 = pmull_hi(d_sum, r_sum);
            let c_0 = veor3q_u64(pm_0, p0_0, p2_0);
            let c_1 = veor3q_u64(pm_1, p0_1, p2_1);

            let zero = vdupq_n_u64(0);
            let lo_0 = veorq_u64(p0_0, vextq_u64::<1>(zero, c_0));
            let hi_0 = veorq_u64(p2_0, vextq_u64::<1>(c_0, zero));
            let lo_1 = veorq_u64(p0_1, vextq_u64::<1>(zero, c_1));
            let hi_1 = veorq_u64(p2_1, vextq_u64::<1>(c_1, zero));

            let rl_0 = pmull_lo(hi_0, c87);
            let rh_0 = pmull_hi(hi_0, c87);
            let rl_1 = pmull_lo(hi_1, c87);
            let rh_1 = pmull_hi(hi_1, c87);
            let cor_0 = pmull_hi(rh_0, c87);
            let cor_1 = pmull_hi(rh_1, c87);

            let res_0 = veor3q_u64(lo_0, rl_0, vzip1q_u64(cor_0, rh_0));
            let res_1 = veor3q_u64(lo_1, rl_1, vzip1q_u64(cor_1, rh_1));
            (res_0, res_1)
        }
    }

    #[inline(always)]
    unsafe fn store_pair_v<const NT: bool>(dst: *mut F128, v0: uint64x2_t, v1: uint64x2_t) {
        unsafe {
            if NT {
                core::arch::asm!(
                    "stnp {a:q}, {b:q}, [{p}]",
                    a = in(vreg) v0,
                    b = in(vreg) v1,
                    p = in(reg) dst,
                    options(nostack, preserves_flags)
                );
            } else {
                core::arch::asm!(
                    "stp {a:q}, {b:q}, [{p}]",
                    a = in(vreg) v0,
                    b = in(vreg) v1,
                    p = in(reg) dst,
                    options(nostack, preserves_flags)
                );
            }
        }
    }

    let len = fc.len();
    debug_assert_eq!(bc.len(), len);
    debug_assert!(len.is_multiple_of(2));
    unsafe {
        let mut src_f = f.as_ptr().add(2 * base) as *const u64;
        let mut src_b = b.as_ptr().add(2 * base) as *const u64;
        let mut dst_f = fc.as_mut_ptr();
        let mut dst_b = bc.as_mut_ptr();
        let r_lo = vdupq_n_u64(r.lo);
        let r_hi = vdupq_n_u64(r.hi);
        let r_sum = vdupq_n_u64(r.lo ^ r.hi);
        let c87 = vdupq_n_u64(0x87);
        let mut acc_d0_0 = vdupq_n_u64(0);
        let mut acc_d0_1 = vdupq_n_u64(0);
        let mut acc_dm_0 = vdupq_n_u64(0);
        let mut acc_dm_1 = vdupq_n_u64(0);
        let mut acc_d2_0 = vdupq_n_u64(0);
        let mut acc_d2_1 = vdupq_n_u64(0);
        let mut remaining = len / 2;
        while remaining != 0 {
            let fe0 = vld1q_u64(src_f);
            let fo0 = vld1q_u64(src_f.add(2));
            let fe1 = vld1q_u64(src_f.add(4));
            let fo1 = vld1q_u64(src_f.add(6));
            let be0 = vld1q_u64(src_b);
            let bo0 = vld1q_u64(src_b.add(2));
            let be1 = vld1q_u64(src_b.add(4));
            let bo1 = vld1q_u64(src_b.add(6));

            let fd0 = veorq_u64(fe0, fo0);
            let fd1 = veorq_u64(fe1, fo1);
            let (pf0, pf1) = mul2_kara_barrett_aos(
                vzip1q_u64(fd0, fd1),
                vzip2q_u64(fd0, fd1),
                r_lo,
                r_hi,
                r_sum,
                c87,
            );
            let bd0 = veorq_u64(be0, bo0);
            let bd1 = veorq_u64(be1, bo1);
            let (pb0, pb1) = mul2_kara_barrett_aos(
                vzip1q_u64(bd0, bd1),
                vzip2q_u64(bd0, bd1),
                r_lo,
                r_hi,
                r_sum,
                c87,
            );
            let f0 = veorq_u64(fe0, pf0);
            let f1 = veorq_u64(fe1, pf1);
            let b0 = veorq_u64(be0, pb0);
            let b1 = veorq_u64(be1, pb1);

            store_pair_v::<NT>(dst_f, f0, f1);
            store_pair_v::<NT>(dst_b, b0, b1);

            let fs = veorq_u64(f0, f1);
            let bs = veorq_u64(b0, b1);
            let a_lo = vzip1q_u64(f0, fs);
            let a_hi = vzip2q_u64(f0, fs);
            let b_lo = vzip1q_u64(b0, bs);
            let b_hi = vzip2q_u64(b0, bs);
            let a_sum = veorq_u64(a_lo, a_hi);
            let b_sum = veorq_u64(b_lo, b_hi);
            acc_d0_0 = veorq_u64(acc_d0_0, pmull_lo(a_lo, b_lo));
            acc_d0_1 = veorq_u64(acc_d0_1, pmull_hi(a_lo, b_lo));
            acc_dm_0 = veorq_u64(acc_dm_0, pmull_lo(a_sum, b_sum));
            acc_dm_1 = veorq_u64(acc_dm_1, pmull_hi(a_sum, b_sum));
            acc_d2_0 = veorq_u64(acc_d2_0, pmull_lo(a_hi, b_hi));
            acc_d2_1 = veorq_u64(acc_d2_1, pmull_hi(a_hi, b_hi));

            src_f = src_f.add(8);
            src_b = src_b.add(8);
            dst_f = dst_f.add(2);
            dst_b = dst_b.add(2);
            remaining -= 1;
        }

        #[inline(always)]
        unsafe fn finish(d0: uint64x2_t, dm: uint64x2_t, d2: uint64x2_t) -> F128 {
            unsafe {
                let c = veor3q_u64(dm, d0, d2);
                crate::field::gf2_128::ghash_reduce(
                    vgetq_lane_u64::<0>(d0),
                    vgetq_lane_u64::<1>(d0) ^ vgetq_lane_u64::<0>(c),
                    vgetq_lane_u64::<0>(d2) ^ vgetq_lane_u64::<1>(c),
                    vgetq_lane_u64::<1>(d2),
                )
            }
        }
        (
            finish(acc_d0_0, acc_dm_0, acc_d2_0),
            finish(acc_d0_1, acc_dm_1, acc_d2_1),
        )
    }
}

pub struct FoldArena {
    ptr: std::ptr::NonNull<F128>,
    cap: usize,
    offset: usize,
    parts: Vec<PrefaultPart>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

struct PrefaultPart {
    start: usize,
    end: usize,
    done: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

unsafe impl Send for FoldArena {}

impl FoldArena {
    pub fn capacity_for(l: usize, k: usize) -> usize {
        debug_assert!(l.is_power_of_two());
        debug_assert!(k >= 1 && k <= l.trailing_zeros() as usize);
        2 * (l - (l >> k))
    }

    pub fn new_prefaulted(cap: usize) -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        assert!(cap > 0);
        let layout = std::alloc::Layout::array::<F128>(cap).expect("FoldArena layout");
        let raw = unsafe { std::alloc::alloc(layout) } as *mut F128;
        let Some(ptr) = std::ptr::NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        struct SendPtr(*mut u8);
        unsafe impl Send for SendPtr {}

        const PREFAULT_THREADS: usize = 3;
        #[cfg(target_os = "macos")]
        const STRIDE: usize = 16384;
        #[cfg(not(target_os = "macos"))]
        const STRIDE: usize = 4096;
        const STRIDE_ELEMS: usize = STRIDE / core::mem::size_of::<F128>();
        const CHUNK_ELEMS: usize = (4 << 20) / core::mem::size_of::<F128>();

        let bound = |i: usize| -> usize {
            let raw = cap * i / PREFAULT_THREADS;
            (raw / STRIDE_ELEMS) * STRIDE_ELEMS
        };
        let mut parts = Vec::with_capacity(PREFAULT_THREADS);
        let mut threads = Vec::with_capacity(PREFAULT_THREADS);
        for i in 0..PREFAULT_THREADS {
            let (start, end) = (
                bound(i),
                if i + 1 == PREFAULT_THREADS {
                    cap
                } else {
                    bound(i + 1)
                },
            );
            let done = Arc::new(AtomicUsize::new(start));
            parts.push(PrefaultPart {
                start,
                end,
                done: done.clone(),
            });
            if start >= end {
                done.store(end, Ordering::Release);
                continue;
            }
            let base = SendPtr(unsafe { raw.add(start) } as *mut u8);
            let elems = end - start;
            threads.push(std::thread::spawn(move || {
                let base = base;
                let mut e = 0usize;
                while e < elems {
                    let chunk_end = (e + CHUNK_ELEMS).min(elems);
                    while e < chunk_end {
                        unsafe {
                            std::ptr::write_volatile(
                                base.0.add(e * core::mem::size_of::<F128>()),
                                0u8,
                            )
                        };
                        e += STRIDE_ELEMS;
                    }
                    let e = e.min(elems);
                    done.store(start + e, Ordering::Release);
                }
                done.store(end, Ordering::Release);
            }));
        }
        Self {
            ptr,
            cap,
            offset: 0,
            parts,
            threads,
        }
    }

    fn carve_pair(&mut self, half: usize) -> Option<(FoldBuf, FoldBuf)> {
        use std::sync::atomic::Ordering;
        if self.cap - self.offset < 2 * half {
            return None;
        }
        let (lo, hi) = (self.offset, self.offset + 2 * half);
        let ready = self.parts.iter().all(|p| {
            hi <= p.start || lo >= p.end || p.done.load(Ordering::Acquire) >= hi.min(p.end)
        });
        if !ready {
            return None;
        }
        let a = unsafe { self.ptr.add(self.offset) };
        let b = unsafe { self.ptr.add(self.offset + half) };
        self.offset += 2 * half;
        Some((
            FoldBuf::Arena { ptr: a, len: half },
            FoldBuf::Arena { ptr: b, len: half },
        ))
    }
}

impl Drop for FoldArena {
    fn drop(&mut self) {
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
        let layout = std::alloc::Layout::array::<F128>(self.cap).expect("FoldArena layout");
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}

pub(crate) enum FoldBuf {
    Owned(Vec<F128>),
    Arena {
        ptr: std::ptr::NonNull<F128>,
        len: usize,
    },
}

unsafe impl Send for FoldBuf {}
unsafe impl Sync for FoldBuf {}

impl Default for FoldBuf {
    fn default() -> Self {
        FoldBuf::Owned(Vec::new())
    }
}

impl std::ops::Deref for FoldBuf {
    type Target = [F128];
    #[inline]
    fn deref(&self) -> &[F128] {
        match self {
            FoldBuf::Owned(v) => v,
            FoldBuf::Arena { ptr, len } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }
}

impl std::ops::DerefMut for FoldBuf {
    #[inline]
    fn deref_mut(&mut self) -> &mut [F128] {
        match self {
            FoldBuf::Owned(v) => v,
            FoldBuf::Arena { ptr, len } => unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), *len)
            },
        }
    }
}

#[inline]
fn ranked_l1_lazy_ood_eq_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
) -> bool {
    ranked_l1_lazy_ood_eq_selected(
        config,
        log_n,
        n_1,
        l1_ood_count,
        current_len,
        direct_fold4_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        crate::sl2_env_set!("FLOCK_NO_LIG_LAZY_OOD_EQ"),
    )
}

#[inline]
fn ranked_l1_lazy_ood_eq_selected(
    config: &ProverConfig,
    log_n: usize,
    n_1: usize,
    l1_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && direct_fold4_mode
        && log_n == 25
        && n_1 == 19
        && current_len == (1usize << 19)
        && l1_ood_count == 1
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.log_inv_rates.first() == Some(&1)
}

#[inline]
fn ranked_deep_lazy_ood_eq_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    level_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
) -> bool {
    ranked_deep_lazy_ood_eq_selected(
        config,
        log_n,
        n_level,
        level_ood_count,
        current_len,
        direct_fold4_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        crate::sl2_env_set!("FLOCK_NO_LIG_LAZY_OOD_EQ")
            || crate::sl2_env_set!("FLOCK_NO_LIG_LAZY_OOD_EQ_DEEP"),
    )
}

#[inline]
fn ranked_deep_lazy_ood_eq_selected(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    level_ood_count: usize,
    current_len: usize,
    direct_fold4_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && direct_fold4_mode
        && log_n == 25
        && matches!(n_level, 16 | 13 | 10 | 7)
        && current_len == (1usize << n_level)
        && level_ood_count == 1
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.log_inv_rates.first() == Some(&1)
}

const ENV_NO_LIG_DEFER_INDUCED_GLUE: &str = "FLOCK_NO_LIG_DEFER_INDUCED_GLUE";
const ENV_NO_OPEN_INDUCE_DUAL: &str = "FLOCK_NO_OPEN_INDUCE_DUAL";
const ENV_NO_OPEN_INDUCE_DUAL2: &str = "FLOCK_NO_OPEN_INDUCE_DUAL2";
const ENV_OPEN_INDUCE_DUAL_DEPTH: &str = "FLOCK_OPEN_INDUCE_DUAL_DEPTH";
const ENV_NO_OPEN_INDUCE_DUAL_DEPTH2: &str = "FLOCK_NO_OPEN_INDUCE_DUAL_DEPTH2";
const RANKED_SPARSE_DUAL_DEFAULT_DEPTH: usize = 2;
const ENV_NO_DEEP_INDUCE_DUAL: &str = "FLOCK_NO_DEEP_INDUCE_DUAL";
#[cfg(test)]
thread_local! {
    static SPARSE_DUAL_TEST_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SPARSE_DUAL_DEEP_TEST_DEPTH: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_deferred_induced_glue_selected(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
    platform_supported: bool,
    disabled: bool,
) -> bool {
    platform_supported
        && !disabled
        && lazy_ood
        && direct_fold8_mode
        && log_n == 25
        && current_len == (1usize << n_level)
        && matches!(
            (n_level, n_queries, log_inv_rate),
            (19, 218, 1) | (16, 106, 2) | (13, 71, 3) | (10, 53, 4) | (7, 43, 5)
        )
        && config.initial_log_msg_cols == 19
        && config.initial_log_num_interleaved == 6
        && config.initial_k == 6
        && config.recursive_steps == 5
        && config.recursive_log_msg_cols.as_slice() == [16, 13, 10, 7, 4]
        && config.recursive_ks.as_slice() == [3, 3, 3, 3, 3]
        && config.log_inv_rates.as_slice() == [1, 2, 3, 4, 5, 6]
        && config.queries.as_slice() == [218, 106, 71, 53, 43, 36]
        && config.grinding_bits.as_slice() == [0, 0, 0, 0, 0, 0]
        && config.fold_grinding_bits.as_slice() == [19, 14, 11, 8, 6, 4]
        && config.ood_samples.as_slice() == [0, 1, 1, 1, 1, 1]
        && config.merkle_hash == HashKind::Blake3
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_deferred_induced_glue_enabled(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
) -> bool {
    ranked_deferred_induced_glue_selected(
        config,
        log_n,
        n_level,
        current_len,
        n_queries,
        log_inv_rate,
        lazy_ood,
        direct_fold8_mode,
        cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )),
        crate::sl2_env_set!(ENV_NO_LIG_DEFER_INDUCED_GLUE),
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_sparse_dual_l0_depth_selected(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
    platform_supported: bool,
    sparse_disabled: bool,
    defer_disabled: bool,
    depth: usize,
) -> Option<usize> {
    let selected = matches!(n_level, 19 | 16 | 13 | 10 | 7)
        && ranked_deferred_induced_glue_selected(
            config,
            log_n,
            n_level,
            current_len,
            n_queries,
            log_inv_rate,
            lazy_ood,
            direct_fold8_mode,
            platform_supported,
            defer_disabled,
        )
        && !sparse_disabled;
    if !selected {
        return None;
    }
    assert!((1..=SPARSE_DUAL_MAX_DEPTH).contains(&depth));
    Some(depth)
}

#[inline]
fn ranked_sparse_dual_l0_depth_setting(
    explicit_depth: Option<&str>,
    depth2_disabled: bool,
) -> usize {
    let depth = explicit_depth
        .map(|value| {
            value
                .parse::<usize>()
                .expect("FLOCK_OPEN_INDUCE_DUAL_DEPTH must be 1, 2, 3, or 4")
        })
        .unwrap_or(if depth2_disabled {
            SPARSE_DUAL_MAX_DEPTH
        } else {
            RANKED_SPARSE_DUAL_DEFAULT_DEPTH
        });
    assert!((1..=SPARSE_DUAL_MAX_DEPTH).contains(&depth));
    depth
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_sparse_dual_l0_depth(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
) -> Option<usize> {
    #[cfg(test)]
    {
        let forced = SPARSE_DUAL_TEST_DEPTH.with(std::cell::Cell::get);
        if forced != 0 {
            assert!((1..=SPARSE_DUAL_MAX_DEPTH).contains(&forced));
            return Some(forced);
        }
    }
    let sparse_disabled = crate::sl2_env_set!(ENV_NO_OPEN_INDUCE_DUAL)
        || crate::sl2_env_set!(ENV_NO_OPEN_INDUCE_DUAL2);
    let defer_disabled = crate::sl2_env_set!(ENV_NO_LIG_DEFER_INDUCED_GLUE);
    let platform_supported = cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ));
    if ranked_sparse_dual_l0_depth_selected(
        config,
        log_n,
        n_level,
        current_len,
        n_queries,
        log_inv_rate,
        lazy_ood,
        direct_fold8_mode,
        platform_supported,
        sparse_disabled,
        defer_disabled,
        SPARSE_DUAL_MAX_DEPTH,
    )
    .is_none()
    {
        return None;
    }
    static EXPLICIT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let explicit_depth = if crate::sl2_env_live() {
        std::env::var(ENV_OPEN_INDUCE_DUAL_DEPTH).ok()
    } else {
        EXPLICIT.get_or_init(|| std::env::var(ENV_OPEN_INDUCE_DUAL_DEPTH).ok()).clone()
    };
    let depth = ranked_sparse_dual_l0_depth_setting(
        explicit_depth.as_deref(),
        crate::sl2_env_set!(ENV_NO_OPEN_INDUCE_DUAL_DEPTH2),
    );
    Some(depth)
}

#[inline]
fn ranked_sparse_dual_deep_depth_for_level(n_level: usize) -> Option<usize> {
    match n_level {
        16 | 13 | 10 => Some(SPARSE_DUAL_MAX_DEPTH),
        7 => Some(3),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ranked_sparse_dual_deep_depth(
    config: &ProverConfig,
    log_n: usize,
    n_level: usize,
    current_len: usize,
    n_queries: usize,
    log_inv_rate: usize,
    lazy_ood: bool,
    direct_fold8_mode: bool,
    remaining_folds: usize,
) -> Option<usize> {
    #[cfg(test)]
    {
        let forced = SPARSE_DUAL_DEEP_TEST_DEPTH.with(std::cell::Cell::get);
        if forced != 0 {
            assert!((2..=SPARSE_DUAL_MAX_DEPTH).contains(&forced));
            let depth = forced.min(remaining_folds);
            return (depth >= 2).then_some(depth);
        }
    }
    if crate::sl2_env_set!(ENV_NO_DEEP_INDUCE_DUAL) {
        return None;
    }
    let depth = ranked_sparse_dual_deep_depth_for_level(n_level)?;
    if depth > remaining_folds {
        return None;
    }
    let sparse_disabled = crate::sl2_env_set!(ENV_NO_OPEN_INDUCE_DUAL)
        || crate::sl2_env_set!(ENV_NO_OPEN_INDUCE_DUAL2);
    let defer_disabled = crate::sl2_env_set!(ENV_NO_LIG_DEFER_INDUCED_GLUE);
    let platform_supported = cfg!(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ));
    ranked_sparse_dual_l0_depth_selected(
        config,
        log_n,
        n_level,
        current_len,
        n_queries,
        log_inv_rate,
        lazy_ood,
        direct_fold8_mode,
        platform_supported,
        sparse_disabled,
        defer_disabled,
        depth,
    )
}

enum PendingOodEq {
    Introduced {
        eq_lo: Vec<F128>,
        eq_hi: Vec<F128>,
        z_0: F128,
        h_new: F128,
    },
    Glued {
        eq_lo: Vec<F128>,
        eq_hi: Vec<F128>,
        z_0: F128,
        beta: F128,
    },
}

#[inline]
fn fold_one_direct_fold8_claim_and_message(
    claim: &mut super::ring_switch::DirectFold8Factors,
    challenge: F128,
) -> (F128, F128) {
    assert_eq!(claim.a_state.len(), claim.w_state.len());
    assert_eq!(claim.a_state.len() % (1usize << super::LOG_PACKING), 0);
    let banks = claim.a_state.len() >> super::LOG_PACKING;
    assert!(banks >= 4 && banks.is_power_of_two());
    crate::field::f128_slice::fold_two_and_msg_in_place(
        &mut claim.a_state,
        &mut claim.w_state,
        challenge,
    )
}

/// Lane opus-pcs: `FLOCK_NO_OPUS_F8B_MSG_FUSION=1` restores the two-pass
/// basis-then-message block form in the GFNI direct-fold8 basis kernel.
/// Read once.
#[allow(dead_code)] // Consumed only by the ranked AVX-512 GFNI route.
fn opus_f8b_msg_fusion_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPUS_F8B_MSG_FUSION").is_none());
    *ON
}

/// Lane opus-pcs: `FLOCK_NO_OPUS_F8_CLAIMS_JOIN=1` folds the two ranked
/// claims' factor states one after the other (incumbent) instead of side by
/// side. Read once.
fn opus_f8_claims_join_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPUS_F8_CLAIMS_JOIN").is_none());
    *ON
}

fn fold_direct_fold8_factors_and_message(
    claims: &mut [super::ring_switch::DirectFold8Factors],
    challenge: F128,
) -> SumcheckMessage {
    // Lane opus-pcs: the two claims' in-place factor folds are independent
    // (each touches only its own states); run them side by side while the
    // states are large enough to amortize the join. The message is the XOR
    // of the same two partial sums, so it is identical.
    if claims.len() == 2 && claims[0].a_state.len() >= 2048 && opus_f8_claims_join_enabled() {
        let (first, second) = claims.split_at_mut(1);
        let (p0, p1) = rayon::join(
            || fold_one_direct_fold8_claim_and_message(&mut first[0], challenge),
            || fold_one_direct_fold8_claim_and_message(&mut second[0], challenge),
        );
        return SumcheckMessage {
            u_0: F128::ZERO + p0.0 + p1.0,
            u_2: F128::ZERO + p0.1 + p1.1,
        };
    }
    let (u0, u2) = claims
        .iter_mut()
        .fold((F128::ZERO, F128::ZERO), |mut total, claim| {
            let part = fold_one_direct_fold8_claim_and_message(claim, challenge);
            total.0 += part.0;
            total.1 += part.1;
            total
        });
    SumcheckMessage { u_0: u0, u_2: u2 }
}

fn direct_fold8_final_generators(
    claim: &super::ring_switch::DirectFold8Factors,
    challenge: F128,
) -> Vec<F128> {
    let mut generators = crate::alloc_uninit_vec(claim.w_state.len() / 2);
    crate::field::f128_slice::fold_pairs(&claim.w_state, 0, &mut generators, challenge);
    generators
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn direct_fold8_b_gfni_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_FOLD8_B_GFNI").is_none());
    *ON
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Fields are consumed only by the ranked AVX-512 route.
struct DirectFold8L1Precommit {
    log_msg_cols: usize,
    log_num_interleaved: usize,
    log_inv_rate: usize,
    kind: HashKind,
}

#[inline]
#[allow(dead_code)] // Selector is consumed only by the ranked AVX-512 route.
fn direct_fold8_b_overlap_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_F8B_OVERLAP").is_none());
    *ON
}

/// Lane rs: `FLOCK_NO_RS_F8_REBAL=1` (or `FLOCK_NO_RS_SEED6=1`) restores
/// the fold64 ‖ basis, then commit ‖ message schedule. Read once.
fn rs_f8_rebal_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_RS_F8_REBAL").is_none()
            && std::env::var_os("FLOCK_NO_RS_SEED6").is_none()
    });
    *ON
}

fn direct_fold8_l1_precommit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_L1_PRECOMMIT").is_none());
    *ON
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn split_eq_lo_halves_avx512(eq_lo: &[F128]) -> (Vec<u64>, Vec<u64>) {
    use core::arch::x86_64::*;
    let n = eq_lo.len();
    let mut lo: Vec<u64> = Vec::with_capacity(n);
    let mut hi: Vec<u64> = Vec::with_capacity(n);
    unsafe {
        let idx_lo = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let idx_hi = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        let src_ptr = eq_lo.as_ptr();
        let lo_ptr = lo.as_mut_ptr();
        let hi_ptr = hi.as_mut_ptr();
        let mut i = 0usize;
        let simd_end = n & !7;
        while i < simd_end {
            let v0 = _mm512_loadu_si512(src_ptr.add(i) as *const __m512i);
            let v1 = _mm512_loadu_si512(src_ptr.add(i + 4) as *const __m512i);
            let lo_vec = _mm512_permutex2var_epi64(v0, idx_lo, v1);
            let hi_vec = _mm512_permutex2var_epi64(v0, idx_hi, v1);
            _mm512_storeu_si512(lo_ptr.add(i) as *mut __m512i, lo_vec);
            _mm512_storeu_si512(hi_ptr.add(i) as *mut __m512i, hi_vec);
            i += 8;
        }
        while i < n {
            let x = *src_ptr.add(i);
            *lo_ptr.add(i) = x.lo;
            *hi_ptr.add(i) = x.hi;
            i += 1;
        }
        lo.set_len(n);
        hi.set_len(n);
    }
    (lo, hi)
}

/// Benchmark/oracle hook: the dense direct-fold8 f materialization exactly
/// as the ranked open runs it (`chunk_len = 512`), bypassing any elided-z
/// registration.
#[doc(hidden)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
pub fn bench_direct_fold8_f_dense(
    packed_witness: &[F128],
    folded_f: &mut [F128],
    fold16_weight: &[F128; 16],
    r4: F128,
    r5: F128,
) {
    use rayon::prelude::*;
    folded_f
        .par_chunks_mut(512)
        .enumerate()
        .for_each(|(block, f_out)| {
            let start = 64 * block * 512;
            let f_in = &packed_witness[start..start + 64 * f_out.len()];
            crate::field::f128_slice::fold64_banked_nested(f_in, f_out, fold16_weight, r4, r5);
        });
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn materialize_direct_fold8_f_for_precommit(
    packed_witness: &[F128],
    folded_f: &mut [F128],
    block_len: usize,
    fold16_weight: &[F128; 16],
    r4: F128,
    r5: F128,
) {
    use rayon::prelude::*;

    // Elided witness: fold straight from the exact regenerator. The weight
    // of source slot 16s+b is the product of the same factors the nested
    // fold applies (fold16 weight, then r4 on s bit 0, r5 on s bit 1); the
    // field is exact, so every output is bit-identical.
    if let Some(regen) = super::elided_z::take(packed_witness) {
        let t = std::time::Instant::now();
        let e4 = [F128::ONE + r4, r4];
        let e5 = [F128::ONE + r5, r5];
        let w: [F128; 64] = std::array::from_fn(|i| {
            let s = i >> 4;
            fold16_weight[i & 15] * e4[s & 1] * e5[s >> 1]
        });
        assert_eq!(folded_f.len() * 64, packed_witness.len());
        regen.fold64(folded_f, &w);
        drop(regen);
        if open_timing() || std::env::var_os("FLOCK_ZR_TRACE").is_some() {
            eprintln!("[zr] regen fold64 {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        return;
    }

    let chunk_len = block_len.min(512);
    folded_f
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(block, f_out)| {
            let start = 64 * block * chunk_len;
            let f_in = &packed_witness[start..start + 64 * f_out.len()];
            crate::field::f128_slice::fold64_banked_nested(f_in, f_out, fold16_weight, r4, r5);
        });
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline(never)]
#[cfg_attr(
    target_feature = "avx512bw",
    target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,vpclmulqdq")
)]
#[cfg_attr(
    not(target_feature = "avx512bw"),
    target_feature(enable = "avx512f,avx512vbmi,gfni,vpclmulqdq")
)]
unsafe fn materialize_direct_fold8_b_gfni_block_x86(
    direct_gfni_mats: &[super::ring_switch::GfniDirectFoldMap; 2],
    direct_gfni_rows: &[(Vec<u64>, Vec<u64>); 2],
    e_hi0: F128,
    e_hi1: F128,
    block_len: usize,
    b_out: &mut [F128],
    f_out: Option<&[F128]>,
    gfni_tmp: &mut [F128],
) -> (F128, F128) {
    use crate::pcs::ring_switch::compose_block_mats_gfni_into;
    use crate::zerocheck::multilinear::kernels::x86_64::gfni_fold64_four_maps_staged;

    let mut mats0_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
    let mut mats0_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
    let mut mats1_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
    let mut mats1_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
    unsafe {
        compose_block_mats_gfni_into(
            &direct_gfni_mats[0],
            e_hi0,
            &mut *mats0_lo.as_mut_ptr(),
            &mut *mats0_hi.as_mut_ptr(),
        );
        compose_block_mats_gfni_into(
            &direct_gfni_mats[1],
            e_hi1,
            &mut *mats1_lo.as_mut_ptr(),
            &mut *mats1_hi.as_mut_ptr(),
        );
        let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
        // Lane opus-pcs: accumulate the round message strip by strip right
        // after each 64-slot strip of b is written (still in L1), instead of
        // a second pass over the whole block. The strips run in slot order,
        // so the raw Karatsuba accumulators see exactly the sequence
        // `msg_reduce_avx512(f_out, b_out)` would; one fold/reduce per block
        // as before. `FLOCK_NO_OPUS_F8B_MSG_FUSION=1` restores the two-pass form.
        let fuse = f_out.filter(|_| opus_f8b_msg_fusion_enabled());
        let mut u0_acc = crate::field::gf2_128::x86_64::WideGhashX4::zero();
        let mut u2_acc = crate::field::gf2_128::x86_64::WideGhashX4::zero();
        for slot in (0..block_len).step_by(64) {
            gfni_fold64_four_maps_staged(
                rows0.0.as_ptr().add(slot).cast::<u8>(),
                &*mats0_lo.as_ptr(),
                rows0.1.as_ptr().add(slot).cast::<u8>(),
                &*mats0_hi.as_ptr(),
                rows1.0.as_ptr().add(slot).cast::<u8>(),
                &*mats1_lo.as_ptr(),
                rows1.1.as_ptr().add(slot).cast::<u8>(),
                &*mats1_hi.as_ptr(),
                b_out.as_mut_ptr().add(slot),
                gfni_tmp.as_mut_ptr().cast(),
            );
            if let Some(f_out) = fuse {
                msg_accumulate_avx512(
                    &f_out[slot..slot + 64],
                    &b_out[slot..slot + 64],
                    &mut u0_acc,
                    &mut u2_acc,
                );
            }
        }
        if fuse.is_some() {
            return (u0_acc.fold().reduce(), u2_acc.fold().reduce());
        }
        match f_out {
            Some(f_out) => msg_reduce_avx512(f_out, b_out),
            None => (F128::ZERO, F128::ZERO),
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn materialize_direct_fold8_b_gfni_for_precommit(
    folded_f: &[F128],
    folded_b: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
    block_len: usize,
) -> (Vec<F128>, SumcheckMessage) {
    materialize_direct_fold8_b_gfni_impl(Some(folded_f), folded_b, claims, challenges, block_len)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn materialize_direct_fold8_b_gfni_impl(
    folded_f: Option<&[F128]>,
    mut folded_b: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
    block_len: usize,
) -> (Vec<F128>, SumcheckMessage) {
    use rayon::prelude::*;

    assert_eq!(claims.len(), 2);
    assert!(block_len.is_multiple_of(64));
    if let Some(folded_f) = folded_f {
        assert_eq!(folded_b.len(), folded_f.len());
    }
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == folded_b.len()
    }));

    let direct_gfni_mats = [
        super::ring_switch::build_gfni_direct_fold_map_from_generators(
            &direct_fold8_final_generators(&claims[0], challenges[5]),
        ),
        super::ring_switch::build_gfni_direct_fold_map_from_generators(
            &direct_fold8_final_generators(&claims[1], challenges[5]),
        ),
    ];
    let direct_gfni_rows = [
        split_eq_lo_halves_avx512(&claims[0].eq_lo),
        split_eq_lo_halves_avx512(&claims[1].eq_lo),
    ];

    const ALIGN64_MIN_F128: usize = (32 * 1024) / core::mem::size_of::<F128>();
    let stats = folded_b
        .par_chunks_mut(block_len)
        .enumerate()
        .map_init(
            || {
                crate::scratch::LocalBuf::new(
                    64usize.max(ALIGN64_MIN_F128),
                    crate::scratch::fold_buf_pool_enabled(),
                )
            },
            |gfni_tmp, (block, b_out)| unsafe {
                let f_out = folded_f.map(|f| &f[block * block_len..(block + 1) * block_len]);
                materialize_direct_fold8_b_gfni_block_x86(
                    &direct_gfni_mats,
                    &direct_gfni_rows,
                    claims[0].eq_hi[block],
                    claims[1].eq_hi[block],
                    block_len,
                    b_out,
                    f_out,
                    gfni_tmp,
                )
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );

    (
        folded_b,
        SumcheckMessage {
            u_0: stats.0,
            u_2: stats.1,
        },
    )
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn direct_fold8_msg_par(folded_f: &[F128], folded_b: &[F128], block_len: usize) -> SumcheckMessage {
    use rayon::prelude::*;
    assert_eq!(folded_f.len(), folded_b.len());
    let stats = folded_f
        .par_chunks(block_len)
        .zip(folded_b.par_chunks(block_len))
        .map(|(f, b)| {
            crate::rstrace::evd("msg:s", 0);
            let r = unsafe { msg_reduce_avx512(f, b) };
            crate::rstrace::evd("msg:e", 0);
            r
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );
    SumcheckMessage {
        u_0: stats.0,
        u_2: stats.1,
    }
}

fn materialize_direct_fold8(
    mut packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    claims: &[super::ring_switch::DirectFold8Factors],
    challenges: [F128; 6],
    l1_precommit: Option<DirectFold8L1Precommit>,
) -> (Vec<F128>, Vec<F128>, SumcheckMessage, Option<LigeroWitness>) {
    use rayon::prelude::*;

    assert!(!claims.is_empty());
    let has_ordinary = !ordinary_basis.is_empty();
    assert!(!has_ordinary || ordinary_basis.len() == packed_witness.len());
    assert!(packed_witness.len().is_multiple_of(64));
    let [r0, r1, r2, r3, r4, r5] = challenges;
    let fold16_weight: [F128; 16] = std::array::from_fn(|bank| {
        let mut weight = F128::ONE;
        for (bit, &challenge) in challenges[..4].iter().enumerate() {
            weight *= if (bank >> bit) & 1 == 0 {
                F128::ONE + challenge
            } else {
                challenge
            };
        }
        weight
    });
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let b_gfni_candidate = direct_fold8_b_gfni_enabled() && claims.len() == 2 && !has_ordinary;
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let b_gfni_candidate = false;

    let out_len = packed_witness.len() / 64;
    let block_len = claims[0].eq_lo.len();
    assert!(block_len.is_multiple_of(4));
    let b_gfni_on = b_gfni_candidate && block_len.is_multiple_of(64);
    assert_eq!(out_len, block_len * claims[0].eq_hi.len());
    assert!(claims.iter().all(|claim| {
        claim.eq_lo.len() == block_len && claim.eq_hi.len() * block_len == out_len
    }));

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    if b_gfni_on && direct_fold8_l1_precommit_enabled() {
        if let Some(precommit) = l1_precommit {
            assert_eq!(
                out_len,
                (1usize << precommit.log_msg_cols) * (1usize << precommit.log_num_interleaved)
            );
            // Lane rs: with the seed6 L1 commit (CLMUL + NT publish, then
            // hash) the GFNI basis fold (+ its fused message) overlaps the
            // commit better than it overlaps fold64 (both p0/p5-heavy), so
            // the sequential-fold schedule below wins; see rs RESULT.md.
            if direct_fold8_b_overlap_enabled() && !rs_f8_rebal_enabled() {
                let t_f8 = std::time::Instant::now();
                crate::rstrace::ev("f8:start", 0);
                let mut folded_f = crate::scratch::take_f128(out_len);
                let folded_b = crate::scratch::take_f128(out_len);
                let ((), (folded_b, _)) = rayon::join(
                    || {
                        materialize_direct_fold8_f_for_precommit(
                            &packed_witness,
                            &mut folded_f,
                            block_len,
                            &fold16_weight,
                            r4,
                            r5,
                        );
                        crate::rstrace::ev("f8:fold64:end", 0);
                    },
                    || {
                        let r = materialize_direct_fold8_b_gfni_impl(
                            None, folded_b, claims, challenges, block_len,
                        );
                        crate::rstrace::ev("f8:basis:end", 0);
                        r
                    },
                );
                crate::rstrace::ev("f8:join1", 0);
                let t_f8a = t_f8.elapsed();
                crate::scratch::give_f128(packed_witness);
                crate::scratch::give_f128(ordinary_basis);
                let (precommitted, msg) = rayon::join(
                    || {
                        let ntt = AdditiveNttF128::standard(
                            precommit.log_msg_cols + precommit.log_inv_rate,
                        );
                        ligero_commit(
                            &folded_f,
                            precommit.log_msg_cols,
                            precommit.log_num_interleaved,
                            precommit.log_inv_rate,
                            &ntt,
                            precommit.kind,
                        )
                    },
                    || {
                        let m = direct_fold8_msg_par(&folded_f, &folded_b, block_len);
                        crate::rstrace::ev("f8:msg:end", 0);
                        m
                    },
                );
                crate::rstrace::ev("f8:join2", 0);
                if open_timing() {
                    eprintln!(
                        "[f8] overlap: f-fold+basis {:.3} ms, commit+msg {:.3} ms",
                        t_f8a.as_secs_f64() * 1e3,
                        (t_f8.elapsed() - t_f8a).as_secs_f64() * 1e3
                    );
                }
                return (folded_f, folded_b, msg, Some(precommitted));
            }
            let t_f8 = std::time::Instant::now();
            crate::rstrace::ev("f8s:start", 0);
            let mut folded_f = crate::scratch::take_f128(out_len);
            let t_f8a = t_f8.elapsed();
            materialize_direct_fold8_f_for_precommit(
                &packed_witness,
                &mut folded_f,
                block_len,
                &fold16_weight,
                r4,
                r5,
            );
            if open_timing() {
                eprintln!("[f8] take {:.3} ms, f-fold {:.3} ms over {} MiB in",
                    t_f8a.as_secs_f64()*1e3, (t_f8.elapsed()-t_f8a).as_secs_f64()*1e3, packed_witness.len()*16>>20);
            }
            crate::rstrace::ev("f8s:fold64", 0);
            let t_f8 = std::time::Instant::now();
            let folded_b = crate::scratch::take_f128(out_len);
            crate::scratch::give_f128(packed_witness);
            crate::scratch::give_f128(ordinary_basis);

            let (precommitted, (folded_b, msg)) = rayon::join(
                || {
                    let ntt =
                        AdditiveNttF128::standard(precommit.log_msg_cols + precommit.log_inv_rate);
                    ligero_commit(
                        &folded_f,
                        precommit.log_msg_cols,
                        precommit.log_num_interleaved,
                        precommit.log_inv_rate,
                        &ntt,
                        precommit.kind,
                    )
                },
                || {
                    let r = materialize_direct_fold8_b_gfni_for_precommit(
                        &folded_f, folded_b, claims, challenges, block_len,
                    );
                    crate::rstrace::ev("f8s:basis", 0);
                    r
                },
            );
            crate::rstrace::ev("f8s:join", 0);
            if open_timing() {
                eprintln!("[f8] give+join {:.3} ms", t_f8.elapsed().as_secs_f64()*1e3);
            }
            return (folded_f, folded_b, msg, Some(precommitted));
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let _ = l1_precommit;
    // Every route below reads the witness directly.
    super::elided_z::ensure_dense(&mut packed_witness, "fold8 non-precommit");

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_tables: Vec<Vec<F128>> = if b_gfni_on {
        Vec::new()
    } else {
        claims
            .par_iter()
            .map(|claim| {
                let generators = direct_fold8_final_generators(claim, challenges[5]);
                super::ring_switch::build_direct_fold8_table_from_generators(&generators)
            })
            .collect()
    };
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let direct_tables: Vec<Vec<F128>> = claims
        .par_iter()
        .map(|claim| {
            let generators = direct_fold8_final_generators(claim, challenges[5]);
            super::ring_switch::build_direct_fold8_table_from_generators(&generators)
        })
        .collect();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_rows: Vec<(Vec<u64>, Vec<u64>)> = if b_gfni_on {
        claims
            .iter()
            .map(|claim| split_eq_lo_halves_avx512(&claim.eq_lo))
            .collect()
    } else {
        Vec::new()
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_gfni_mats: Vec<super::ring_switch::GfniDirectFoldMap> = if b_gfni_on {
        claims
            .iter()
            .map(|claim| {
                let generators = direct_fold8_final_generators(claim, challenges[5]);
                super::ring_switch::build_gfni_direct_fold_map_from_generators(&generators)
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut folded_f = crate::scratch::take_f128(out_len);
    let mut folded_b = crate::scratch::take_f128(out_len);
    const SUB: usize = 256;
    const ALIGN64_MIN_F128: usize = (32 * 1024) / core::mem::size_of::<F128>();
    let stats = folded_b
        .par_chunks_mut(block_len)
        .zip(folded_f.par_chunks_mut(block_len))
        .enumerate()
        .map_init(
            || {
                let pooled = crate::scratch::fold_buf_pool_enabled();
                (
                    crate::scratch::LocalBuf::new(
                        if b_gfni_on {
                            0
                        } else {
                            super::ring_switch::FOLD_TABLE_TOTAL
                        },
                        pooled,
                    ),
                    crate::scratch::LocalBuf::new(if has_ordinary { 16 * SUB } else { 0 }, pooled),
                    crate::scratch::LocalBuf::new(
                        if has_ordinary {
                            (4 * SUB).max(ALIGN64_MIN_F128)
                        } else {
                            0
                        },
                        pooled,
                    ),
                    crate::scratch::LocalBuf::new(
                        if b_gfni_on {
                            (64usize).max(ALIGN64_MIN_F128)
                        } else {
                            0
                        },
                        pooled,
                    ),
                )
            },
            |(scratch, mid16, mid4, gfni_tmp), (block, (b_out, f_out))| {
                let _ = &gfni_tmp;
                let start = 64 * block * block_len;
                let f_in = &packed_witness[start..start + 64 * block_len];
                let b_in: &[F128] = if has_ordinary {
                    &ordinary_basis[start..start + 64 * block_len]
                } else {
                    &[]
                };
                crate::field::f128_slice::fold64_banked_nested(
                    f_in,
                    f_out,
                    &fold16_weight,
                    r4,
                    r5,
                );
                if has_ordinary {
                    let mut slot = 0usize;
                    while slot < block_len {
                        let n = SUB.min(block_len - slot);
                        let m16 = &mut mid16[..16 * n];
                        let m4 = &mut mid4[..4 * n];
                        crate::field::f128_slice::fold4_nested(
                            &b_in[64 * slot..64 * (slot + n)],
                            m16,
                            r0,
                            r1,
                        );
                        crate::field::f128_slice::fold4_nested(m16, m4, r2, r3);
                        crate::field::f128_slice::fold4_nested(
                            m4,
                            &mut b_out[slot..slot + n],
                            r4,
                            r5,
                        );
                        slot += n;
                    }
                }

                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512vbmi",
                    target_feature = "vpclmulqdq",
                    target_feature = "gfni"
                ))]
                if b_gfni_on {
                    use crate::zerocheck::multilinear::kernels::x86_64::gfni_fold64_four_maps_staged;
                    use crate::pcs::ring_switch::compose_block_mats_gfni_into;
                    let (claim0, claim1) = (&claims[0], &claims[1]);
                    let mut mats0_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats0_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats1_lo = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    let mut mats1_hi = core::mem::MaybeUninit::<[u64; 128]>::uninit();
                    unsafe {
                        compose_block_mats_gfni_into(
                            &direct_gfni_mats[0],
                            claim0.eq_hi[block],
                            &mut *mats0_lo.as_mut_ptr(),
                            &mut *mats0_hi.as_mut_ptr(),
                        );
                        compose_block_mats_gfni_into(
                            &direct_gfni_mats[1],
                            claim1.eq_hi[block],
                            &mut *mats1_lo.as_mut_ptr(),
                            &mut *mats1_hi.as_mut_ptr(),
                        );
                    }
                    let (rows0, rows1) = (&direct_gfni_rows[0], &direct_gfni_rows[1]);
                    for slot in (0..block_len).step_by(64) {
                        unsafe {
                            gfni_fold64_four_maps_staged(
                                rows0.0.as_ptr().add(slot).cast::<u8>(),
                                &*mats0_lo.as_ptr(),
                                rows0.1.as_ptr().add(slot).cast::<u8>(),
                                &*mats0_hi.as_ptr(),
                                rows1.0.as_ptr().add(slot).cast::<u8>(),
                                &*mats1_lo.as_ptr(),
                                rows1.1.as_ptr().add(slot).cast::<u8>(),
                                &*mats1_hi.as_ptr(),
                                b_out.as_mut_ptr().add(slot),
                                gfni_tmp.as_mut_ptr().cast(),
                            );
                        }
                    }
                }
                if !b_gfni_on {
                    let (first_claim, rest_claims) = claims.split_first().unwrap();
                    let (first_table, _) = direct_tables.split_first().unwrap();
                    super::ring_switch::compose_block_table(
                        first_table,
                        first_claim.eq_hi[block],
                        scratch,
                    );
                    for slot in 0..block_len {
                        let direct =
                            super::ring_switch::fold_one_slot(first_claim.eq_lo[slot], scratch);
                        b_out[slot] = if has_ordinary {
                            direct + b_out[slot]
                        } else {
                            direct
                        };
                    }
                    for (claim, table) in rest_claims.iter().zip(direct_tables.iter().skip(1)) {
                        super::ring_switch::compose_block_table(table, claim.eq_hi[block], scratch);
                        for (slot, out) in b_out.iter_mut().enumerate() {
                            *out += super::ring_switch::fold_one_slot(claim.eq_lo[slot], scratch);
                        }
                    }
                }
                #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
                {
                    unsafe { msg_reduce_avx512(f_out, b_out) }
                }
                #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
                {
                    super::round0_scalar(f_out, b_out)
                }
            },
        )
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
        );

    crate::scratch::give_f128(packed_witness);
    crate::scratch::give_f128(ordinary_basis);
    (
        folded_f,
        folded_b,
        SumcheckMessage {
            u_0: stats.0,
            u_2: stats.1,
        },
        None,
    )
}

pub struct SumcheckProver {
    f: FoldBuf,
    combined_basis: FoldBuf,
    fold_arena: Option<FoldArena>,
    t_r: F128,
    transcript: Vec<SumcheckMessage>,
    pending_glue: Option<(Vec<F128>, F128)>,
    pending_fold_basis: Option<(Vec<F128>, F128)>,
    pending_ood_eq: Option<PendingOodEq>,
}

impl SumcheckProver {
    pub fn new(f: Vec<F128>, b1: Vec<F128>, h1: F128) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(b1),
            fold_arena: None,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        };
        let msg = round_msg_lsb(&inst.f, &inst.combined_basis);
        inst.transcript.push(msg);
        (inst, msg)
    }

    pub fn set_fold_arena(&mut self, arena: FoldArena) {
        self.fold_arena = Some(arena);
    }

    pub(crate) fn f_is_arena(&self) -> bool {
        matches!(self.f, FoldBuf::Arena { .. })
    }

    pub fn new_with_first_msg(
        f: Vec<F128>,
        b1: Vec<F128>,
        h1: F128,
        first_msg: SumcheckMessage,
    ) -> (Self, SumcheckMessage) {
        assert_eq!(f.len(), b1.len());
        let mut inst = Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(b1),
            fold_arena: None,
            t_r: h1,
            transcript: Vec::new(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        };
        inst.transcript.push(first_msg);
        (inst, first_msg)
    }

    fn new_after_direct_fold4(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 5],
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    fn new_after_direct_fold2(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 3],
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    fn new_after_direct_fold8(
        f: Vec<F128>,
        basis: Vec<F128>,
        target: F128,
        transcript: [SumcheckMessage; 7],
        fold_arena: Option<FoldArena>,
    ) -> Self {
        assert_eq!(f.len(), basis.len());
        Self {
            f: FoldBuf::Owned(f),
            combined_basis: FoldBuf::Owned(basis),
            fold_arena,
            t_r: target,
            transcript: transcript.to_vec(),
            pending_glue: None,
            pending_fold_basis: None,
            pending_ood_eq: None,
        }
    }

    pub fn fold(&mut self, r: F128) -> SumcheckMessage {
        assert!(self.pending_glue.is_none(), "fold before ordinary glue");
        let pending_ood = self.pending_ood_eq.take();
        let pending_basis = self.pending_fold_basis.take();
        let (nf, nb, msg) = match (pending_ood, pending_basis) {
            (
                Some(PendingOodEq::Glued {
                    eq_lo,
                    eq_hi,
                    z_0,
                    beta,
                }),
                deferred_basis,
            ) => {
                let gamma = beta * (F128::ONE + z_0 + r);
                fold_and_msg_lsb_inner(
                    &self.f,
                    &self.combined_basis,
                    r,
                    self.fold_arena.as_mut(),
                    Some((&eq_lo, &eq_hi, gamma)),
                    deferred_basis
                        .as_ref()
                        .map(|(basis, alpha)| (basis.as_slice(), *alpha)),
                )
            }
            (Some(PendingOodEq::Introduced { .. }), _) => {
                panic!("fold before factorized OOD glue")
            }
            (None, Some(deferred_basis)) => fold_and_msg_lsb_inner(
                &self.f,
                &self.combined_basis,
                r,
                self.fold_arena.as_mut(),
                None,
                Some((&deferred_basis.0, deferred_basis.1)),
            ),
            (None, None) => {
                fold_and_msg_lsb(&self.f, &self.combined_basis, r, self.fold_arena.as_mut())
            }
        };
        #[cfg(target_arch = "x86_64")]
        {
            if let FoldBuf::Owned(v) = std::mem::replace(&mut self.f, nf) {
                crate::scratch::give_f128(v);
            }
            if let FoldBuf::Owned(v) = std::mem::replace(&mut self.combined_basis, nb) {
                crate::scratch::give_f128(v);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.f = nf;
            self.combined_basis = nb;
        }
        self.transcript.push(msg);
        msg
    }

    pub fn introduce_new(&mut self, b_new: Vec<F128>, h_new: F128) -> SumcheckMessage {
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary introduction across deferred basis"
        );
        assert_eq!(b_new.len(), self.f.len());
        let msg = round_msg_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        msg
    }

    fn introduce_sparse_dual(&mut self, msg: SumcheckMessage) {
        assert!(self.pending_glue.is_none());
        assert!(self.pending_fold_basis.is_none());
        self.transcript.push(msg);
    }

    fn glue_sparse_dual(&mut self, alpha: F128, h_new: F128) {
        assert!(self.pending_glue.is_none());
        assert!(self.pending_fold_basis.is_none());
        assert!(!matches!(
            self.pending_ood_eq,
            Some(PendingOodEq::Introduced { .. })
        ));
        self.t_r += alpha * h_new;
    }

    fn add_to_last_message(&mut self, delta: SumcheckMessage) -> SumcheckMessage {
        let msg = self.transcript.last_mut().expect("fold message missing");
        msg.u_0 += delta.u_0;
        msg.u_2 += delta.u_2;
        *msg
    }

    fn defer_folded_sparse_dual(&mut self, basis: Vec<F128>, alpha: F128) {
        assert!(self.pending_glue.is_none());
        assert!(self.pending_ood_eq.is_none());
        assert!(self.pending_fold_basis.is_none());
        assert_eq!(basis.len(), self.f.len());
        self.pending_fold_basis = Some((basis, alpha));
    }

    fn merge_folded_sparse_dual(&mut self, basis: Vec<F128>) {
        use rayon::prelude::*;
        assert!(self.pending_glue.is_none());
        assert!(self.pending_ood_eq.is_none());
        assert!(self.pending_fold_basis.is_none());
        assert_eq!(basis.len(), self.combined_basis.len());
        if self.combined_basis.len() <= 8192 {
            transpose_top_only(&mut self.combined_basis, &basis);
        } else {
            self.combined_basis
                .par_chunks_mut(4096)
                .zip(basis.par_chunks(4096))
                .for_each(|(dst, src)| transpose_top_only(dst, src));
        }
    }

    pub fn introduce_new_with_eval(&mut self, b_new: Vec<F128>) -> (SumcheckMessage, F128) {
        assert!(
            self.pending_fold_basis.is_none(),
            "OOD introduction across deferred basis"
        );
        assert_eq!(b_new.len(), self.f.len());
        let (msg, h_new) = round_msg_and_eval_lsb(&self.f, &b_new);
        self.transcript.push(msg);
        self.pending_glue = Some((b_new, h_new));
        (msg, h_new)
    }

    fn introduce_new_ood_factorized(&mut self, z: &[F128]) -> Option<(SumcheckMessage, F128)> {
        let expected_len = 1usize.checked_shl(z.len().try_into().ok()?);
        if z.is_empty()
            || expected_len != Some(self.f.len())
            || self.pending_glue.is_some()
            || self.pending_fold_basis.is_some()
            || self.pending_ood_eq.is_some()
        {
            return None;
        }
        let tail = &z[1..];
        let split = tail.len().min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
        let eq_lo = build_eq_table(&tail[..split]);
        let eq_hi = build_eq_table(&tail[split..]);
        let z_0 = z[0];
        let (msg, h_new) = round_msg_and_eval_lsb_factorized_eq(&self.f, &eq_lo, &eq_hi, z_0);
        self.transcript.push(msg);
        self.pending_ood_eq = Some(PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        });
        Some((msg, h_new))
    }

    fn glue_factorized_ood(&mut self, beta: F128) {
        assert!(
            self.pending_glue.is_none(),
            "factorized OOD glue across ordinary pending glue"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "factorized OOD glue across deferred ordinary basis"
        );
        let pending = self
            .pending_ood_eq
            .take()
            .expect("factorized OOD glue without introduction");
        let PendingOodEq::Introduced {
            eq_lo,
            eq_hi,
            z_0,
            h_new,
        } = pending
        else {
            panic!("factorized OOD equality glued twice");
        };
        self.t_r += beta * h_new;
        self.pending_ood_eq = Some(PendingOodEq::Glued {
            eq_lo,
            eq_hi,
            z_0,
            beta,
        });
    }

    pub fn glue(&mut self, alpha: F128) {
        use rayon::prelude::*;
        assert!(
            self.pending_fold_basis.is_none(),
            "ordinary glue across deferred ordinary basis"
        );
        let (b_new, h_new) = self
            .pending_glue
            .take()
            .expect("glue without introduce_new");
        assert_eq!(b_new.len(), self.combined_basis.len());
        const PAR_THRESHOLD: usize = 4096;
        #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
        let x4 = open_ood_x4_enabled();
        #[cfg(not(all(target_feature = "avx512f", target_feature = "vpclmulqdq")))]
        let x4 = false;
        if x4 {
            #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
            {
                const CHUNK: usize = 2048;
                let acc: &mut [F128] = &mut self.combined_basis;
                if acc.len() < PAR_THRESHOLD {
                    unsafe { glue_block_x4(acc, &b_new, alpha) };
                } else {
                    acc.par_chunks_mut(CHUNK)
                        .zip(b_new.par_chunks(CHUNK))
                        .for_each(|(a, v)| {
                            unsafe { glue_block_x4(a, v, alpha) }
                        });
                }
            }
        } else if self.combined_basis.len() < PAR_THRESHOLD {
            for (acc, &v) in self.combined_basis.iter_mut().zip(b_new.iter()) {
                *acc += alpha * v;
            }
        } else {
            self.combined_basis
                .par_iter_mut()
                .zip(b_new.par_iter())
                .with_min_len(PAR_THRESHOLD / 4)
                .for_each(|(acc, &v)| *acc += alpha * v);
        }
        self.t_r += alpha * h_new;
    }

    fn glue_deferred_into_factorized_ood_fold(&mut self, alpha: F128) {
        assert!(
            matches!(self.pending_ood_eq, Some(PendingOodEq::Glued { .. })),
            "deferred ordinary glue requires a glued factorized OOD term"
        );
        assert!(
            self.pending_fold_basis.is_none(),
            "more than one ordinary basis deferred"
        );
        let (basis, h_new) = self
            .pending_glue
            .take()
            .expect("deferred glue without ordinary introduction");
        assert_eq!(basis.len(), self.combined_basis.len());
        self.t_r += alpha * h_new;
        self.pending_fold_basis = Some((basis, alpha));
    }

    pub fn f(&self) -> &[F128] {
        &self.f
    }

    pub fn transcript(&self) -> &[SumcheckMessage] {
        &self.transcript
    }
}


/// Lane opus-pcs: `FLOCK_NO_OPUS_DUAL_JOIN=1` runs the sparse duals' fold
/// step after `sc_prover.fold` (incumbent order) instead of beside it.
/// Read once.
fn opus_dual_join_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_OPUS_DUAL_JOIN").is_none());
    *ON
}

/// Lane opus-pcs: what the sparse duals contribute to one recursive fold —
/// message deltas (to be XOR-added into the fold's message, in order) and at
/// most one materialized basis (to be deferred or merged by the caller).
struct SparseDualFoldWork {
    l0_delta: Option<SumcheckMessage>,
    l0_basis: Option<Vec<F128>>,
    /// Sum of the deep duals' round deltas (F128 addition is XOR, so the
    /// sum equals adding them one by one; no per-round Vec).
    deep_delta: Option<SumcheckMessage>,
    deep_basis: Option<Vec<F128>>,
}

/// Lane opus-pcs: whether the next fold step materializes any sparse dual
/// (its challenge count reaches the dual's depth with this round).
fn sparse_duals_materialize_next(
    l0: &Option<(SparseDualL0, Vec<F128>)>,
    deep: &[(SparseDualL0, Vec<F128>)],
) -> bool {
    l0.as_ref().is_some_and(|(dual, fc)| fc.len() + 1 == dual.depth)
        || deep.iter().any(|(dual, fc)| fc.len() + 1 == dual.depth)
}

/// Lane opus-pcs: the duals' half of one recursive fold, exactly the
/// incumbent per-dual sequence (push challenge, round step while within
/// depth, materialize at depth) minus every `SumcheckProver` mutation, which
/// the caller applies afterwards in the incumbent order. Touches only the
/// duals, so it may run concurrently with `SumcheckProver::fold`.
fn sparse_duals_fold_step(
    l0: &mut Option<(SparseDualL0, Vec<F128>)>,
    deep: &mut [(SparseDualL0, Vec<F128>)],
    ri: F128,
) -> SparseDualFoldWork {
    let mut work = SparseDualFoldWork {
        l0_delta: None,
        l0_basis: None,
        deep_delta: None,
        deep_basis: None,
    };
    if let Some((dual, fold_challenges)) = l0.as_mut() {
        fold_challenges.push(ri);
        if fold_challenges.len() <= dual.depth {
            work.l0_delta = Some(dual.round_msg_step(fold_challenges));
        }
        if fold_challenges.len() == dual.depth {
            work.l0_basis = Some(dual.materialize_after_folds(fold_challenges));
        }
    }
    let l0_materialized_here = work.l0_basis.is_some();
    for (dual, fold_challenges) in deep.iter_mut() {
        fold_challenges.push(ri);
        if fold_challenges.len() <= dual.depth {
            let d = dual.round_msg_step(fold_challenges);
            work.deep_delta = Some(match work.deep_delta {
                Some(acc) => SumcheckMessage {
                    u_0: acc.u_0 + d.u_0,
                    u_2: acc.u_2 + d.u_2,
                },
                None => d,
            });
        }
        if fold_challenges.len() == dual.depth {
            assert!(
                !l0_materialized_here && work.deep_basis.is_none(),
                "two sparse duals materialize into one fold"
            );
            work.deep_basis = Some(dual.materialize_after_folds(fold_challenges));
        }
    }
    work
}

fn serial_par_enabled() -> bool {
    crate::serial_par_enabled()
}

const ROW_GATHER_PAR_MIN_ELEMS: usize = 512;

/// Lane lf1: `FLOCK_LF1_QS_HASHSET=1` restores the SipHash set in
/// `sample_distinct_queries` (default: bitmap). Read once.
fn lf1_qs_bitmap() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_LF1_QS_HASHSET").is_none());
    *ON
}

/// Lane lf1: the recursive levels' opened rows and multiproof siblings
/// (~100 rows of 128 B, ~1K hashes) are gathered serially on the joined
/// thread; `FLOCK_NO_LF1_OPEN_SERIAL=1` restores the rayon gathers. Read once.
fn lf1_open_par() -> bool {
    static PAR: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LF1_OPEN_SERIAL").is_some());
    serial_par_enabled() && *PAR
}

/// Lane lf1: the sparse dual's query split takes at least 8 queries per
/// rayon task (`FLOCK_LF1_DUAL_MINLEN=k` overrides, 1 = rayon adaptive as
/// before). Read once.
fn lf1_dual_minlen() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("FLOCK_LF1_DUAL_MINLEN").ok().and_then(|s| s.parse().ok()).filter(|&v| v >= 1).unwrap_or(8)
    });
    *V
}

fn gather_opened_rows<'a, F>(queries: &[usize], row: F, par: bool) -> Vec<Vec<F128>>
where
    F: Fn(usize) -> &'a [F128] + Sync,
{
    use rayon::prelude::*;
    let row_len = queries.first().map_or(0, |&q| row(q).len());
    if par && queries.len() * row_len >= ROW_GATHER_PAR_MIN_ELEMS {
        queries.par_iter().map(|&q| row(q).to_vec()).collect()
    } else {
        queries.iter().map(|&q| row(q).to_vec()).collect()
    }
}

const MULTIPROOF_PAR_MIN_SIBLINGS: usize = 512;

const MULTIPROOF_PAR_CHUNK: usize = 64;

/// Lane gl2: `FLOCK_NO_GL2_DEFER=1` keeps the opened-row + multiproof gathers
/// of L1 and the recursive levels inside the `rayon::join` with the sparse
/// dual (incumbent). Default: they run as detached jobs off the open's
/// critical path -- neither enters the transcript, so they only have to be
/// complete when the proof is assembled. Read once per open.
fn gl2_defer_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL2_DEFER").is_none());
    *ON
}

type Gl2Opened = (Vec<Vec<F128>>, Vec<Hash>);

/// Lane gl2: `FLOCK_NO_GL2_SIDE=1` dispatches the deferred gathers and the
/// streamed-dual helpers with `rayon::spawn` (onto the owner's deque, where
/// the owner's own join waits can pop them) instead of the hot-pool side
/// list. Read once.
fn gl2_side_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL2_SIDE").is_none());
    *ON
}

/// Lane gl2: streamed-dual tuning, read once: helper count (default
/// threads - 2) and the pause count between a waiting helper's polls of the
/// owner's publication counter (default 16).
fn gl2_stream_knobs() -> (usize, usize) {
    static K: std::sync::LazyLock<(usize, usize)> = std::sync::LazyLock::new(|| {
        let h = std::env::var("FLOCK_GL2_STREAM_HELPERS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let p = std::env::var("FLOCK_GL2_STREAM_POLL").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        (h, p)
    });
    *K
}

#[repr(align(128))]
struct Gl2Pad(std::sync::atomic::AtomicUsize);

#[repr(align(64))]
#[derive(Clone, Copy)]
struct Gl2Line([usize; 8]);

/// A detached open-arm gather. Claimed exactly once (by the rayon job or by
/// the waiter, whichever comes first), so a waiter never blocks on a job that
/// no worker has started.
#[repr(align(128))]
struct Gl2Deferred {
    state: std::sync::atomic::AtomicU8,
    job: std::cell::UnsafeCell<Option<Box<dyn FnOnce() -> Gl2Opened + Send>>>,
    out: std::cell::UnsafeCell<Option<Gl2Opened>>,
    side: std::sync::Mutex<Option<crate::hot_pool::side::Task>>,
}

// SAFETY: `job` is taken only by the thread that wins the 0 -> 1 claim, `out`
// is written by that thread before the release store of 2 and read only after
// an acquire load of 2.
unsafe impl Sync for Gl2Deferred {}
unsafe impl Send for Gl2Deferred {}

impl Gl2Deferred {
    fn spawn(job: Box<dyn FnOnce() -> Gl2Opened + Send>) -> std::sync::Arc<Self> {
        let d = std::sync::Arc::new(Self {
            state: std::sync::atomic::AtomicU8::new(0),
            job: std::cell::UnsafeCell::new(Some(job)),
            out: std::cell::UnsafeCell::new(None),
            side: std::sync::Mutex::new(None),
        });
        if gl2_side_enabled() {
            // Run by an idle hot-pool spinner; never popped by the owner's
            // own join waits (a `rayon::spawn` lands on the owner's deque).
            let d2 = d.clone();
            let t: crate::hot_pool::side::Task = std::sync::Arc::new(move |_sib: bool| {
                if d2.state.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                    d2.run();
                }
            });
            *d.side.lock().unwrap_or_else(|e| e.into_inner()) = Some(t.clone());
            crate::hot_pool::side::register(t);
        } else {
            let d2 = d.clone();
            rayon::spawn(move || d2.run());
        }
        d
    }

    fn run(&self) {
        use std::sync::atomic::Ordering;
        if self.state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return;
        }
        // SAFETY: claim won above; see the `Sync` impl.
        let job = unsafe { (*self.job.get()).take() }.expect("deferred job claimed twice");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).ok();
        unsafe { *self.out.get() = r };
        self.state.store(2, Ordering::Release);
    }

    fn take_side(&self) -> Option<crate::hot_pool::side::Task> {
        self.side.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    fn wait(&self) -> Option<Gl2Opened> {
        use std::sync::atomic::Ordering;
        self.run();
        while self.state.load(Ordering::Acquire) != 2 {
            std::hint::spin_loop();
        }
        // SAFETY: state 2 observed with acquire; only the single waiter takes.
        unsafe { (*self.out.get()).take() }
    }
}

/// Waits for every pending deferred gather when dropped (also on unwind), so
/// no job can outlive the witnesses / L0 buffers its raw pointers refer to.
/// Declare it AFTER the storage those jobs read.
struct Gl2DeferSet(Vec<std::sync::Arc<Gl2Deferred>>);

impl Drop for Gl2DeferSet {
    fn drop(&mut self) {
        for d in &self.0 {
            let _ = d.wait();
        }
    }
}

#[derive(Clone, Copy)]
struct Gl2Ptr<T>(*const T, usize);
// SAFETY: only ever dereferenced as a shared slice while the owner is kept
// alive by `Gl2DeferSet`'s ordering contract.
unsafe impl<T> Send for Gl2Ptr<T> {}
impl<T> Gl2Ptr<T> {
    fn of(s: &[T]) -> Self {
        Self(s.as_ptr(), s.len())
    }
    /// # Safety
    /// The slice this was made from must still be alive and unmodified.
    unsafe fn get<'a>(self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.0, self.1) }
    }
}

/// Lane gl2: the dedupe set of query sampling. A 4 KiB open-addressing table
/// (thread-local, cache-resident, reset after use) instead of a block_len-bit
/// bitmap: the L1 bitmap is 128 KiB, a fresh calloc is an mmap + page faults
/// + munmap, and a reused one is cache-cold at every probe.
struct Gl2QSet {
    slots: [u32; 1024],
}

impl Gl2QSet {
    const EMPTY: u32 = u32::MAX;

    /// Insert `q`; true when it was not present. Needs `q < u32::MAX` and
    /// fewer than 1024 members.
    #[inline]
    fn insert(&mut self, q: usize) -> bool {
        let key = q as u32;
        let mut i = (key.wrapping_mul(0x9E37_79B1) >> 22) as usize;
        loop {
            let s = self.slots[i];
            if s == key {
                return false;
            }
            if s == Self::EMPTY {
                self.slots[i] = key;
                return true;
            }
            i = (i + 1) & 1023;
        }
    }

    fn with<R>(f: impl FnOnce(&mut Self) -> R) -> R {
        thread_local! {
            static SET: std::cell::RefCell<Box<Gl2QSet>> =
                std::cell::RefCell::new(Box::new(Gl2QSet { slots: [Gl2QSet::EMPTY; 1024] }));
        }
        SET.with(|c| {
            let mut set = c.borrow_mut();
            let r = f(&mut set);
            set.slots.fill(Self::EMPTY);
            r
        })
    }
}

/// Lane gl2: `FLOCK_NO_GL2_QS_REUSE=1` restores lf1's per-call calloc'd
/// query bitmap (default: [`Gl2QSet`] when `count <= 256` and
/// `block_len < 2^32`). Read once.
fn gl2_qs_reuse() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL2_QS_REUSE").is_none());
    *ON
}

fn sample_distinct_queries<Ch: Challenger>(
    challenger: &mut Ch,
    block_len: usize,
    count: usize,
) -> Vec<usize> {
    assert!(
        count <= block_len,
        "sample_distinct_queries: count ({count}) > block_len ({block_len}) — config is too thin for this query count"
    );
    let mut out = Vec::with_capacity(count);
    if gl2_qs_reuse() && count <= 256 && block_len < u32::MAX as usize {
        // Lane gl2: same draws, same first-occurrence dedupe, small hash set.
        Gl2QSet::with(|seen| {
            crate::rstrace::evd("qs:begin", count as u32);
            while out.len() < count {
                let v = challenger.sample_f128();
                let q = (v.lo as usize) % block_len;
                if seen.insert(q) {
                    out.push(q);
                }
            }
            crate::rstrace::evd("qs:drawn", count as u32);
        });
        out.sort_unstable();
        return out;
    }
    if lf1_qs_bitmap() {
        // Lane lf1: same draws, same first-occurrence dedupe, bitmap
        // membership instead of a SipHash set.
        let mut seen = vec![0u64; block_len.div_ceil(64)];
        crate::rstrace::evd("qs:begin", count as u32);
        while out.len() < count {
            let v = challenger.sample_f128();
            let q = (v.lo as usize) % block_len;
            let (w, b) = (q >> 6, 1u64 << (q & 63));
            if seen[w] & b == 0 {
                seen[w] |= b;
                out.push(q);
            }
        }
        crate::rstrace::evd("qs:drawn", count as u32);
        out.sort_unstable();
        return out;
    }
    let mut seen = std::collections::HashSet::new();
    while out.len() < count {
        let v = challenger.sample_f128();
        let q = (v.lo as usize) % block_len;
        if seen.insert(q) {
            out.push(q);
        }
    }
    out.sort_unstable();
    out
}

fn merkle_multi_proof_for(tree: &[Hash], block_len: usize, queries: &[usize]) -> Vec<Hash> {
    multi_proof_gather(tree, block_len, queries, serial_par_enabled())
}

#[inline]
fn ranked_l0_auth_batch_enabled(
    tree_len: usize,
    block_len: usize,
    num_interleaved: usize,
    n_queries: usize,
    kind: HashKind,
) -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_L0_AUTH_BATCH").is_none());
    cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
        && *ON
        && tree_len == block_len - 1
        && block_len == (1 << 20)
        && num_interleaved == 64
        && n_queries == 218
        && kind == HashKind::Blake3
}

fn merkle_multi_proof_for_l0(
    tree: &[Hash],
    codeword: &[F128],
    block_len: usize,
    num_interleaved: usize,
    queries: &[usize],
    kind: HashKind,
) -> Vec<Hash> {
    if tree.len() == 2 * block_len - 1 {
        return merkle_multi_proof_for(tree, block_len, queries);
    }
    assert_eq!(tree.len(), block_len - 1, "unsupported compact L0 tree");
    assert_eq!(codeword.len(), block_len * num_interleaved);

    use rayon::prelude::*;
    let indices = merkle::merkle_multi_proof_sibling_indices(block_len, queries);
    let leaf_size = num_interleaved * core::mem::size_of::<F128>();
    let bytes = unsafe {
        core::slice::from_raw_parts(
            codeword.as_ptr().cast::<u8>(),
            codeword.len() * core::mem::size_of::<F128>(),
        )
    };

    if ranked_l0_auth_batch_enabled(
        tree.len(),
        block_len,
        num_interleaved,
        queries.len(),
        kind,
    ) {
        let leaf_siblings = indices.partition_point(|&i| i < block_len);
        debug_assert!(indices[leaf_siblings..].iter().all(|&i| i >= block_len));
        let mut proof: Vec<Hash> = crate::alloc_uninit_vec(indices.len());
        debug_assert_eq!(leaf_size, 1024);
        let (leaf_out, stored_out) = proof.split_at_mut(leaf_siblings);
        let leaf_indices = &indices[..leaf_siblings];
        let stored_indices = &indices[leaf_siblings..];
        rayon::join(
            || merkle::hash_indexed_blake3_1k(bytes, leaf_indices, leaf_out),
            || {
                if serial_par_enabled()
                    && stored_indices.len() >= MULTIPROOF_PAR_MIN_SIBLINGS
                {
                    stored_out
                        .par_iter_mut()
                        .zip(stored_indices.par_iter())
                        .with_min_len(MULTIPROOF_PAR_CHUNK)
                        .for_each(|(out, &i)| *out = tree[i - block_len]);
                } else {
                    for (out, &i) in stored_out.iter_mut().zip(stored_indices) {
                        *out = tree[i - block_len];
                    }
                }
            },
        );
        return proof;
    }

    let gather = |&i: &usize| {
        if i < block_len {
            let start = i * leaf_size;
            merkle::hash_leaf(&bytes[start..start + leaf_size], kind)
        } else {
            tree[i - block_len]
        }
    };

    if serial_par_enabled() && indices.len() >= MULTIPROOF_PAR_MIN_SIBLINGS {
        indices
            .par_iter()
            .with_min_len(MULTIPROOF_PAR_CHUNK)
            .map(gather)
            .collect()
    } else {
        indices.iter().map(gather).collect()
    }
}

fn multi_proof_gather(tree: &[Hash], block_len: usize, queries: &[usize], par: bool) -> Vec<Hash> {
    use rayon::prelude::*;
    if par {
        let indices = merkle::merkle_multi_proof_sibling_indices(block_len, queries);
        if indices.len() >= MULTIPROOF_PAR_MIN_SIBLINGS {
            return indices
                .par_iter()
                .with_min_len(MULTIPROOF_PAR_CHUNK)
                .map(|&i| tree[i])
                .collect();
        }
    }
    merkle::merkle_multi_proof(tree, block_len, queries)
}

pub fn recursive_prover<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = crate::sl2_env_set!("LIGERITO_TRACE");
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let mut t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(
        config.log_inv_rates.len(),
        r + 1,
        "log_inv_rates must have R+1 entries"
    );
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
    let t = std::time::Instant::now();
    let wtns_0 = ligero_commit(
        poly,
        log_msg_cols_0,
        initial_k,
        log_inv_rate_0,
        &ntt_0,
        config.merkle_hash,
    );
    let t_l0 = t.elapsed();
    t_commits += t_l0;
    tlog!("  [ligerito]   L0 commit: {:.2?}", t_l0);
    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

pub fn recursive_prover_with_l0<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    l0_codeword: Vec<F128>,
    l0_tree: Vec<Hash>,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> LigeritoProof {
    let trace = crate::sl2_env_set!("LIGERITO_TRACE");
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    let t_total = std::time::Instant::now();
    let t_commits = std::time::Duration::ZERO;
    let t_induce = std::time::Duration::ZERO;
    let t_sumcheck = std::time::Duration::ZERO;
    let t_opens = std::time::Duration::ZERO;

    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;

    assert_eq!(poly.len(), 1usize << log_n);
    assert_eq!(eval_point.len(), log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1, "recursive_steps must be ≥ 1");

    let block_len = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved = 1usize << initial_k;
    let _ = r; // used implicitly via config in inner
    assert_eq!(
        l0_codeword.len(),
        block_len * num_interleaved,
        "external L0 codeword wrong size"
    );
    assert_eq!(
        l0_tree.len(),
        2 * block_len - 1,
        "external L0 tree wrong size"
    );

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    let wtns_0 = LigeroWitness {
        mat: l0_codeword,
        tree: l0_tree,
        block_len,
        num_interleaved,
    };
    tlog!("  [ligerito]   L0 commit: REUSED (skipped)");

    recursive_prover_inner(
        config,
        poly,
        wtns_0,
        eval_point,
        claimed_value,
        challenger,
        t_total,
        t_commits,
        t_induce,
        t_sumcheck,
        t_opens,
        trace,
    )
}

pub fn recursive_prover_with_basis<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_prover_with_basis_precomputed_round0<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        b_initial,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        fold_arena,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_ab_fold2<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold2Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    round1_lookahead: [F128; 6],
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        Some(round1_lookahead),
        None,
        None,
        None,
        None,
        Some(direct),
        None,
        None,
        fold_arena,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_fold4<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold4Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    round1_lookahead: [F128; 6],
    round2_lookahead: super::Fold4Lookahead2,
    round3_lookahead: super::Fold4Lookahead3,
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    assert!(
        config.initial_k >= 4,
        "direct-fold4 scaffold requires initial_k >= 4"
    );
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        Some(round1_lookahead),
        Some(round2_lookahead),
        Some(round3_lookahead),
        None,
        None,
        None,
        Some(direct),
        None,
        fold_arena,
        challenger,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn recursive_prover_with_basis_direct_fold8<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    ordinary_basis: Vec<F128>,
    direct: Vec<super::ring_switch::DirectFold8Factors>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    round0_uv: (F128, F128),
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    assert_eq!(
        config.initial_k, 6,
        "direct-fold8 scaffold requires initial_k=6"
    );
    recursive_prover_with_basis_impl(
        config,
        packed_witness,
        ordinary_basis,
        target,
        l0_codeword,
        l0_tree,
        Some(SumcheckMessage {
            u_0: round0_uv.0,
            u_2: round0_uv.1,
        }),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(direct),
        fold_arena,
        challenger,
    )
}

#[allow(clippy::too_many_arguments)]
fn recursive_prover_with_basis_impl<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    first_msg: Option<SumcheckMessage>,
    round1_lookahead: Option<[F128; 6]>,
    round2_lookahead: Option<super::Fold4Lookahead2>,
    round3_lookahead: Option<super::Fold4Lookahead3>,
    _round4_lookahead: Option<super::Fold8Lookahead4>,
    _round5_lookahead: Option<super::Fold8Lookahead5>,
    direct_fold2: Option<Vec<super::ring_switch::DirectFold2Factors>>,
    direct_fold4: Option<Vec<super::ring_switch::DirectFold4Factors>>,
    direct_fold8: Option<Vec<super::ring_switch::DirectFold8Factors>>,
    fold_arena: Option<FoldArena>,
    challenger: &mut Ch,
) -> LigeritoProof {
    let log_n = packed_witness.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(packed_witness.len(), 1usize << log_n);
    assert!(
        direct_fold2.is_none() || (direct_fold4.is_none() && direct_fold8.is_none()),
        "direct-fold2 and direct-fold4/fold8 modes are mutually exclusive"
    );
    assert!(
        direct_fold4.is_none() || direct_fold8.is_none(),
        "direct-fold4 and direct-fold8 modes are mutually exclusive"
    );
    if direct_fold2.is_some() || direct_fold4.is_some() || direct_fold8.is_some() {
        assert!(b_initial.is_empty() || b_initial.len() == 1usize << log_n);
    } else {
        assert_eq!(b_initial.len(), 1usize << log_n);
    }
    if direct_fold8.is_some() {
        assert_eq!(initial_k, 6, "direct-fold8 scaffold requires initial_k=6");
    }
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    assert_eq!(l0_codeword.len(), block_len_0 * num_interleaved_0);
    assert!(
        l0_tree.len() == 2 * block_len_0 - 1 || l0_tree.len() == block_len_0 - 1,
        "external L0 tree has unsupported layout"
    );

    let trace = crate::sl2_env_set!("LIG_PROVE_TRACE") || open_timing();
    let mut t_init_sumcheck = std::time::Duration::ZERO;
    let mut t_commits = std::time::Duration::ZERO;
    let mut t_opens = std::time::Duration::ZERO;
    let mut t_induce = std::time::Duration::ZERO;
    let mut t_sumcheck_folds = std::time::Duration::ZERO;
    let mut t_intro_glue = std::time::Duration::ZERO;
    let mut t_ood = std::time::Duration::ZERO;

    let t_total = std::time::Instant::now();
    crate::rstrace::ev("lig:start", 0);

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);

    let initial_root: Hash = l0_tree[l0_tree.len() - 1];
    let l0_block_len = block_len_0;
    let l0_num_interleaved = num_interleaved_0;
    let l0_row = |q: usize| -> &[F128] {
        let start = q * l0_num_interleaved;
        &l0_codeword[start..start + l0_num_interleaved]
    };
    challenger.observe_bytes(&initial_root);

    assert_eq!(
        config.ood_samples.first().copied().unwrap_or(0),
        0,
        "L0 must not take explicit OOD samples"
    );
    let mut ood_values: Vec<F128> = Vec::new();
    let mut fold_grinding_nonces: Vec<u64> = Vec::new();
    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };

    let _t = std::time::Instant::now();
    assert!(
        direct_fold2.is_none() || (direct_fold4.is_none() && direct_fold8.is_none()),
        "direct-fold2 and direct-fold4/fold8 modes are mutually exclusive"
    );
    assert!(direct_fold4.is_none() || direct_fold8.is_none());
    let direct_fold4_mode = direct_fold4.is_some();
    let direct_fold8_mode = direct_fold8.is_some();
    let direct_mode = direct_fold2.is_some() || direct_fold4_mode || direct_fold8_mode;
    if direct_fold8_mode {
        assert_eq!(initial_k, 6, "direct fold8 needs six initial rounds");
    } else if direct_fold4_mode {
        assert!(initial_k >= 4, "direct fold4 needs four initial rounds");
    } else if direct_mode {
        assert!(initial_k >= 2, "direct AB fold2 needs two initial rounds");
    }
    let mut packed_witness = Some(packed_witness);
    let mut b_initial = Some(b_initial);
    let mut direct_fold2 = direct_fold2;
    let mut direct_fold4 = direct_fold4;
    let mut direct_fold8 = direct_fold8;
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let l1_precommit_candidate = direct_fold8_mode
        && log_n == 25
        && initial_k == 6
        && log_msg_cols_1 == 16
        && log_num_interleaved_1 == 3
        && log_inv_rate_1 == 2
        && config.merkle_hash == HashKind::Blake3
        && direct_fold8_l1_precommit_enabled()
        && direct_fold8_b_gfni_enabled()
        && b_initial.as_ref().is_some_and(|basis| basis.is_empty())
        && direct_fold8
            .as_ref()
            .is_some_and(|claims| claims.len() == 2 && claims[0].eq_lo.len().is_multiple_of(64));
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let l1_precommit_candidate = false;
    // Only the direct-fold8 L1-precommit route can fold an elided witness
    // from its regenerator; every other route reads it directly.
    if !l1_precommit_candidate {
        if let Some(w) = packed_witness.as_mut() {
            super::elided_z::ensure_dense(w, "ligerito non-fold8");
        }
    }
    let mut early_wtns_1: Option<LigeroWitness> = None;
    let mut fold4_challenges: Vec<F128> = Vec::with_capacity(6);
    let mut fold4_msgs: Vec<SumcheckMessage> = Vec::with_capacity(6);
    let mut fold_arena = fold_arena;
    let (mut sc_prover, start_msg) = if direct_mode {
        (
            None,
            first_msg.expect("direct mode requires a precomputed round-zero message"),
        )
    } else {
        let (mut prover, msg) = match first_msg {
            Some(msg) => SumcheckProver::new_with_first_msg(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                target,
                msg,
            ),
            None => SumcheckProver::new(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                target,
            ),
        };
        if let Some(arena) = fold_arena.take() {
            prover.set_fold_arena(arena);
        }
        (Some(prover), msg)
    };
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    let mut direct_r0 = None;
    let mut direct_msg1 = None;
    let mut round_diag: Vec<(f64, f64, bool)> = Vec::new();
    let mut induce_diag: Vec<(usize, usize, usize, f64, bool)> = Vec::new();
    for j in 0..initial_k {
        let _tg = std::time::Instant::now();
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            fold_grinding_nonces.push(challenger.grind_pow(bits));
        }
        let grind_ms = _tg.elapsed().as_secs_f64() * 1e3;
        crate::rstrace::ev("r:grind", j as u32);
        let r = challenger.sample_f128();
        let _tf = std::time::Instant::now();
        let msg = if direct_fold8_mode && j < 6 {
            let msg = if j < 5 {
                fold_direct_fold8_factors_and_message(
                    direct_fold8
                        .as_mut()
                        .expect("direct-fold8 factors remain live through round five"),
                    r,
                )
            } else {
                let direct = direct_fold8
                    .take()
                    .expect("direct-fold8 factors consumed once");
                let l1_precommit = l1_precommit_candidate.then_some(DirectFold8L1Precommit {
                    log_msg_cols: log_msg_cols_1,
                    log_num_interleaved: log_num_interleaved_1,
                    log_inv_rate: log_inv_rate_1,
                    kind: config.merkle_hash,
                });
                let (f8, b8, msg, precommitted) = materialize_direct_fold8(
                    packed_witness.take().unwrap(),
                    b_initial.take().unwrap(),
                    &direct,
                    [
                        fold4_challenges[0],
                        fold4_challenges[1],
                        fold4_challenges[2],
                        fold4_challenges[3],
                        fold4_challenges[4],
                        r,
                    ],
                    l1_precommit,
                );
                early_wtns_1 = precommitted;
                debug_assert_eq!(l1_precommit_candidate, early_wtns_1.is_some());
                sc_prover = Some(SumcheckProver::new_after_direct_fold8(
                    f8,
                    b8,
                    target,
                    [
                        start_msg,
                        fold4_msgs[0],
                        fold4_msgs[1],
                        fold4_msgs[2],
                        fold4_msgs[3],
                        fold4_msgs[4],
                        msg,
                    ],
                    fold_arena.take(),
                ));
                msg
            };
            fold4_challenges.push(r);
            fold4_msgs.push(msg);
            msg
        } else if direct_fold4_mode && j < 4 {
            let msg = match j {
                0 => eval_lookahead(
                    round1_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-one lookahead"),
                    r,
                ),
                1 => eval_fold4_lookahead2(
                    round2_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-two lookahead"),
                    fold4_challenges[0],
                    r,
                ),
                2 => eval_fold4_lookahead3(
                    round3_lookahead
                        .as_ref()
                        .expect("direct-fold4 requires round-three lookahead"),
                    fold4_challenges[0],
                    fold4_challenges[1],
                    r,
                ),
                _ => {
                    let direct = direct_fold4
                        .take()
                        .expect("direct-fold4 factors consumed once");
                    let (f4, b4, msg) = materialize_direct_fold4(
                        packed_witness.take().unwrap(),
                        b_initial.take().unwrap(),
                        &direct,
                        [
                            fold4_challenges[0],
                            fold4_challenges[1],
                            fold4_challenges[2],
                            r,
                        ],
                    );
                    sc_prover = Some(SumcheckProver::new_after_direct_fold4(
                        f4,
                        b4,
                        target,
                        [start_msg, fold4_msgs[0], fold4_msgs[1], fold4_msgs[2], msg],
                        fold_arena.take(),
                    ));
                    msg
                }
            };
            fold4_challenges.push(r);
            fold4_msgs.push(msg);
            msg
        } else if direct_mode && j == 0 {
            let msg = eval_lookahead(
                round1_lookahead
                    .as_ref()
                    .expect("direct AB fold2 requires round-one lookahead"),
                r,
            );
            direct_r0 = Some(r);
            direct_msg1 = Some(msg);
            msg
        } else if direct_mode && j == 1 {
            let direct = direct_fold2
                .take()
                .expect("direct AB factors consumed once");
            let (f2, b2, msg) = materialize_direct_ab_fold2(
                packed_witness.take().unwrap(),
                b_initial.take().unwrap(),
                &direct,
                direct_r0.take().unwrap(),
                r,
            );
            sc_prover = Some(SumcheckProver::new_after_direct_fold2(
                f2,
                b2,
                target,
                [start_msg, direct_msg1.take().unwrap(), msg],
                fold_arena.take(),
            ));
            msg
        } else {
            sc_prover.as_mut().unwrap().fold(r)
        };
        if trace {
            round_diag.push((
                grind_ms,
                _tf.elapsed().as_secs_f64() * 1e3,
                sc_prover.as_ref().is_some_and(SumcheckProver::f_is_arena),
            ));
        }
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        crate::rstrace::ev("r:fold", j as u32);
        r_lane_fold.push(r);
    }
    if trace {
        t_init_sumcheck += _t.elapsed();
    }
    let mut sc_prover = sc_prover.expect("direct state must materialize during the initial rounds");

    let wtns_1 = match early_wtns_1.take() {
        Some(witness) => witness,
        None => {
            let _t = std::time::Instant::now();
            let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
            let witness = ligero_commit(
                sc_prover.f(),
                log_msg_cols_1,
                log_num_interleaved_1,
                log_inv_rate_1,
                &ntt_1,
                config.merkle_hash,
            );
            if trace {
                t_commits += _t.elapsed();
            }
            witness
        }
    };
    challenger.observe_bytes(&wtns_1.root());
    crate::rstrace::ev("L1:root", 0);

    let lazy_l1_ood = ranked_l1_lazy_ood_eq_enabled(
        config,
        log_n,
        n1,
        ood_count(1),
        sc_prover.f().len(),
        direct_fold4_mode || direct_fold8_mode,
    );

    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);
            let (intro, y) = if lazy_l1_ood {
                sc_prover
                    .introduce_new_ood_factorized(&z)
                    .expect("ranked factorized OOD introduction shape changed")
            } else {
                let eq_z = build_eq_table_split(&z);
                sc_prover.introduce_new_with_eval(eq_z)
            };
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            if lazy_l1_ood {
                sc_prover.glue_factorized_ood(beta);
            } else {
                sc_prover.glue(beta);
            }
        }
        if trace {
            t_ood += _t.elapsed();
        }
    }

    crate::rstrace::ev("L1:ood", 0);
    // Lane gl2: retired level witnesses stay alive until every deferred
    // gather has been waited for (`gl2_jobs` is dropped first).
    let gl2_defer = gl2_defer_enabled();
    let mut gl2_retired: Vec<LigeroWitness> = Vec::new();
    let mut gl2_jobs = Gl2DeferSet(Vec::new());
    let mut gl2_l1_job: Option<std::sync::Arc<Gl2Deferred>> = None;
    let mut gl2_level_jobs: Vec<(usize, std::sync::Arc<Gl2Deferred>)> = Vec::new();
    let mut gl2_side_done: Vec<crate::hot_pool::side::Task> = Vec::new();
    let pow_nonce_0 = challenger.grind_pow(config.grinding_bits[0] as u32);
    crate::rstrace::ev("L1:qgrind", 0);
    let mut grinding_nonces: Vec<u64> = vec![pow_nonce_0];

    let num_queries_0 = config.queries[0];
    let sparse_dual_depth = ranked_sparse_dual_l0_depth(
        config,
        log_n,
        n1,
        sc_prover.f().len(),
        num_queries_0,
        log_inv_rate_0,
        lazy_l1_ood,
        direct_fold8_mode,
    );
    let gl2_s0 = match sparse_dual_depth {
        Some(depth) if gl2_defer && gl2_stream_enabled() && lig_sparse_dual_snapshot_enabled() => {
            Some(Gl2DualStream::start(
                depth,
                n1 + log_inv_rate_0,
                l0_codeword,
                l0_num_interleaved,
                &r_lane_fold,
                num_queries_0,
            ))
        }
        _ => None,
    };
    let (queries_0, gl2_perm0) = match &gl2_s0 {
        Some(st) => sample_distinct_queries_streamed(challenger, l0_block_len, num_queries_0, st),
        None => (sample_distinct_queries(challenger, l0_block_len, num_queries_0), Vec::new()),
    };
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    crate::rstrace::evd("L1:qs", queries_0.len() as u32);
    let open_l0 = || {
        let opened_rows = gather_opened_rows(&queries_0, l0_row, serial_par_enabled());
        crate::rstrace::evd("L1:rows", 0);
        let merkle_proof = merkle_multi_proof_for_l0(
            l0_tree,
            l0_codeword,
            l0_block_len,
            l0_num_interleaved,
            &queries_0,
            config.merkle_hash,
        );
        crate::rstrace::evd("L1:mp", 0);
        (opened_rows, merkle_proof)
    };

    let sks_vks_n1 = if induce_lazy_sks_enabled() {
        None
    } else {
        Some(eval_sk_at_vks(n1))
    };
    let _t = std::time::Instant::now();
    let ((opened_rows_0, merkle_proof_0), (basis_0_induced, sparse_dual_0, enforced_sum_0)) =
        if let (Some(depth), true) = (sparse_dual_depth, gl2_defer) {
            let (tree_p, cw_p) = (Gl2Ptr::of(l0_tree), Gl2Ptr::of(l0_codeword));
            let (bl, ni, kind) = (l0_block_len, l0_num_interleaved, config.merkle_hash);
            let qs = queries_0.clone();
            let job = Gl2Deferred::spawn(Box::new(move || {
                // SAFETY: the L0 tree / codeword are borrowed for the whole
                // open; `gl2_jobs` waits for this job before it returns.
                let (tree, cw) = unsafe { (tree_p.get(), cw_p.get()) };
                let rows = gather_opened_rows(&qs, |q| &cw[q * ni..(q + 1) * ni], serial_par_enabled());
                let mp = merkle_multi_proof_for_l0(tree, cw, bl, ni, &qs, kind);
                crate::rstrace::evd("L1:deferred", 0);
                (rows, mp)
            }));
            gl2_jobs.0.push(job.clone());
            gl2_l1_job = Some(job);
            crate::rstrace::evd("L1:dual0", 0);
            // Lane sl1: warm holders for L1's three fold grinds (~0.05 ms ahead).
            crate::challenger::sl1_grind_prewarm(sl1_hold_us(120));
            let (dual, enforced_sum) = if let Some(st) = gl2_s0.as_deref() {
                let r = SparseDualL0::from_stream(st, log_inv_rate_0, queries_0.clone(), &gl2_perm0, &alpha_0);
                gl2_side_done.extend(st.take_side());
                r
            } else {
                SparseDualL0::new(
                    depth,
                    n1 + log_inv_rate_0,
                    log_inv_rate_0,
                    l0_codeword,
                    l0_num_interleaved,
                    &r_lane_fold,
                    &queries_0,
                    &alpha_0,
                )
            };
            crate::rstrace::evd("L1:dual1", 0);
            ((Vec::new(), Vec::new()), (None, Some(dual), enforced_sum))
        } else if let Some(depth) = sparse_dual_depth {
            let ((dual, enforced_sum), opened) = rayon::join(
                || {
                    crate::rstrace::evd("L1:dual0", 0);
                    let r = SparseDualL0::new(
                        depth,
                        n1 + log_inv_rate_0,
                        log_inv_rate_0,
                        l0_codeword,
                        l0_num_interleaved,
                        &r_lane_fold,
                        &queries_0,
                        &alpha_0,
                    );
                    crate::rstrace::evd("L1:dual1", 0);
                    r
                },
                open_l0,
            );
            (opened, (None, Some(dual), enforced_sum))
        } else {
            let opened = open_l0();
            if trace {
                t_opens += _t.elapsed();
            }
            let (basis, enforced_sum) = induce_sumcheck_poly_auto(
                n1,
                log_inv_rate_0,
                sks_vks_n1.as_deref(),
                &opened.0,
                &r_lane_fold,
                &queries_0,
                &alpha_0,
            );
            (opened, (Some(basis), None, enforced_sum))
        };
    if trace {
        let d = _t.elapsed();
        t_induce += d;
        let lb = n1 + log_inv_rate_0;
        induce_diag.push((
            0,
            n1,
            queries_0.len(),
            d.as_secs_f64() * 1e3,
            n1 >= 12 && queries_0.len() > 4 * (1usize << log_inv_rate_0) * lb.max(1),
        ));
    }

    crate::rstrace::ev("L1:open+induce", 0);
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0,
        merkle_proof: merkle_proof_0,
    };

    let _t = std::time::Instant::now();
    let intro_msg_0 = if let Some(dual) = sparse_dual_0.as_ref() {
        let msg = dual.round_msg(&[]);
        sc_prover.introduce_sparse_dual(msg);
        msg
    } else {
        sc_prover.introduce_new(
            basis_0_induced.expect("ordinary L0 induce basis missing"),
            enforced_sum_0,
        )
    };
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    let mut active_sparse_dual_0 = if let Some(mut dual) = sparse_dual_0 {
        sc_prover.glue_sparse_dual(beta_0, enforced_sum_0);
        dual.scale(beta_0);
        let depth = dual.depth;
        Some((dual, Vec::<F128>::with_capacity(depth)))
    } else if ranked_deferred_induced_glue_enabled(
        config,
        log_n,
        n1,
        sc_prover.f().len(),
        num_queries_0,
        log_inv_rate_0,
        lazy_l1_ood,
        direct_fold8_mode,
    ) {
        sc_prover.glue_deferred_into_factorized_ood_fold(beta_0);
        None
    } else {
        sc_prover.glue(beta_0);
        None
    };
    if trace {
        t_intro_glue += _t.elapsed();
    }

    crate::rstrace::ev("L1:intro+glue", 0);
    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();
    let mut active_deep_duals: Vec<(SparseDualL0, Vec<F128>)> = Vec::with_capacity(2);

    let mut gl2_tail: Option<crate::hot_pool::side::Task> = None;
    for i in 0..r {
        if i + 2 == r || r == 1 {
            // Lane gl2: the publisher's tail hook, run by an idle spinner
            // during the last two (small) levels so it is done well before
            // the publish (see `GL2_TAIL_HOOK`).
            gl2_tail = crate::GL2_TAIL_HOOK.get().copied().map(|h| {
                let once = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let t: crate::hot_pool::side::Task = std::sync::Arc::new(move |_sib: bool| {
                    if !once.swap(true, std::sync::atomic::Ordering::AcqRel) {
                        h();
                        crate::rstrace::evd("fin:hook", 0);
                    }
                });
                crate::hot_pool::side::register(t.clone());
                t
            });
        }
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let _t = std::time::Instant::now();
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            crate::rstrace::ev("lv:fgrind", (i * 16 + j) as u32);
            let ri = challenger.sample_f128();
            // Lane opus-pcs: the sparse duals' round steps and materialization
            // read only their own state and the fold challenges, never the
            // sumcheck prover, so they run beside `sc_prover.fold(ri)` (a
            // bandwidth-bound pass) instead of after it. Their deltas and
            // bases are then applied in the incumbent order: identical
            // messages (XOR-adds into the same last message), identical
            // defer/merge sequence, identical transcript.
            let has_dual_work = active_sparse_dual_0.is_some() || !active_deep_duals.is_empty();
            // Join only where it can pay: the fold is a parallel pass (input
            // >= 8192, i.e. half >= the fold's PAR_THRESHOLD) or a dual
            // materializes this round. A tiny serial fold beside a few
            // query-sized round steps is cheaper inline than a join.
            let join_pays = sc_prover.f().len() >= (1 << 13)
                || sparse_duals_materialize_next(&active_sparse_dual_0, &active_deep_duals);
            let (mut msg, dual_work) = if has_dual_work && join_pays && opus_dual_join_enabled() {
                let (l0, deep, scp) =
                    (&mut active_sparse_dual_0, &mut active_deep_duals, &mut sc_prover);
                let (work, msg) =
                    rayon::join(|| sparse_duals_fold_step(l0, deep, ri), || scp.fold(ri));
                (msg, work)
            } else {
                let msg = sc_prover.fold(ri);
                (msg, sparse_duals_fold_step(&mut active_sparse_dual_0, &mut active_deep_duals, ri))
            };
            if let Some(delta) = dual_work.l0_delta {
                msg = sc_prover.add_to_last_message(delta);
            }
            if let Some(basis) = dual_work.l0_basis {
                if j + 1 < k_i {
                    sc_prover.defer_folded_sparse_dual(basis, F128::ONE);
                } else {
                    sc_prover.merge_folded_sparse_dual(basis);
                }
                active_sparse_dual_0 = None;
            }
            if !active_deep_duals.is_empty() {
                if let Some(delta) = dual_work.deep_delta {
                    msg = sc_prover.add_to_last_message(delta);
                }
                if let Some(basis) = dual_work.deep_basis {
                    if j + 1 < k_i {
                        sc_prover.defer_folded_sparse_dual(basis, F128::ONE);
                    } else {
                        sc_prover.merge_folded_sparse_dual(basis);
                    }
                }
                active_deep_duals
                    .retain(|(dual, fold_challenges)| fold_challenges.len() < dual.depth);
            }
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            crate::rstrace::ev("lv:fold", (i * 16 + j) as u32);
            level_rs.push(ri);
        }
        if trace {
            t_sumcheck_folds += _t.elapsed();
        }

        if i == r - 1 {
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            crate::rstrace::ev("fin:yr", 0);
            let nonce_last = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
            crate::rstrace::ev("fin:qgrind", 0);
            grinding_nonces.push(nonce_last);
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let _t = std::time::Instant::now();
            crate::rstrace::evd("fin:qs", 0);
            // Lane gl2: rows and multiproof of the final level side by side
            // (`FLOCK_NO_GL2_FIN_JOIN=1`: one after the other, incumbent).
            static FIN_JOIN: std::sync::LazyLock<bool> =
                std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GL2_FIN_JOIN").is_none());
            let (opened_rows_last, merkle_proof_last): (Vec<Vec<F128>>, Vec<Hash>) = if *FIN_JOIN {
                rayon::join(
                    || gather_opened_rows(&queries_last, |q| wtns_prev.row(q), false),
                    || multi_proof_gather(&wtns_prev.tree, wtns_prev.block_len, &queries_last, false),
                )
            } else {
                let rows = gather_opened_rows(&queries_last, |q| wtns_prev.row(q), lf1_open_par());
                crate::rstrace::evd("fin:rows", 0);
                let mp = multi_proof_gather(&wtns_prev.tree, wtns_prev.block_len, &queries_last, lf1_open_par());
                (rows, mp)
            };
            crate::rstrace::ev("fin:open", 0);
            let mut initial_proof = initial_proof;
            if let Some(job) = gl2_l1_job.take() {
                gl2_side_done.extend(job.take_side());
                let (rows, mp) = job.wait().expect("deferred L1 open gather panicked");
                initial_proof.opened_rows = rows;
                initial_proof.merkle_proof = mp;
            }
            for (k, job) in gl2_level_jobs.drain(..) {
                gl2_side_done.extend(job.take_side());
                let (rows, mp) = job.wait().expect("deferred level open gather panicked");
                recursive_proofs[k].opened_rows = rows;
                recursive_proofs[k].merkle_proof = mp;
            }
            gl2_side_done.extend(gl2_tail.take());
            crate::hot_pool::side::unregister_many(&gl2_side_done);
            crate::rstrace::ev("fin:joined", 0);
            if trace {
                t_opens += _t.elapsed();
            }
            if trace {
                let total = t_total.elapsed();
                eprintln!("[lig-prove] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  initial sumcheck (initial_k folds + SC build): {:.2} ms",
                    t_init_sumcheck.as_secs_f64() * 1e3
                );
                for (j, (g, f, a)) in round_diag.iter().enumerate() {
                    eprintln!("    round {j}: grind {g:.2} ms, fold {f:.2} ms, arena={a}",);
                }
                eprintln!(
                    "  recursive commits (NTT + merkle):              {:.2} ms",
                    t_commits.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  opens (rows + multi-proof):                    {:.2} ms",
                    t_opens.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  induce_sumcheck_poly:                          {:.2} ms",
                    t_induce.as_secs_f64() * 1e3
                );
                for (lvl, lmc, nq, ms, ntt) in induce_diag.iter() {
                    eprintln!(
                        "    induce L{lvl}: log_msg_cols={lmc} queries={nq} ntt={ntt} {ms:.2} ms",
                    );
                }
                eprintln!(
                    "  sumcheck recursive folds:                      {:.2} ms",
                    t_sumcheck_folds.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  introduce_new + glue:                          {:.2} ms",
                    t_intro_glue.as_secs_f64() * 1e3
                );
                if !ood_values.is_empty() {
                    eprintln!(
                        "  OOD samples ({}): MLE evals + glue:            {:.2} ms",
                        ood_values.len(),
                        t_ood.as_secs_f64() * 1e3
                    );
                }
            }
            let proof = LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: std::mem::take(&mut sc_prover.transcript),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces,
            };
            // Lane sl2 (`FLOCK_NO_SL2_LEAK`): the timed prove's process exits
            // right after publishing; skip freeing (munmap + TLB shootdowns)
            // the retired level witnesses and the last fold state here.
            if crate::sl2_leak_tail() {
                std::mem::forget(std::mem::take(&mut gl2_retired));
                std::mem::forget(wtns_prev);
                std::mem::forget(sc_prover);
            }
            return proof;
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(n_next >= log_num_interleaved_next);
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let _t = std::time::Instant::now();
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let wtns_next = ligero_commit(
            sc_prover.f(),
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
            config.merkle_hash,
        );
        if trace {
            t_commits += _t.elapsed();
        }
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        crate::rstrace::ev("lv:root", i as u32);
        recursive_roots.push(root_next);

        let lazy_deep_ood = ranked_deep_lazy_ood_eq_enabled(
            config,
            log_n,
            n_next,
            ood_count(i + 2),
            sc_prover.f().len(),
            direct_fold4_mode || direct_fold8_mode,
        );

        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                crate::rstrace::evd("lv:oodz", i as u32);
                let (intro, y) = if lazy_deep_ood {
                    sc_prover
                        .introduce_new_ood_factorized(&z)
                        .expect("ranked factorized deep OOD introduction shape changed")
                } else {
                    let eq_z = build_eq_table_split(&z);
                    sc_prover.introduce_new_with_eval(eq_z)
                };
                crate::rstrace::evd("lv:oodi", i as u32);
                challenger.observe_f128(y);
                ood_values.push(y);
                challenger.observe_f128(intro.u_0);
                challenger.observe_f128(intro.u_2);
                let beta = challenger.sample_f128();
                if lazy_deep_ood {
                    sc_prover.glue_factorized_ood(beta);
                } else {
                    sc_prover.glue(beta);
                }
            }
            if trace {
                t_ood += _t.elapsed();
            }
        }

        crate::rstrace::ev("lv:ood", i as u32);
        let nonce_i = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
        crate::rstrace::ev("lv:qgrind", i as u32);
        grinding_nonces.push(nonce_i);
        let num_queries_i = config.queries[i + 1];
        let remaining_folds: usize = config.recursive_ks[i + 1..].iter().sum();
        let deep_dual_depth = ranked_sparse_dual_deep_depth(
            config,
            log_n,
            n_next,
            sc_prover.f().len(),
            num_queries_i,
            config.log_inv_rates[i + 1],
            lazy_deep_ood,
            direct_fold8_mode,
            remaining_folds,
        );
        let gl2_si = match deep_dual_depth {
            Some(depth) if gl2_defer && gl2_stream_enabled() && lig_sparse_dual_snapshot_enabled() => {
                Some(Gl2DualStream::start(
                    depth,
                    n_next + config.log_inv_rates[i + 1],
                    &wtns_prev.mat,
                    wtns_prev.num_interleaved,
                    &level_rs,
                    num_queries_i,
                ))
            }
            _ => None,
        };
        let (queries_i, gl2_permi) = match &gl2_si {
            Some(st) => sample_distinct_queries_streamed(challenger, wtns_prev.block_len, num_queries_i, st),
            None => (sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i), Vec::new()),
        };
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        crate::rstrace::evd("lv:qs", i as u32);
        let open_i = || {
            let opened_rows =
                gather_opened_rows(&queries_i, |q| wtns_prev.row(q), lf1_open_par());
            crate::rstrace::evd("lv:rows", i as u32);
            let merkle_proof =
                multi_proof_gather(&wtns_prev.tree, wtns_prev.block_len, &queries_i, lf1_open_par());
            crate::rstrace::evd("lv:mp", i as u32);
            (opened_rows, merkle_proof)
        };

        let sks_vks_i = if induce_sched_enabled() && induce_lazy_sks_enabled() {
            None
        } else {
            Some(eval_sk_at_vks(n_next))
        };
        let _t = std::time::Instant::now();
        let ((opened_rows_i, merkle_proof_i), (basis_i_induced, deep_dual_i, enforced_sum_i)) =
            if let (Some(depth), true) = (deep_dual_depth, gl2_defer) {
                let (mat_p, tree_p) = (Gl2Ptr::of(&wtns_prev.mat), Gl2Ptr::of(&wtns_prev.tree));
                let (bl, ni) = (wtns_prev.block_len, wtns_prev.num_interleaved);
                let qs = queries_i.clone();
                let lvl = i as u32;
                let job = Gl2Deferred::spawn(Box::new(move || {
                    // SAFETY: this level's witness moves into `gl2_retired`
                    // (heap buffers unchanged) and outlives `gl2_jobs`.
                    let (mat, tree) = unsafe { (mat_p.get(), tree_p.get()) };
                    let rows = gather_opened_rows(&qs, |q| &mat[q * ni..(q + 1) * ni], false);
                    crate::rstrace::evd("lv:rows", lvl);
                    let mp = multi_proof_gather(tree, bl, &qs, false);
                    crate::rstrace::evd("lv:mp", lvl);
                    (rows, mp)
                }));
                gl2_jobs.0.push(job.clone());
                gl2_level_jobs.push((recursive_proofs.len(), job));
                crate::rstrace::evd("lv:dual0", i as u32);
                let (dual, enforced_sum) = if let Some(st) = gl2_si.as_deref() {
                    let r = SparseDualL0::from_stream(
                        st,
                        config.log_inv_rates[i + 1],
                        queries_i.clone(),
                        &gl2_permi,
                        &alpha_i,
                    );
                    gl2_side_done.extend(st.take_side());
                    r
                } else {
                    SparseDualL0::new(
                        depth,
                        n_next + config.log_inv_rates[i + 1],
                        config.log_inv_rates[i + 1],
                        &wtns_prev.mat,
                        wtns_prev.num_interleaved,
                        &level_rs,
                        &queries_i,
                        &alpha_i,
                    )
                };
                crate::rstrace::evd("lv:dual1", i as u32);
                ((Vec::new(), Vec::new()), (None, Some(dual), enforced_sum))
            } else if let Some(depth) = deep_dual_depth {
                let ((dual, enforced_sum), opened) = rayon::join(
                    || {
                        crate::rstrace::evd("lv:dual0", i as u32);
                        let r = SparseDualL0::new(
                            depth,
                            n_next + config.log_inv_rates[i + 1],
                            config.log_inv_rates[i + 1],
                            &wtns_prev.mat,
                            wtns_prev.num_interleaved,
                            &level_rs,
                            &queries_i,
                            &alpha_i,
                        );
                        crate::rstrace::evd("lv:dual1", i as u32);
                        r
                    },
                    open_i,
                );
                (opened, (None, Some(dual), enforced_sum))
            } else if induce_sched_enabled() {
                let opened = open_i();
                let (basis, enforced_sum) = induce_sumcheck_poly_auto(
                    n_next,
                    config.log_inv_rates[i + 1],
                    sks_vks_i.as_deref(),
                    &opened.0,
                    &level_rs,
                    &queries_i,
                    &alpha_i,
                );
                (opened, (Some(basis), None, enforced_sum))
            } else {
                let opened = open_i();
                let (basis, enforced_sum) = induce_sumcheck_poly(
                    n_next,
                    sks_vks_i
                        .as_deref()
                        .expect("sched-disabled induce always precomputes sks_vks"),
                    &opened.0,
                    &level_rs,
                    &queries_i,
                    &alpha_i,
                );
                (opened, (Some(basis), None, enforced_sum))
            };
        if trace {
            let d = _t.elapsed();
            t_induce += d;
            let rate = config.log_inv_rates[i + 1];
            induce_diag.push((
                i + 1,
                n_next,
                queries_i.len(),
                d.as_secs_f64() * 1e3,
                induce_sched_enabled()
                    && n_next >= 12
                    && queries_i.len()
                        > induce_ntt_crossover_c() * (1usize << rate) * (n_next + rate).max(1),
            ));
        }

        crate::rstrace::ev("lv:open+induce", i as u32);
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i,
            merkle_proof: merkle_proof_i,
        });

        let _t = std::time::Instant::now();
        let intro_msg_i = if let Some(dual) = deep_dual_i.as_ref() {
            let msg = dual.round_msg(&[]);
            sc_prover.introduce_sparse_dual(msg);
            msg
        } else {
            sc_prover.introduce_new(
                basis_i_induced.expect("ordinary deep induce basis missing"),
                enforced_sum_i,
            )
        };
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        if let Some(mut dual) = deep_dual_i {
            sc_prover.glue_sparse_dual(beta_i, enforced_sum_i);
            dual.scale(beta_i);
            let depth = dual.depth;
            active_deep_duals.push((dual, Vec::<F128>::with_capacity(depth)));
        } else if ranked_deferred_induced_glue_enabled(
            config,
            log_n,
            n_next,
            sc_prover.f().len(),
            num_queries_i,
            config.log_inv_rates[i + 1],
            lazy_deep_ood,
            direct_fold8_mode,
        ) {
            sc_prover.glue_deferred_into_factorized_ood_fold(beta_i);
        } else {
            sc_prover.glue(beta_i);
        }
        if trace {
            t_intro_glue += _t.elapsed();
        }

        crate::rstrace::ev("lv:intro+glue", i as u32);
        let retired = std::mem::replace(&mut wtns_prev, wtns_next);
        if gl2_defer {
            gl2_retired.push(retired);
        } else {
            drop(retired);
        }
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis_succinct<Ch, F>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    log_n: usize,
    target: F128,
    expected_initial_root: &Hash,
    eval_b_residual: F,
    challenger: &mut Ch,
) -> bool
where
    Ch: Challenger,
    F: Fn(&[F128], usize) -> Vec<F128>,
{
    let trace = std::env::var("LIG_VERIFY_TRACE").is_ok();
    let mut t_merkle = std::time::Duration::ZERO;
    let mut t_sample_q = std::time::Duration::ZERO;
    let mut t_enforced = std::time::Duration::ZERO;
    let mut t_residual = std::time::Duration::ZERO;
    let mut t_evalb = std::time::Duration::ZERO;
    let t_start = std::time::Instant::now();

    let initial_k = config.initial_k;
    let r = config.recursive_steps;
    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    struct OodCtx {
        z: Vec<F128>,
        ris_start: usize,
        beta: F128,
    }
    let mut ood_ctxs: Vec<OodCtx> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_ctxs.push(OodCtx {
            z,
            ris_start: initial_k,
            beta,
        });
    }

    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let _t = std::time::Instant::now();
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    if trace {
        t_sample_q += _t.elapsed();
    }
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }
    if trace {
        t_merkle += _t.elapsed();
    }

    let n1 = log_n - initial_k;
    let _t = std::time::Instant::now();
    let enforced_sum_0 = induce_sumcheck_enforced_sum(
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_enforced += _t.elapsed();
    }

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    struct LevelCtx {
        log_msg_cols: usize,
        queries: Vec<usize>,
        alpha: Vec<F128>, // ⌈log₂ Q⌉ field elements (eq-tensor combination)
        ris_start: usize,
        beta: F128,
    }
    let mut level_ctxs: Vec<LevelCtx> = vec![LevelCtx {
        log_msg_cols: n1,
        queries: queries_0.clone(),
        alpha: alpha_0,
        ris_start: initial_k,
        beta: beta_0,
    }];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let _t = std::time::Instant::now();
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if trace {
                t_sample_q += _t.elapsed();
            }
            let _t = std::time::Instant::now();
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }
            if trace {
                t_merkle += _t.elapsed();
            }

            let enforced_sum_last = induce_sumcheck_enforced_sum(
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            level_ctxs.push(LevelCtx {
                log_msg_cols: n_current,
                queries: queries_last.clone(),
                alpha: alpha_last,
                ris_start: ris.len(),
                beta: beta_last,
            });

            let yr_len = yr.len();
            let yr_log_n = n_current;

            let _t = std::time::Instant::now();
            let induced_residuals: Vec<Vec<F128>> = level_ctxs
                .iter()
                .map(|ctx| {
                    let sks_vks = eval_sk_at_vks(ctx.log_msg_cols);
                    let ris_for_basis =
                        &ris[ctx.ris_start..ctx.ris_start + ctx.log_msg_cols - yr_log_n];
                    induce_sumcheck_evaluate_at_residual(
                        ctx.log_msg_cols,
                        &sks_vks,
                        &ctx.queries,
                        &ctx.alpha,
                        ris_for_basis,
                        yr_log_n,
                    )
                })
                .collect();
            if trace {
                t_residual += _t.elapsed();
            }
            for resid in &induced_residuals {
                if resid.len() != yr_len {
                    return false;
                }
            }

            let mut ood_residuals: Vec<Vec<F128>> = Vec::with_capacity(ood_ctxs.len());
            for ctx in &ood_ctxs {
                if ctx.z.len() < yr_log_n || ctx.ris_start + (ctx.z.len() - yr_log_n) > ris.len() {
                    return false;
                }
                let folded = ctx.z.len() - yr_log_n;
                let mut scalar = ctx.beta;
                for b in 0..folded {
                    scalar *= F128::ONE + ctx.z[b] + ris[ctx.ris_start + b];
                }
                let mut tail = build_eq_table(&ctx.z[folded..]);
                for v in tail.iter_mut() {
                    *v *= scalar;
                }
                ood_residuals.push(tail);
            }

            let _te = std::time::Instant::now();
            let evb_vec = eval_b_residual(&ris, yr_log_n);
            if trace {
                t_evalb += _te.elapsed();
            }
            if evb_vec.len() != yr_len {
                return false;
            }
            let mut inner = F128::ZERO;
            let _t = std::time::Instant::now();
            for y in 0..yr_len {
                let mut combined_y = evb_vec[y];
                for (k, residual) in induced_residuals.iter().enumerate() {
                    combined_y += level_ctxs[k].beta * residual[y];
                }
                for resid in &ood_residuals {
                    combined_y += resid[y];
                }
                inner += yr[y] * combined_y;
            }
            if trace {
                t_residual += _t.elapsed();
            }
            if trace {
                let total = t_start.elapsed();
                eprintln!("[lig-verify] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  merkle multi-proofs:       {:.2} ms",
                    t_merkle.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sample_distinct_queries:   {:.2} ms",
                    t_sample_q.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  enforced_sum (eq+dot):     {:.2} ms",
                    t_enforced.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  residual basis eval:       {:.2} ms",
                    t_residual.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  eval_b (yr_len positions): {:.2} ms",
                    t_evalb.as_secs_f64() * 1e3
                );
            }
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_ctxs.push(OodCtx {
                z,
                ris_start: ris.len(),
                beta,
            });
        }

        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let _t = std::time::Instant::now();
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        if trace {
            t_sample_q += _t.elapsed();
        }
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        let _t = std::time::Instant::now();
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }
        if trace {
            t_merkle += _t.elapsed();
        }

        let _t = std::time::Instant::now();
        let enforced_sum_i =
            induce_sumcheck_enforced_sum(&rp.opened_rows, &level_rs, &queries_i, &alpha_i);
        if trace {
            t_enforced += _t.elapsed();
        }

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        level_ctxs.push(LevelCtx {
            log_msg_cols: n_current,
            queries: queries_i.clone(),
            alpha: alpha_i,
            ris_start: ris.len(),
            beta: beta_i,
        });

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
pub fn recursive_verifier_with_basis<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    b_initial: &[F128],
    target: F128,
    expected_initial_root: &Hash,
    challenger: &mut Ch,
) -> bool {
    let log_n = b_initial.len().trailing_zeros() as usize;
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if b_initial.len() != 1usize << log_n {
        return false;
    }
    if &proof.initial_root != expected_initial_root {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);
    challenger.observe_bytes(&proof.initial_root);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;

    let mut t_r = target;
    let mut tx_idx = 0usize;
    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };
    if config.ood_samples.first().copied().unwrap_or(0) != 0 {
        return false; // L0 must be bound by the opening's own eval claim
    }
    let mut fold_nonce_idx = 0usize;
    let mut ood_idx = 0usize;
    let mut ood_bases: Vec<(Vec<F128>, usize, F128)> = Vec::new();

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                return false;
            }
            fold_nonce_idx += 1;
        }
        let ri = challenger.sample_f128();
        r_lane_fold.push(ri);
        t_r = running_quad.eval(ri);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        running_quad = RoundQuad::from_msg(msg, t_r);
    }

    if proof.recursive_roots.is_empty() {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    for _ in 0..ood_count(1) {
        let z = challenger.sample_f128_vec(log_n - initial_k);
        if ood_idx >= proof.ood_values.len() {
            return false;
        }
        let y = proof.ood_values[ood_idx];
        ood_idx += 1;
        challenger.observe_f128(y);
        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg.u_0);
        challenger.observe_f128(intro_msg.u_2);
        let intro_quad = RoundQuad::from_msg(intro_msg, y);
        let beta = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
        t_r += beta * y;
        ood_bases.push((build_eq_table(&z), initial_k, beta));
    }

    let mut nonce_idx = 0usize;
    if nonce_idx >= proof.grinding_nonces.len() {
        return false;
    }
    if !challenger.verify_pow(
        proof.grinding_nonces[nonce_idx],
        config.grinding_bits[0] as u32,
    ) {
        return false;
    }
    nonce_idx += 1;

    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }

    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        Some(&sks_vks_n1),
        &proof.initial_proof.opened_rows,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;

    let mut basis_polys: Vec<Vec<F128>> = vec![b_initial.to_vec(), basis_0_induced];
    let mut basis_ris_starts: Vec<usize> = vec![0, initial_k];
    let mut basis_separations: Vec<F128> = vec![beta_0];
    let mut ris: Vec<F128> = r_lane_fold.clone();

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1];
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for j in 0..k_i {
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                if fold_nonce_idx >= proof.fold_grinding_nonces.len() {
                    return false;
                }
                if !challenger.verify_pow(proof.fold_grinding_nonces[fold_nonce_idx], bits) {
                    return false;
                }
                fold_nonce_idx += 1;
            }
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            if ood_idx != proof.ood_values.len()
                || fold_nonce_idx != proof.fold_grinding_nonces.len()
            {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            if nonce_idx >= proof.grinding_nonces.len() {
                return false;
            }
            if !challenger.verify_pow(
                proof.grinding_nonces[nonce_idx],
                config.grinding_bits[i + 1] as u32,
            ) {
                return false;
            }

            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }

            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * rr;
                }
            }
            for (basis, start, beta) in &ood_bases {
                let residual = partial_eval_lsb(basis, &ris[*start..]);
                if residual.len() != yr_len {
                    return false;
                }
                for (c, &rr) in combined.iter_mut().zip(residual.iter()) {
                    *c += *beta * rr;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        for _ in 0..ood_count(i + 2) {
            let z = challenger.sample_f128_vec(n_current);
            if ood_idx >= proof.ood_values.len() {
                return false;
            }
            let y = proof.ood_values[ood_idx];
            ood_idx += 1;
            challenger.observe_f128(y);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let intro_msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(intro_msg.u_0);
            challenger.observe_f128(intro_msg.u_2);
            let intro_quad = RoundQuad::from_msg(intro_msg, y);
            let beta = challenger.sample_f128();
            running_quad = RoundQuad::fold(&running_quad, &intro_quad, beta);
            t_r += beta * y;
            ood_bases.push((build_eq_table(&z), ris.len(), beta));
        }

        if nonce_idx >= proof.grinding_nonces.len() {
            return false;
        }
        if !challenger.verify_pow(
            proof.grinding_nonces[nonce_idx],
            config.grinding_bits[i + 1] as u32,
        ) {
            return false;
        }
        nonce_idx += 1;

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!()
}

#[allow(clippy::too_many_arguments)]
fn recursive_prover_inner<Ch: Challenger>(
    config: &ProverConfig,
    poly: &[F128],
    wtns_0: LigeroWitness,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
    t_total: std::time::Instant,
    mut t_commits: std::time::Duration,
    mut t_induce: std::time::Duration,
    mut t_sumcheck: std::time::Duration,
    mut t_opens: std::time::Duration,
    trace: bool,
) -> LigeritoProof {
    macro_rules! tlog {
        ($($arg:tt)*) => { if trace { eprintln!($($arg)*); } }
    }
    assert!(
        config.ood_samples.iter().all(|&s| s == 0)
            && config.fold_grinding_bits.iter().all(|&b| b == 0),
        "OOD samples / fold grinding require the with_basis prover path"
    );
    let log_n = poly.len().trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;
    let log_inv_rate_0 = config.log_inv_rates[0];

    let initial_root = wtns_0.root();
    challenger.observe_bytes(&initial_root);

    let v_challenges_0 = eval_point[..initial_k].to_vec();
    let f1 = partial_eval_lsb(poly, &v_challenges_0);
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1, "n1 < k_0");
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let t = std::time::Instant::now();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
        config.merkle_hash,
    );
    let t_l1 = t.elapsed();
    t_commits += t_l1;
    tlog!("  [ligerito]   L1 commit: {:.2?}", t_l1);
    challenger.observe_bytes(&wtns_1.root());

    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, wtns_0.block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| wtns_0.row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(&wtns_0.tree, wtns_0.block_len, &queries_0);
    t_opens += t.elapsed();
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    let sks_vks_n1 = eval_sk_at_vks(n1);
    let t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        Some(&sks_vks_n1),
        &opened_rows_0,
        &v_challenges_0,
        &queries_0,
        &alpha_0,
    );
    t_induce += t.elapsed();

    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    let t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = SumcheckProver::new(f1, eq_z_residual, claimed_value);
    t_sumcheck += t.elapsed();
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);

    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let t = std::time::Instant::now();
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        t_sumcheck += t.elapsed();

        if i == r - 1 {
            tlog!(
                "  [ligerito] commits: {:.2?}  induce: {:.2?}  sumcheck: {:.2?}  opens: {:.2?}  TOTAL: {:.2?}",
                t_commits,
                t_induce,
                t_sumcheck,
                t_opens,
                t_total.elapsed()
            );
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            let num_queries_last = udr_queries(config.log_inv_rates[i + 1]);
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            return LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: std::mem::take(&mut sc_prover.transcript),
                grinding_nonces: Vec::new(), // legacy recursive_prover_inner: no grinding plumbed
                ood_values: Vec::new(),
                fold_grinding_nonces: Vec::new(),
            };
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(
            n_next >= log_num_interleaved_next,
            "f.n ({n_next}) < k_{} ({log_num_interleaved_next})",
            i + 1
        );
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let t = std::time::Instant::now();
        let wtns_next = ligero_commit(
            sc_prover.f(),
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
            config.merkle_hash,
        );
        let t_li = t.elapsed();
        t_commits += t_li;
        tlog!("  [ligerito]   L{} commit: {:.2?}", i + 2, t_li);
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        let num_queries_i = udr_queries(config.log_inv_rates[i + 1]);
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        t_opens += t.elapsed();
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i.clone(),
            merkle_proof: merkle_proof_i,
        });

        let sks_vks_i = eval_sk_at_vks(n_next);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);

        wtns_prev = wtns_next;
    }

    unreachable!("recursive loop should return on last iter")
}

fn verify_level_opens(
    root: &Hash,
    block_len: usize,
    queries: &[usize],
    opened_rows: &[Vec<F128>],
    expected_num_interleaved: usize,
    multi_proof: &[Hash],
    kind: HashKind,
) -> bool {
    if queries.len() != opened_rows.len() {
        return false;
    }
    let mut leaf_hashes: Vec<Hash> = Vec::with_capacity(opened_rows.len());
    for row in opened_rows {
        if row.len() != expected_num_interleaved {
            return false;
        }
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                row.as_ptr() as *const u8,
                row.len() * core::mem::size_of::<F128>(),
            )
        };
        leaf_hashes.push(merkle::hash_leaf(bytes, kind));
    }
    merkle::verify_merkle_multi_proof(root, block_len, queries, &leaf_hashes, multi_proof, kind)
}

pub fn recursive_verifier<Ch: Challenger>(
    config: &VerifierConfig,
    proof: &LigeritoProof,
    eval_point: &[F128],
    claimed_value: F128,
    challenger: &mut Ch,
) -> bool {
    let log_n = eval_point.len();
    let initial_k = config.initial_k;
    let r = config.recursive_steps;

    if r < 1 || config.recursive_ks.len() != r || config.log_inv_rates.len() != r + 1 {
        return false;
    }
    if config.ood_samples.iter().any(|&s| s != 0)
        || config.fold_grinding_bits.iter().any(|&b| b != 0)
    {
        return false;
    }

    challenger.observe_label(b"flock-ligerito-v0");
    challenger.observe_f128(claimed_value);
    challenger.observe_f128_slice(eval_point);

    challenger.observe_bytes(&proof.initial_root);
    if proof.recursive_roots.len() != r {
        return false;
    }
    let root_1 = proof.recursive_roots[0];
    challenger.observe_bytes(&root_1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    let num_queries_0 = udr_queries(log_inv_rate_0);
    let queries_0 = sample_distinct_queries(challenger, block_len_0, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));

    if !verify_level_opens(
        &proof.initial_root,
        block_len_0,
        &queries_0,
        &proof.initial_proof.opened_rows,
        num_interleaved_0,
        &proof.initial_proof.merkle_proof,
        config.merkle_hash,
    ) {
        return false;
    }

    let n1 = log_n - initial_k;
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        Some(&sks_vks_n1),
        &proof.initial_proof.opened_rows,
        &eval_point[..initial_k],
        &queries_0,
        &alpha_0,
    );

    let eq_z_residual = build_eq_table(&eval_point[initial_k..]);
    let mut basis_polys: Vec<Vec<F128>> = vec![eq_z_residual];
    let mut basis_ris_starts: Vec<usize> = vec![0];
    let mut basis_separations: Vec<F128> = Vec::new(); // separation for basis_polys[k+1]
    let mut ris: Vec<F128> = Vec::new();
    let mut t_r = claimed_value;
    let mut tx_idx = 0usize;

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let start_msg = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);
    let mut running_quad = RoundQuad::from_msg(start_msg, t_r);

    if tx_idx >= proof.sumcheck_transcript.len() {
        return false;
    }
    let intro_msg_0 = proof.sumcheck_transcript[tx_idx];
    tx_idx += 1;
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let intro_quad_0 = RoundQuad::from_msg(intro_msg_0, enforced_sum_0);
    let beta_0 = challenger.sample_f128();
    running_quad = RoundQuad::fold(&running_quad, &intro_quad_0, beta_0);
    t_r += beta_0 * enforced_sum_0;
    basis_polys.push(basis_0_induced);
    basis_ris_starts.push(0);
    basis_separations.push(beta_0);

    let mut prev_root = root_1;
    let mut prev_log_num_interleaved = config.recursive_ks[0];
    let mut prev_log_msg_cols = n1 - prev_log_num_interleaved;
    let mut prev_log_inv_rate = config.log_inv_rates[1]; // wtns_1's rate
    let mut next_root_idx = 1usize;
    let mut recursive_proof_idx = 0usize;
    let mut n_current = n1;

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        if n_current < k_i {
            return false;
        }
        let mut level_rs = Vec::with_capacity(k_i);
        for _ in 0..k_i {
            let ri = challenger.sample_f128();
            ris.push(ri);
            level_rs.push(ri);
            t_r = running_quad.eval(ri);
            if tx_idx >= proof.sumcheck_transcript.len() {
                return false;
            }
            let msg = proof.sumcheck_transcript[tx_idx];
            tx_idx += 1;
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            running_quad = RoundQuad::from_msg(msg, t_r);
        }
        n_current -= k_i;

        if i == r - 1 {
            if tx_idx != proof.sumcheck_transcript.len() {
                return false;
            }
            let yr = &proof.final_proof.yr;
            if yr.len() != 1 << n_current {
                return false;
            }
            for v in yr {
                challenger.observe_f128(*v);
            }
            let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
            let prev_num_interleaved = 1usize << prev_log_num_interleaved;
            let num_queries_last = udr_queries(prev_log_inv_rate);
            let queries_last =
                sample_distinct_queries(challenger, prev_block_len, num_queries_last);
            let alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            if !verify_level_opens(
                &prev_root,
                prev_block_len,
                &queries_last,
                &proof.final_proof.opened_rows,
                prev_num_interleaved,
                &proof.final_proof.merkle_proof,
                config.merkle_hash,
            ) {
                return false;
            }

            let sks_vks_last = eval_sk_at_vks(n_current);
            let (basis_last_induced, enforced_sum_last) = induce_sumcheck_poly(
                n_current,
                &sks_vks_last,
                &proof.final_proof.opened_rows,
                &level_rs,
                &queries_last,
                &alpha_last,
            );
            let beta_last = challenger.sample_f128();
            t_r += beta_last * enforced_sum_last;
            basis_polys.push(basis_last_induced);
            basis_ris_starts.push(ris.len());
            basis_separations.push(beta_last);

            let yr_len = yr.len();
            let mut combined = vec![F128::ZERO; yr_len];
            for (k, basis) in basis_polys.iter().enumerate() {
                let start = basis_ris_starts[k];
                let residual = partial_eval_lsb(basis, &ris[start..]);
                if residual.len() != yr_len {
                    return false;
                }
                let sep = if k == 0 {
                    F128::ONE
                } else {
                    basis_separations[k - 1]
                };
                for (c, &r) in combined.iter_mut().zip(residual.iter()) {
                    *c += sep * r;
                }
            }
            let inner: F128 = yr
                .iter()
                .zip(combined.iter())
                .map(|(&y, &c)| y * c)
                .fold(F128::ZERO, |a, v| a + v);
            return inner == t_r;
        }

        if next_root_idx >= proof.recursive_roots.len() {
            return false;
        }
        let root_next = proof.recursive_roots[next_root_idx];
        next_root_idx += 1;
        challenger.observe_bytes(&root_next);

        let prev_block_len = 1usize << (prev_log_msg_cols + prev_log_inv_rate);
        let prev_num_interleaved = 1usize << prev_log_num_interleaved;
        let num_queries_i = udr_queries(prev_log_inv_rate);
        let queries_i = sample_distinct_queries(challenger, prev_block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));

        if recursive_proof_idx >= proof.recursive_proofs.len() {
            return false;
        }
        let rp = &proof.recursive_proofs[recursive_proof_idx];
        recursive_proof_idx += 1;
        if !verify_level_opens(
            &prev_root,
            prev_block_len,
            &queries_i,
            &rp.opened_rows,
            prev_num_interleaved,
            &rp.merkle_proof,
            config.merkle_hash,
        ) {
            return false;
        }

        let sks_vks_i = eval_sk_at_vks(n_current);
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_current,
            &sks_vks_i,
            &rp.opened_rows,
            &level_rs,
            &queries_i,
            &alpha_i,
        );

        if tx_idx >= proof.sumcheck_transcript.len() {
            return false;
        }
        let intro_msg_i = proof.sumcheck_transcript[tx_idx];
        tx_idx += 1;
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let intro_quad_i = RoundQuad::from_msg(intro_msg_i, enforced_sum_i);
        let beta_i = challenger.sample_f128();
        running_quad = RoundQuad::fold(&running_quad, &intro_quad_i, beta_i);
        t_r += beta_i * enforced_sum_i;
        basis_polys.push(basis_i_induced);
        basis_ris_starts.push(ris.len());
        basis_separations.push(beta_i);

        prev_root = root_next;
        let k_next = config.recursive_ks[i + 1];
        if n_current < k_next {
            return false;
        }
        prev_log_num_interleaved = k_next;
        prev_log_msg_cols = n_current - k_next;
        prev_log_inv_rate = config.log_inv_rates[i + 2];
    }

    unreachable!("loop should return at i = r - 1")
}

#[cfg(test)]
mod tests {
    /// Lane opus-pcs: strip-wise message accumulation (as fused into the
    /// GFNI basis kernel) equals the two-pass block reduction and a scalar
    /// oracle, for the ranked block length and short spans; negative control.
    #[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
    #[test]
    fn opus_strip_msg_accumulate_matches_block_reduce() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0x0905_F8B0);
        for &n in &[64usize, 128, 2048] {
            let f = rng.sample_f128_vec(n);
            let b = rng.sample_f128_vec(n);
            let whole = unsafe { msg_reduce_avx512(&f, &b) };
            let mut oracle = (F128::ZERO, F128::ZERO);
            for k in (0..n).step_by(2) {
                oracle.0 += f[k] * b[k];
                oracle.1 += (f[k] + f[k + 1]) * (b[k] + b[k + 1]);
            }
            assert_eq!(whole, oracle, "two-pass vs scalar n={n}");
            let mut u0 = unsafe { crate::field::gf2_128::x86_64::WideGhashX4::zero() };
            let mut u2 = unsafe { crate::field::gf2_128::x86_64::WideGhashX4::zero() };
            for s in (0..n).step_by(64) {
                unsafe { msg_accumulate_avx512(&f[s..s + 64], &b[s..s + 64], &mut u0, &mut u2) };
            }
            let fused = unsafe { (u0.fold().reduce(), u2.fold().reduce()) };
            assert_eq!(fused, whole, "strip-fused vs two-pass n={n}");
            let mut bad = b.clone();
            bad[n - 2] += F128::ONE;
            let mut v0 = unsafe { crate::field::gf2_128::x86_64::WideGhashX4::zero() };
            let mut v2 = unsafe { crate::field::gf2_128::x86_64::WideGhashX4::zero() };
            for s in (0..n).step_by(64) {
                unsafe { msg_accumulate_avx512(&f[s..s + 64], &bad[s..s + 64], &mut v0, &mut v2) };
            }
            let corrupted = unsafe { (v0.fold().reduce(), v2.fold().reduce()) };
            assert_ne!(corrupted, whole, "corrupted basis went undetected n={n}");
        }
    }

    #[test]
    fn opus_tntt_chunk_log_preserves_outputs() {
        // Lane opus-pcs: any site-17/18 CHUNK_LOG only re-partitions the same
        // butterflies. Check the retained outputs against the per-layer
        // reference (site 17) and the incumbent C=16 (site 18) on a 16-thread
        // pool (the scorer's split floor), incl. the ranked L0 shape d18/k10.
        use crate::challenger::Challenger;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        pool.install(|| {
            let set = |c: Option<usize>| OPUS_TNTT_C_TEST.with(|x| x.set(c));
            for &log_d in &[11usize, 12, 14, 16, 18] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n = 1usize << log_d;
                let mut ch = crate::challenger::RandomChallenger::new(0xC13 ^ log_d as u64);
                let base: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
                for &top in &[1usize, 4, 5, 8, log_d] {
                    let mut want = base.clone();
                    transpose_forward_ntt_dense_layers_per_layer(&ntt, &mut want, top);
                    for dead in [0usize, 2.min(top)] {
                        let keep = 1usize << (log_d - dead);
                        for c in [10usize, 13, 14, 16, 20] {
                            set(Some(c));
                            let mut got = base.clone();
                            transpose_forward_ntt_dense_layers_blocked_dead(&ntt, &mut got, top, dead);
                            assert_eq!(
                                &got[..keep],
                                &want[..keep],
                                "site17 log_d={log_d} top={top} dead={dead} C={c}"
                            );
                        }
                    }
                }
                let mut positions: Vec<usize> = (0..218usize)
                    .map(|i| (i.wrapping_mul(2_654_435_761) + 11) & (n - 1))
                    .collect();
                positions.sort_unstable();
                positions.dedup();
                let values = ch.sample_f128_vec(positions.len());
                for k_override in [None, Some(4usize), Some(10)] {
                    for dead in [0usize, 1, 3] {
                        set(Some(16));
                        let (want, _) = transpose_forward_ntt_sparse_inner_dead(
                            &ntt, &positions, &values, log_d, None, k_override, dead,
                        );
                        let keep = 1usize << (log_d - dead);
                        for c in [10usize, 12, 13, 14, 20] {
                            set(Some(c));
                            let (got, _) = transpose_forward_ntt_sparse_inner_dead(
                                &ntt, &positions, &values, log_d, None, k_override, dead,
                            );
                            assert_eq!(
                                &got[..keep],
                                &want[..keep],
                                "site18 log_d={log_d} k={k_override:?} dead={dead} C={c}"
                            );
                        }
                        // Negative control: a perturbed value must change the output.
                        set(Some(13));
                        let mut bad = values.clone();
                        bad[0] += F128::ONE;
                        let (got, _) = transpose_forward_ntt_sparse_inner_dead(
                            &ntt, &positions, &bad, log_d, None, k_override, dead,
                        );
                        assert_ne!(&got[..keep], &want[..keep], "perturbed input undetected");
                    }
                }
            }
            set(None);
        });
    }

    #[test]
    fn x1_batch_inverse_or_zero_matches_scalar_and_preserves_positions() {
        use super::{batch_inverse_or_zero, eval_sk_at_vks};
        use crate::field::F128;
        let scalar = |values: &[F128]| -> Vec<F128> {
            values
                .iter()
                .map(|&v| if v.is_zero() { F128::ZERO } else { v.inv() })
                .collect()
        };
        let x = |lo: u64, hi: u64| F128::new(lo, hi);
        let cases: Vec<Vec<F128>> = vec![
            vec![],
            vec![F128::ZERO],
            vec![x(7, 0)],
            vec![F128::ZERO, F128::ZERO, F128::ZERO],
            vec![F128::ZERO, x(3, 9), F128::ZERO, x(0x1234, 0xffff_0000_0000_0001), F128::ZERO],
            vec![x(1, 0), x(2, 0), x(u64::MAX, u64::MAX), x(5, 1 << 63)],
            (1..40u64).map(|i| x(i.wrapping_mul(0x9e37_79b9_7f4a_7c15), i << 40)).collect(),
            eval_sk_at_vks(10),
            eval_sk_at_vks(16),
        ];
        for values in &cases {
            assert_eq!(batch_inverse_or_zero(values), scalar(values), "{values:?}");
        }
    }

    #[test]
    fn sparse_transpose_dead_top_keeps_prefix() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0xDEAD_B077);
        for &(log_d, n_pos) in &[(6usize, 5usize), (12, 40), (14, 200), (16, 218)] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut positions: Vec<usize> = (0..n_pos)
                .map(|i| (i.wrapping_mul(2_654_435_761) + 7) & ((1usize << log_d) - 1))
                .collect();
            positions.sort_unstable();
            positions.dedup();
            let values = rng.sample_f128_vec(positions.len());
            for dead in 0..=3usize {
                for k_override in [None, Some(4usize), Some(8)] {
                    let (full, _) = transpose_forward_ntt_sparse_inner(
                        &ntt, &positions, &values, log_d, None, k_override,
                    );
                    let (cut, _) = transpose_forward_ntt_sparse_inner_dead(
                        &ntt, &positions, &values, log_d, None, k_override, dead,
                    );
                    let keep = 1usize << (log_d - dead);
                    assert_eq!(
                        &full[..keep],
                        &cut[..keep],
                        "log_d={log_d} dead={dead} k={k_override:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn fold_with_deferred_basis_matches_two_fold_oracle() {
        use crate::challenger::Challenger;
        let mut rng = crate::challenger::RandomChallenger::new(0xDEF0_0E01);
        for n in [64usize, 1024, 1 << 14] {
            let f = rng.sample_f128_vec(n);
            let b = rng.sample_f128_vec(n);
            let basis = rng.sample_f128_vec(n);
            let r = rng.sample_f128();
            for alpha in [F128::ONE, rng.sample_f128()] {
                let (nf, nb, msg) = fold_and_msg_lsb_inner(&f, &b, r, None, None, Some((&basis, alpha)));
                let half = n / 2;
                let mut u_0 = F128::ZERO;
                let mut u_2 = F128::ZERO;
                for j in 0..half {
                    let ef = f[2 * j] + r * (f[2 * j] + f[2 * j + 1]);
                    let eb = b[2 * j]
                        + r * (b[2 * j] + b[2 * j + 1])
                        + alpha * (basis[2 * j] + r * (basis[2 * j] + basis[2 * j + 1]));
                    assert_eq!(nf[j], ef, "n={n} j={j} f");
                    assert_eq!(nb[j], eb, "n={n} j={j} b");
                }
                let mut k = 0;
                while k + 1 < half {
                    u_0 += nf[k] * nb[k];
                    u_2 += (nf[k] + nf[k + 1]) * (nb[k] + nb[k + 1]);
                    k += 2;
                }
                assert_eq!((msg.u_0, msg.u_2), (u_0, u_2), "n={n} message");
            }
        }
    }

    use super::*;

    #[test]
    fn query_phase_gathers_match_sequential_oracles() {
        let mut state = 0x9A7B_5EED_F00D_u64;
        let mut random_u64 = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &(log_leaves, n_queries, row_len) in &[(12usize, 218usize, 64usize), (4, 3, 2)] {
            let num_leaves = 1usize << log_leaves;
            let mut tree: Vec<Hash> = Vec::with_capacity(2 * num_leaves - 1);
            for _ in 0..(2 * num_leaves - 1) {
                let mut h = [0u8; 32];
                h[..8].copy_from_slice(&random_u64().to_le_bytes());
                h[8..16].copy_from_slice(&random_u64().to_le_bytes());
                tree.push(h);
            }
            let codeword: Vec<F128> = (0..num_leaves * row_len)
                .map(|_| F128::new(random_u64(), random_u64()))
                .collect();
            let mut queries: Vec<usize> = (0..n_queries)
                .map(|_| random_u64() as usize % num_leaves)
                .collect();
            queries.sort_unstable();
            queries.dedup();

            let row = |q: usize| &codeword[q * row_len..(q + 1) * row_len];
            let rows_seq = gather_opened_rows(&queries, row, false);
            let rows_par = gather_opened_rows(&queries, row, true);
            assert_eq!(rows_par, rows_seq, "rows log_leaves={log_leaves}");

            let proof_seq = multi_proof_gather(&tree, num_leaves, &queries, false);
            let proof_par = multi_proof_gather(&tree, num_leaves, &queries, true);
            assert_eq!(proof_par, proof_seq, "multi-proof log_leaves={log_leaves}");
            assert_eq!(
                proof_seq,
                merkle::merkle_multi_proof(&tree, num_leaves, &queries),
                "sequential arm drifted from the merkle oracle"
            );

            let mut bad_codeword = codeword.clone();
            bad_codeword[queries[0] * row_len] += F128::ONE;
            let bad_row = |q: usize| &bad_codeword[q * row_len..(q + 1) * row_len];
            assert_ne!(
                gather_opened_rows(&queries, bad_row, true),
                rows_seq,
                "corrupted row went undetected log_leaves={log_leaves}"
            );
            let sibling_indices = merkle::merkle_multi_proof_sibling_indices(num_leaves, &queries);
            assert!(!sibling_indices.is_empty());
            let mut bad_tree = tree.clone();
            bad_tree[sibling_indices[0]][0] ^= 1;
            assert_ne!(
                multi_proof_gather(&bad_tree, num_leaves, &queries, true),
                proof_seq,
                "corrupted sibling went undetected log_leaves={log_leaves}"
            );
        }
    }

    #[test]
    fn direct_ab_materialization_matches_full_basis_oracle() {
        let mut state = 0xD1CE_F01D_u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(state, state.rotate_left(29))
        };
        let n = 1usize << 8;
        let f: Vec<F128> = (0..n).map(|_| random()).collect();
        let ordinary_c: Vec<F128> = (0..n).map(|_| random()).collect();
        let r0 = random();
        let r1 = random();
        let suffix: Vec<F128> = (0..8).map(|_| random()).collect();
        let gamma = random();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| random())
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|value| gamma * value)
        .collect();
        let direct_full =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let combined_full: Vec<F128> = ordinary_c
            .iter()
            .zip(direct_full)
            .map(|(&ordinary, direct)| ordinary + direct)
            .collect();
        let (eq_lo, eq_hi) = super::super::ring_switch::build_eq_split(&suffix[2..], 3);
        let direct = vec![super::super::ring_switch::DirectFold2Factors {
            eq_lo,
            eq_hi,
            low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
            table: super::super::ring_switch::build_fold_byte_table(&scaled_rdp),
            products: None,
        }];

        let mut want_f = f.clone();
        let mut want_b = combined_full;
        partial_eval_lsb_one(&mut want_f, r0);
        partial_eval_lsb_one(&mut want_b, r0);
        partial_eval_lsb_one(&mut want_f, r1);
        partial_eval_lsb_one(&mut want_b, r1);
        let want_msg = round_msg_lsb(&want_f, &want_b);
        let (got_f, got_b, got_msg) = materialize_direct_ab_fold2(f, ordinary_c, &direct, r0, r1);
        assert_eq!(got_f, want_f);
        assert_eq!(got_b, want_b);
        assert_eq!(got_msg, want_msg);
    }

    #[test]
    fn direct_all_full_proof_matches_ordinary_transcript() {
        use crate::challenger::Challenger;

        let log_n = 12;
        let initial_k = 2;
        let k_0 = 2;
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_AB02);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let suffix_ab: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let suffix_c: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp_ab: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let scaled_rdp_c: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let basis_ab =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix_ab), &scaled_rdp_ab);
        let basis_c =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix_c), &scaled_rdp_c);
        let combined_basis: Vec<F128> = basis_ab
            .iter()
            .zip(basis_c)
            .map(|(&ab, c)| ab + c)
            .collect();
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);
        let (round0, lookahead) = super::super::round0_and_round1_lookahead(&poly, &combined_basis);
        let direct = [(&suffix_ab, &scaled_rdp_ab), (&suffix_c, &scaled_rdp_c)]
            .into_iter()
            .map(|(suffix, scaled_rdp)| {
                let (eq_lo, eq_hi) =
                    super::super::ring_switch::build_eq_split(&suffix[2..], (log_n - 2) / 2);
                super::super::ring_switch::DirectFold2Factors {
                    eq_lo,
                    eq_hi,
                    low_eq: build_eq_table(&suffix[..2]).try_into().unwrap(),
                    table: super::super::ring_switch::build_fold_byte_table(scaled_rdp),
                    products: None,
                }
            })
            .collect();

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates
                .iter()
                .map(|&rate| udr_queries(rate))
                .collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_n - initial_k,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mut ordinary_challenger =
            crate::challenger::FsChallenger::new(b"direct-all-proof-byte-oracle");
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            None,
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-all-proof-byte-oracle");
        let got = recursive_prover_with_basis_direct_ab_fold2(
            &cfg,
            poly,
            Vec::new(),
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            lookahead,
            None,
            &mut direct_challenger,
        );

        assert_eq!(got, ordinary);
    }

    #[test]
    fn direct_fold4_full_proof_and_claim_bytes_match_ordinary() {
        use crate::challenger::Challenger;

        for (log_n, seed) in [(12usize, 0xD1CE_F004u64), (13, 0xD1CE_F005)] {
            let initial_k = 6;
            let k_0 = 2;
            let log_inv_rate = 3;
            let mut rng = crate::challenger::RandomChallenger::new(seed);
            let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
            let suffix_ab: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let suffix_c: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
            let scaled_rdp_ab: Vec<F128> = build_eq_table(
                &(0..crate::pcs::LOG_PACKING)
                    .map(|_| rng.sample_f128())
                    .collect::<Vec<_>>(),
            );
            let scaled_rdp_c: Vec<F128> = build_eq_table(
                &(0..crate::pcs::LOG_PACKING)
                    .map(|_| rng.sample_f128())
                    .collect::<Vec<_>>(),
            );
            let basis_ab = super::super::ring_switch::fold_b128_elems(
                &build_eq_table(&suffix_ab),
                &scaled_rdp_ab,
            );
            let basis_c = super::super::ring_switch::fold_b128_elems(
                &build_eq_table(&suffix_c),
                &scaled_rdp_c,
            );
            let combined_basis: Vec<F128> = basis_ab
                .iter()
                .zip(basis_c.iter())
                .map(|(&ab, &c)| ab + c)
                .collect();
            let target = poly
                .iter()
                .zip(combined_basis.iter())
                .map(|(&f, &b)| f * b)
                .fold(F128::ZERO, |acc, value| acc + value);

            let direct: Vec<super::super::ring_switch::DirectFold4Factors> = [
                (&suffix_ab, &scaled_rdp_ab, &basis_ab),
                (&suffix_c, &scaled_rdp_c, &basis_c),
            ]
            .into_iter()
            .map(|(suffix, scaled_rdp, basis)| {
                let mut products = [F128::ZERO; 256];
                for high in 0..poly.len() / 16 {
                    for e in 0..16 {
                        for d in 0..16 {
                            products[16 * e + d] += poly[16 * high + e] * basis[16 * high + d];
                        }
                    }
                }
                let (eq_lo, eq_hi) =
                    super::super::ring_switch::build_eq_split(&suffix[4..], (log_n - 4) / 2);
                super::super::ring_switch::DirectFold4Factors {
                    eq_lo,
                    eq_hi,
                    low_eq: build_eq_table(&suffix[..4]).try_into().unwrap(),
                    table: super::super::ring_switch::build_fold_byte_table(scaled_rdp),
                    products,
                }
            })
            .collect();
            let (round0, round1, round2, round3) =
                super::super::messages_from_direct_products_fold4(&direct);
            let (round0_ref, round1_ref) =
                super::super::round0_and_round1_lookahead(&poly, &combined_basis);
            assert_eq!(round0, round0_ref, "round-0 message from products");
            assert_eq!(round1, round1_ref, "round-1 lookahead from products");

            let log_inv_rates = vec![log_inv_rate, log_inv_rate];
            let cfg = ProverConfig {
                log_inv_rates: log_inv_rates.clone(),
                recursive_steps: 1,
                initial_log_msg_cols: log_n - initial_k,
                initial_log_num_interleaved: initial_k,
                initial_k,
                recursive_log_msg_cols: vec![log_n - initial_k - k_0],
                recursive_ks: vec![k_0],
                queries: log_inv_rates
                    .iter()
                    .map(|&rate| udr_queries(rate))
                    .collect(),
                grinding_bits: vec![0; log_inv_rates.len()],
                fold_grinding_bits: vec![0; 2],
                ood_samples: vec![0; 2],
                merkle_hash: Default::default(),
            };
            let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
            let wtns_0 = ligero_commit(
                &poly,
                log_n - initial_k,
                initial_k,
                log_inv_rate,
                &ntt_0,
                HashKind::Sha256,
            );

            let mut ordinary_challenger =
                crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
            let ordinary = recursive_prover_with_basis_precomputed_round0(
                &cfg,
                poly.clone(),
                combined_basis.clone(),
                target,
                &wtns_0.mat,
                &wtns_0.tree,
                round0,
                None,
                &mut ordinary_challenger,
            );
            let mut direct_challenger =
                crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
            let got = recursive_prover_with_basis_direct_fold4(
                &cfg,
                poly,
                Vec::new(),
                direct,
                target,
                &wtns_0.mat,
                &wtns_0.tree,
                round0,
                round1,
                round2,
                round3,
                None,
                &mut direct_challenger,
            );

            assert_eq!(got, ordinary, "direct-fold4 proof differs at log_n={log_n}");
            assert_eq!(
                bincode::serialize(&(got.clone(), target))
                    .expect("serialize direct-fold4 proof/claim"),
                bincode::serialize(&(ordinary, target)).expect("serialize ordinary proof/claim"),
            );

            let v_cfg = VerifierConfig {
                log_inv_rates: log_inv_rates.clone(),
                recursive_steps: 1,
                initial_log_msg_cols: log_n - initial_k,
                initial_log_num_interleaved: initial_k,
                initial_k,
                recursive_log_msg_cols: vec![log_n - initial_k - k_0],
                recursive_ks: vec![k_0],
                queries: log_inv_rates
                    .iter()
                    .map(|&rate| udr_queries(rate))
                    .collect(),
                grinding_bits: vec![0; log_inv_rates.len()],
                fold_grinding_bits: vec![0; 2],
                ood_samples: vec![0; 2],
                merkle_hash: Default::default(),
            };
            let mut verifier_challenger =
                crate::challenger::FsChallenger::new(b"direct-fold4-proof-byte-oracle");
            assert!(recursive_verifier_with_basis(
                &v_cfg,
                &got,
                &combined_basis,
                target,
                &wtns_0.root(),
                &mut verifier_challenger,
            ));
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_and_msg_leaf_x86_matches_generic() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(8usize, 0usize), (32, 4), (64, 16), (2048, 0)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }
            let (r_x4, r_x64) = unsafe {
                use crate::field::gf2_128::x86_64::ghash_shift64_x4;
                use core::arch::x86_64::*;
                let r_x4 =
                    _mm512_broadcast_i32x4(_mm_set_epi64x(r.hi as i64, r.lo as i64));
                (r_x4, ghash_shift64_x4(r_x4))
            };

            for stream in [false, true] {
                let mut fc_x86 = vec![F128::ZERO; n_pairs];
                let mut bc_x86 = vec![F128::ZERO; n_pairs];
                let (u0_x86, u2_x86) = unsafe {
                    super::fold_and_msg_chunk_x86::<false>(
                        &f, &b, base, &mut fc_x86, &mut bc_x86, r, r_x4, r_x64, stream, None,
                        None,
                    )
                };
                assert_eq!(fc_ref, fc_x86, "folded f mismatch n_pairs={n_pairs}");
                assert_eq!(bc_ref, bc_x86, "folded b mismatch n_pairs={n_pairs}");
                assert_eq!(u0_ref, u0_x86, "u0 mismatch n_pairs={n_pairs}");
                assert_eq!(u2_ref, u2_x86, "u2 mismatch n_pairs={n_pairs}");
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold_and_msg_nt_leaf_matches_generic() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(8usize, 0usize), (32, 4), (64, 16)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }

            let mut fc_nt = vec![F128::ZERO; n_pairs];
            let mut bc_nt = vec![F128::ZERO; n_pairs];
            let (u0_nt, u2_nt) =
                unsafe { fold_and_msg_chunk_nt_neon(&f, &b, base, &mut fc_nt, &mut bc_nt, r) };
            assert_eq!(fc_ref, fc_nt, "folded f mismatch n_pairs={n_pairs}");
            assert_eq!(bc_ref, bc_nt, "folded b mismatch n_pairs={n_pairs}");
            assert_eq!(u0_ref, u0_nt, "u0 mismatch n_pairs={n_pairs}");
            assert_eq!(u2_ref, u2_nt, "u2 mismatch n_pairs={n_pairs}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fold_and_msg_soa_leaf_matches_nt_and_generic() {
        let mut state = 0x0fed_cba9_8765_4321_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for (n_pairs, base) in [(2usize, 0usize), (8, 0), (32, 4), (64, 16), (2048, 2048)] {
            let total = 2 * (base + n_pairs);
            let f: Vec<F128> = (0..total).map(|_| f128()).collect();
            let b: Vec<F128> = (0..total).map(|_| f128()).collect();
            let r = f128();

            let mut fc_ref = vec![F128::ZERO; n_pairs];
            let mut bc_ref = vec![F128::ZERO; n_pairs];
            crate::field::f128_slice::fold_pairs(&f, base, &mut fc_ref, r);
            crate::field::f128_slice::fold_pairs(&b, base, &mut bc_ref, r);
            let mut u0_ref = F128::ZERO;
            let mut u2_ref = F128::ZERO;
            let mut k = 0;
            while k + 1 < n_pairs {
                let (f0, f1, b0, b1) = (fc_ref[k], fc_ref[k + 1], bc_ref[k], bc_ref[k + 1]);
                u0_ref += f0 * b0;
                u2_ref += (f0 + f1) * (b0 + b1);
                k += 2;
            }

            let mut fc_nt = vec![F128::ZERO; n_pairs];
            let mut bc_nt = vec![F128::ZERO; n_pairs];
            let (u0_nt, u2_nt) =
                unsafe { fold_and_msg_chunk_nt_neon(&f, &b, base, &mut fc_nt, &mut bc_nt, r) };

            let mut fc_soa = vec![F128::ZERO; n_pairs];
            let mut bc_soa = vec![F128::ZERO; n_pairs];
            let (u0_soa, u2_soa) = unsafe {
                fold_and_msg_chunk_nt_neon_soa::<true>(&f, &b, base, &mut fc_soa, &mut bc_soa, r)
            };
            let mut fc_soa_r = vec![F128::ZERO; n_pairs];
            let mut bc_soa_r = vec![F128::ZERO; n_pairs];
            let (u0_soa_r, u2_soa_r) = unsafe {
                fold_and_msg_chunk_nt_neon_soa::<false>(
                    &f,
                    &b,
                    base,
                    &mut fc_soa_r,
                    &mut bc_soa_r,
                    r,
                )
            };

            assert_eq!(fc_ref, fc_soa, "folded f mismatch n_pairs={n_pairs}");
            assert_eq!(bc_ref, bc_soa, "folded b mismatch n_pairs={n_pairs}");
            assert_eq!(u0_ref, u0_soa, "u0 vs generic n_pairs={n_pairs}");
            assert_eq!(u2_ref, u2_soa, "u2 vs generic n_pairs={n_pairs}");
            assert_eq!(fc_nt, fc_soa, "folded f vs NT n_pairs={n_pairs}");
            assert_eq!(bc_nt, bc_soa, "folded b vs NT n_pairs={n_pairs}");
            assert_eq!(u0_nt, u0_soa, "u0 vs NT n_pairs={n_pairs}");
            assert_eq!(u2_nt, u2_soa, "u2 vs NT n_pairs={n_pairs}");
            assert_eq!(fc_soa, fc_soa_r, "folded f NT vs stp n_pairs={n_pairs}");
            assert_eq!(bc_soa, bc_soa_r, "folded b NT vs stp n_pairs={n_pairs}");
            assert_eq!(u0_soa, u0_soa_r, "u0 NT vs stp n_pairs={n_pairs}");
            assert_eq!(u2_soa, u2_soa_r, "u2 NT vs stp n_pairs={n_pairs}");
        }
    }

    #[test]
    fn fold_and_msg_serial_leaf_matches_two_mul() {
        let mut state = 0xA5A5_5A5A_1234_5678_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut f128 = || F128 {
            lo: next(),
            hi: next(),
        };
        for n in [2usize, 4, 8, 16, 64, 256] {
            let f: Vec<F128> = (0..n).map(|_| f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| f128()).collect();
            let r = f128();
            let one_plus_r = F128::ONE + r;
            let half = n / 2;
            let mut nf_ref = Vec::with_capacity(half);
            let mut nb_ref = Vec::with_capacity(half);
            for j in 0..half {
                nf_ref.push(f[2 * j] * one_plus_r + f[2 * j + 1] * r);
                nb_ref.push(b[2 * j] * one_plus_r + b[2 * j + 1] * r);
            }
            let mut u0 = F128::ZERO;
            let mut u2 = F128::ZERO;
            let mut k = 0;
            while k + 1 < half {
                u0 += nf_ref[k] * nb_ref[k];
                u2 += (nf_ref[k] + nf_ref[k + 1]) * (nb_ref[k] + nb_ref[k + 1]);
                k += 2;
            }
            let (nf, nb, msg) = fold_and_msg_lsb(&f, &b, r, None);
            assert_eq!(&nf[..], nf_ref.as_slice(), "nf n={n}");
            assert_eq!(&nb[..], nb_ref.as_slice(), "nb n={n}");
            assert_eq!(msg.u_0, u0, "u0 n={n}");
            assert_eq!(msg.u_2, u2, "u2 n={n}");
        }
    }

    fn blake3_m29_udr_example() -> LigeritoSecurityConfig {
        LigeritoSecurityConfig::derive_paper_compatible(29, 1, 100).expect("derive m29 fast")
    }

    #[test]
    fn ligerito_security_config_m29_toml_loads() {
        let toml_str = include_str!("../../configs/ligerito/m29_fast.toml");
        let cfg = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_fast.toml must parse and validate");
        assert_eq!(cfg.m, 29);
        assert_eq!(cfg.log_n, 22);
        assert_eq!(cfg.initial_k, 6);
        assert_eq!(cfg.hash, "sha256");
        assert_eq!(cfg.levels.len(), 5);
        assert_eq!(cfg.levels[0].regime, SoundnessRegime::JohnsonOod);
        assert_eq!(cfg.levels[0].queries, 218);
        assert_eq!(cfg.levels[0].grinding_bits, 0);
        assert!(cfg.levels[0].fold_grinding_bits > 0);
        assert_eq!(cfg.levels[0].ood_samples, 0); // L0: bound by eval claim
        assert!(cfg.levels[1].ood_samples >= 1);
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.queries[0], 218);

        let toml_str = include_str!("../../configs/ligerito/m29_slim.toml");
        let cfg_slim = LigeritoSecurityConfig::from_toml_str(toml_str)
            .expect("m29_slim.toml must parse and validate");
        assert_eq!(cfg_slim.levels[0].log_inv_rate, 2);
        assert_eq!(cfg_slim.levels[0].queries, 90);
        assert_eq!(cfg_slim.levels[0].grinding_bits, 16);
        let (pv_slim, _vc_slim) = cfg_slim.to_prover_verifier_configs().unwrap();
        let default_slim = default_config(22, 6, 2).unwrap();
        assert_eq!(pv_slim.log_inv_rates, default_slim.log_inv_rates);
        assert_eq!(pv_slim.recursive_ks, default_slim.recursive_ks);
    }

    #[test]
    #[ignore]
    fn regen_embedded_tomls() {
        for m in [22usize, 29, 32] {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m{m}_{}: {e}", profile.as_str()));
                let toml = cfg.to_toml_string().expect("serialize");
                eprintln!(
                    "\n# ====== configs/ligerito/m{m}_{}.toml ======",
                    profile.as_str()
                );
                eprintln!("{toml}");
            }
        }
    }

    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_pg() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_pg_bits = 50.0; // very wrong
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    #[test]
    fn ligerito_security_config_rejects_paper_inconsistent_eps_query() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_query_bits += 5.0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("doesn't match") && err.contains("prediction"),
            "expected paper-mismatch error, got: {err}"
        );
    }

    #[test]
    fn ligerito_all_embedded_configs_validate() {
        for &(key, toml) in EMBEDDED_CONFIGS {
            LigeritoSecurityConfig::from_toml_str(toml).unwrap_or_else(|e| {
                panic!(
                    "embedded config m={} profile={} invalid: {e}",
                    key.0,
                    key.1.as_str()
                )
            });
        }
    }

    #[test]
    fn ligerito_derive_paper_compatible_for_all_embedded() {
        let pairs: &[(usize, usize)] = &[(22, 1), (28, 1), (29, 1), (29, 2), (30, 1), (30, 2)];
        for &(m, r) in pairs {
            let cfg = LigeritoSecurityConfig::derive_paper_compatible(m, r, 100)
                .unwrap_or_else(|e| panic!("derive m={m} r={r}: {e}"));
            cfg.validate()
                .unwrap_or_else(|e| panic!("derived m={m} r={r} fails validate: {e}"));
        }
        for m in 22..=35usize {
            for profile in [
                LigeritoProfile::Fast,
                LigeritoProfile::Slim,
                LigeritoProfile::Secure,
            ] {
                let cfg = LigeritoSecurityConfig::derive_profile(m, profile)
                    .unwrap_or_else(|e| panic!("derive m={m} {}: {e}", profile.as_str()));
                cfg.validate().unwrap_or_else(|e| {
                    panic!("derived m={m} {} fails validate: {e}", profile.as_str())
                });
            }
        }
    }

    #[test]
    fn ligerito_prover_config_for_lookup() {
        let pv = prover_config_for(22, 6, LigeritoProfile::Fast).expect("m29 fast must load");
        assert_eq!(pv.queries[0], 218);
        assert_eq!(pv.fold_grinding_bits[0], 16);

        let pv = prover_config_for(22, 6, LigeritoProfile::Slim).expect("m29 slim must load");
        assert_eq!(pv.queries[0], 90);
        assert_eq!(pv.grinding_bits[0], 16);

        let pv = prover_config_for(22, 6, LigeritoProfile::Secure).expect("m29 secure must load");
        assert!(pv.queries[0] > 280);
        assert_eq!(pv.ood_samples.iter().sum::<usize>(), 0);

        let err = prover_config_for(29, 6, LigeritoProfile::Fast).unwrap_err();
        assert!(
            err.contains("no security config registered"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ligerito_security_config_toml_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let s = cfg.to_toml_string().expect("serialize");
        let back = LigeritoSecurityConfig::from_toml_str(&s).expect("deserialize");
        assert_eq!(back.levels.len(), cfg.levels.len());
        assert_eq!(back.levels[0].queries, cfg.levels[0].queries);
        assert_eq!(back.levels[0].grinding_bits, cfg.levels[0].grinding_bits);
        assert_eq!(back.final_block.yr_log_n, cfg.final_block.yr_log_n);
    }

    #[test]
    fn ligerito_security_config_validates() {
        let cfg = blake3_m29_udr_example();
        cfg.validate()
            .unwrap_or_else(|e| panic!("validate failed: {e}"));
    }

    #[test]
    fn ligerito_security_config_hash_field_selects_merkle_hash() {
        let mut cfg = blake3_m29_udr_example();
        assert_eq!(cfg.hash, "sha256", "example config baseline");
        let (p, v) = cfg.to_prover_verifier_configs().expect("sha256 configs");
        assert_eq!(p.merkle_hash, HashKind::Sha256);
        assert_eq!(v.merkle_hash, HashKind::Sha256);

        cfg.hash = "blake3".into();
        let (p, v) = cfg.to_prover_verifier_configs().expect("blake3 configs");
        assert_eq!(p.merkle_hash, HashKind::Blake3);
        assert_eq!(v.merkle_hash, HashKind::Blake3);

        cfg.validate().expect("blake3 config validates");
        let back = LigeritoSecurityConfig::from_toml_str(&cfg.to_toml_string().unwrap())
            .expect("toml roundtrip");
        assert_eq!(back.merkle_hash().unwrap(), HashKind::Blake3);
    }

    #[test]
    fn ligerito_security_config_rejects_unknown_hash() {
        let mut cfg = blake3_m29_udr_example();
        cfg.hash = "keccak256".into();
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("hash") && err.contains("keccak256"),
            "err = {err}"
        );
        assert!(cfg.to_prover_verifier_configs().is_err());
    }

    #[test]
    fn embedded_configs_all_declare_a_supported_hash() {
        for &((m, profile), toml) in EMBEDDED_CONFIGS {
            let cfg = LigeritoSecurityConfig::from_toml_str(toml)
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
            cfg.merkle_hash()
                .unwrap_or_else(|e| panic!("m{m} {profile:?}: {e}"));
        }
    }

    #[test]
    fn ligerito_security_config_rejects_insufficient_queries() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].expected_eps_query_bits = 50.0; // < target 100 (grinding 0)
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("expected_eps_query_bits"), "err = {err}");
    }

    #[test]
    fn ligerito_security_config_rejects_udr_with_eta() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].eta = Some(0.02); // eta is Johnson-only — should fail
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("udr") && err.contains("eta"), "err = {err}");
    }

    #[test]
    fn ligerito_security_config_rejects_udr_without_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].proximity_loss = None; // missing!
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("udr") && err.contains("proximity_loss"),
            "err = {err}"
        );
    }

    #[test]
    fn ligerito_security_config_rejects_johnson_with_proximity_loss() {
        let mut cfg = blake3_m29_udr_example();
        cfg.levels[0].regime = SoundnessRegime::JohnsonOod;
        cfg.levels[0].eta = Some(0.02);
        cfg.levels[0].proximity_loss = Some(0.01);
        let err = cfg.validate().unwrap_err();
        assert!(
            err.contains("proximity_loss") && err.contains("udr"),
            "err = {err}"
        );
    }

    #[test]
    fn ligerito_security_config_udr_regime_validates() {
        let mut cfg = blake3_m29_udr_example();
        let eps_star = 0.0f64;
        let rho = 0.5f64;
        let delta = 1.0 - rho;
        let n = ((cfg.levels[0].log_msg_cols + cfg.levels[0].log_inv_rate) as f64).exp2();
        let gamma = delta / 2.0 - 3.0 / (delta * n) - eps_star;
        let per_q = (1.0 / (1.0 - gamma)).log2();
        let queries = (100.0 / per_q).ceil() as usize;
        let log_a_base = (gamma * n + 1.0).log2();
        let eps_pg = 128.0 - log_a_base;
        cfg.levels[0].regime = SoundnessRegime::Udr;
        cfg.levels[0].eta = None;
        cfg.levels[0].proximity_loss = Some(eps_star);
        cfg.levels[0].queries = queries;
        cfg.levels[0].grinding_bits = 0;
        cfg.levels[0].fold_grinding_bits = (100.0 - eps_pg).ceil().max(0.0) as usize;
        cfg.levels[0].expected_eps_pg_bits = (eps_pg * 10.0).round() / 10.0;
        cfg.levels[0].expected_eps_query_bits = ((queries as f64 * per_q) * 10.0).round() / 10.0;
        cfg.validate()
            .unwrap_or_else(|e| panic!("UDR config failed to validate: {e}"));
    }

    #[test]
    fn ligerito_security_config_serde_roundtrip() {
        let cfg = blake3_m29_udr_example();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let back: LigeritoSecurityConfig = serde_json::from_str(&json).expect("deserialize");
        back.validate().expect("roundtripped config validates");
        assert_eq!(back.levels.len(), cfg.levels.len());
        assert_eq!(back.levels[0].queries, 241);
        assert_eq!(back.levels[0].grinding_bits, 0);
    }

    #[test]
    fn ligerito_security_config_drives_roundtrip_with_grinding() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x6817_D146);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![6usize, 0]; // L0 grinds 6 bits, L1 doesn't
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );
        assert_eq!(proof.grinding_nonces.len(), 2, "one nonce per level");

        let v_cfg = VerifierConfig {
            log_inv_rates,
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            ok,
            "verifier should accept proof with valid grinding nonces"
        );

        let mut bad_proof = proof.clone();
        bad_proof.grinding_nonces[0] = bad_proof.grinding_nonces[0].wrapping_add(1);
        let mut v_ch = crate::challenger::FsChallenger::new(b"pow-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &bad_proof, &b, target, &initial_root, &mut v_ch);
        assert!(
            !ok,
            "verifier must reject proof with tampered grinding nonce"
        );
    }

    #[test]
    fn ligerito_security_config_matches_default_config() {
        let cfg = blake3_m29_udr_example();
        let (pv, _vc) = cfg.to_prover_verifier_configs().unwrap();
        let default = default_config(22, 6, 1).unwrap();
        assert_eq!(pv.log_inv_rates, default.log_inv_rates);
        assert_eq!(pv.recursive_ks, default.recursive_ks);
        assert_eq!(pv.initial_k, default.initial_k);
    }

    #[test]
    fn partial_eval_then_eval_equals_full_eval() {
        let n = 6;
        let len = 1usize << n;
        let evals: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xDEAD_BEEF_CAFE_BABE),
                    0xA5A5 ^ i as u64,
                )
            })
            .collect();
        let point: Vec<F128> = (0..n)
            .map(|i| F128::new(0x1111 * (i as u64 + 1), 0x2222 * (i as u64 + 1)))
            .collect();

        let full = eval_mle_lsb(&evals, &point);
        let k = 3;
        let (lo, hi) = point.split_at(k);
        let residual = partial_eval_lsb(&evals, lo);
        assert_eq!(residual.len(), 1usize << (n - k));
        let after = eval_mle_lsb(&residual, hi);
        assert_eq!(full, after);

        let eq = build_eq_table(&point);
        let dot = evals
            .iter()
            .zip(eq.iter())
            .map(|(&e, &q)| e * q)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(dot, full);
    }

    #[test]
    fn stateful_sumcheck_single_basis_roundtrip() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let f: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0),
                    0x55AA ^ i as u64,
                )
            })
            .collect();
        let b: Vec<F128> = (0..len)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0xFEDC_BA98_7654_3210),
                    0xAA55 ^ i as u64,
                )
            })
            .collect();
        let h: F128 = f
            .iter()
            .zip(b.iter())
            .map(|(&fi, &bi)| fi * bi)
            .fold(F128::ZERO, |a, v| a + v);

        let (mut prover, _first) = SumcheckProver::new(f.clone(), b.clone(), h);
        let mut ch = crate::challenger::RandomChallenger::new(0xC0FFEE);
        let mut ris: Vec<F128> = Vec::new();
        for _ in 0..(n - 1) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        assert_eq!(prover.f().len(), 2);
        assert_eq!(prover.combined_basis.len(), 2);

        let msgs = prover.transcript().to_vec();
        assert_eq!(msgs.len(), n);
        let r_last = ch.sample_f128();
        let mut t_r = h;
        for (i, msg) in msgs.iter().enumerate() {
            let quad = RoundQuad::from_msg(*msg, t_r);
            assert_eq!(
                quad.eval(F128::ZERO) + quad.eval(F128::ONE),
                t_r,
                "round {i}: q(0)+q(1) != T_r"
            );
            let r_i = if i < n - 1 { ris[i] } else { r_last };
            t_r = quad.eval(r_i);
        }
        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        let b_resid = prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(f_resid * b_resid, t_r, "residual inner product != t_r");
    }

    #[test]
    fn stateful_sumcheck_introduce_glue() {
        use crate::challenger::Challenger;
        let n = 5;
        let len = 1usize << n;
        let mk = |seed: u64| -> Vec<F128> {
            (0..len)
                .map(|i| F128::new(seed.wrapping_mul(i as u64 + 1), seed ^ (i as u64) << 7))
                .collect()
        };
        let f = mk(0xC1);
        let b1 = mk(0xB1);
        let b2 = mk(0xB2);
        let h1: F128 = f
            .iter()
            .zip(b1.iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);

        let (mut prover, _first) = SumcheckProver::new(f.clone(), b1.clone(), h1);
        let mut ch = crate::challenger::RandomChallenger::new(0xBEEF);

        let r0 = ch.sample_f128();
        prover.fold(r0);
        let mut b2_folded = b2.clone();
        partial_eval_lsb_one(&mut b2_folded, r0);
        let h2_folded: F128 = b2_folded
            .iter()
            .zip(prover.f().iter())
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |a, v| a + v);
        prover.introduce_new(b2_folded.clone(), h2_folded);
        let alpha = ch.sample_f128();
        prover.glue(alpha);

        let mut ris = vec![r0];
        for _ in 0..(n - 2) {
            let r = ch.sample_f128();
            ris.push(r);
            prover.fold(r);
        }
        let r_last = ch.sample_f128();
        ris.push(r_last);
        assert_eq!(prover.f().len(), 2);

        let msgs = prover.transcript().to_vec();
        assert_eq!(msgs.len(), 1 + 1 + 1 + (n - 2));

        let mut t_r = h1;
        let q0 = RoundQuad::from_msg(msgs[0], t_r);
        assert_eq!(q0.eval(F128::ZERO) + q0.eval(F128::ONE), t_r);
        t_r = q0.eval(r0); // fold(r0)
        let q1 = RoundQuad::from_msg(msgs[1], t_r);
        assert_eq!(q1.eval(F128::ZERO) + q1.eval(F128::ONE), t_r);
        let q_intro = RoundQuad::from_msg(msgs[2], h2_folded);
        assert_eq!(
            q_intro.eval(F128::ZERO) + q_intro.eval(F128::ONE),
            h2_folded
        );
        let combined = RoundQuad::fold(&q1, &q_intro, alpha);
        t_r += alpha * h2_folded;
        assert_eq!(combined.eval(F128::ZERO) + combined.eval(F128::ONE), t_r);
        let mut running = combined;
        assert_eq!(msgs.len(), n + 1);
        for (k, &r) in ris.iter().enumerate().skip(1).take(n - 2) {
            t_r = running.eval(r);
            let msg = msgs[2 + k]; // idx 3, 4, ...
            running = RoundQuad::from_msg(msg, t_r);
            assert_eq!(
                running.eval(F128::ZERO) + running.eval(F128::ONE),
                t_r,
                "post-glue round k={k}"
            );
        }
        t_r = running.eval(r_last);

        let one_plus_r = F128::ONE + r_last;
        let f_resid = prover.f()[0] * one_plus_r + prover.f()[1] * r_last;
        let combined_resid =
            prover.combined_basis[0] * one_plus_r + prover.combined_basis[1] * r_last;
        assert_eq!(
            f_resid * combined_resid,
            t_r,
            "residual inner product != t_r"
        );
    }

    #[test]
    fn induce_sumcheck_poly_consistent_with_codeword() {
        use crate::challenger::Challenger;
        let log_msg = 4;
        let log_inv_rate = 1;
        let msg_cols = 1usize << log_msg;
        let block_len = msg_cols << log_inv_rate;

        let mut ch = crate::challenger::RandomChallenger::new(0xF00DCAFE);
        let msg: Vec<F128> = (0..msg_cols).map(|_| ch.sample_f128()).collect();

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let mut codeword = vec![F128::ZERO; block_len];
        codeword[..msg_cols].copy_from_slice(&msg);
        ntt.forward_transform(&mut codeword);

        let num_queries = 6;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| vec![codeword[q]]).collect();
        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);

        let (basis_poly, enforced_sum) =
            induce_sumcheck_poly(log_msg, &sks_vks, &opened_rows, &[], &queries, &alpha);
        assert_eq!(basis_poly.len(), msg_cols);

        let alpha_weights: Vec<F128> = crate::lincheck::build_eq_table(&alpha)
            .into_iter()
            .take(queries.len())
            .collect();
        let expected: F128 = queries
            .iter()
            .zip(alpha_weights.iter())
            .map(|(&q, &w)| w * codeword[q])
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(enforced_sum, expected, "enforced_sum != eq(α)-batched c[q]");

        let inner: F128 = msg
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(inner, enforced_sum, "msg · basis_poly != enforced_sum");
    }

    #[test]
    fn induce_sumcheck_poly_via_ntt_matches_dense() {
        use crate::challenger::Challenger;
        let shapes = [
            (4usize, 1usize, 0usize, 6usize),
            (3, 1, 2, 5),
            (6, 2, 3, 30),
            (10, 1, 6, 218),
            (14, 2, 3, 106),
            (8, 3, 3, 71),
            (5, 5, 3, 43),
            (0, 2, 1, 3),
        ];
        for (si, &(log_msg, log_inv_rate, log_int, n_queries)) in shapes.iter().enumerate() {
            let block_len = 1usize << (log_msg + log_inv_rate);
            let num_interleaved = 1usize << log_int;
            let mut ch = crate::challenger::RandomChallenger::new(0xA11CE ^ si as u64);
            let mut queries: Vec<usize> = Vec::new();
            while queries.len() < n_queries.min(block_len) {
                let q = (ch.sample_f128().lo as usize) % block_len;
                if !queries.contains(&q) {
                    queries.push(q);
                }
            }
            let nq = queries.len();
            let opened_rows: Vec<Vec<F128>> = (0..nq)
                .map(|_| ch.sample_f128_vec(num_interleaved))
                .collect();
            let v_challenges = ch.sample_f128_vec(log_int);
            let alpha = ch.sample_f128_vec(ceil_log2(nq.max(1)));
            let sks_vks = eval_sk_at_vks(log_msg);

            let dense = induce_sumcheck_poly(
                log_msg,
                &sks_vks,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            let ntt = induce_sumcheck_poly_via_ntt(
                log_msg,
                log_inv_rate,
                &opened_rows,
                &v_challenges,
                &queries,
                &alpha,
            );
            assert_eq!(ntt.1, dense.1, "shape {si}: enforced_sum");
            assert_eq!(ntt.0, dense.0, "shape {si}: basis_poly");
        }
    }

    #[test]
    fn sparse_dual_l0_messages_and_fold2_materialization_match_dense() {
        use crate::challenger::Challenger;

        for (log_msg, log_inv_rate) in [
            (5usize, 1usize),
            (7, 1),
            (10, 1),
            (7, 2),
            (8, 3),
            (7, 4),
            (6, 5),
        ] {
            let log_d = log_msg + log_inv_rate;
            let msg_cols = 1usize << log_msg;
            let block_len = 1usize << log_d;
            let log_interleaved = 2usize;
            let num_interleaved = 1usize << log_interleaved;
            let mut ch = crate::challenger::RandomChallenger::new(
                0xD0A1_2000 ^ ((log_msg * 8 + log_inv_rate) as u64),
            );
            let poly = ch.sample_f128_vec(msg_cols * num_interleaved);
            let lane_challenges = ch.sample_f128_vec(log_interleaved);
            let f = partial_eval_lsb(&poly, &lane_challenges);

            let ntt = AdditiveNttF128::standard(log_d);
            let mut l0_codeword = vec![F128::ZERO; block_len * num_interleaved];
            ntt.rs_encode_interleaved(&poly, &mut l0_codeword, num_interleaved);

            let mut queries = vec![1usize, 2, 3, 6, 7, 8, 9, 13];
            queries.retain(|&q| q < block_len);
            let opened_rows: Vec<Vec<F128>> = queries
                .iter()
                .map(|&q| l0_codeword[q * num_interleaved..(q + 1) * num_interleaved].to_vec())
                .collect();
            let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
            let (basis, enforced) = induce_sumcheck_poly_via_ntt(
                log_msg,
                log_inv_rate,
                &opened_rows,
                &lane_challenges,
                &queries,
                &alpha,
            );
            for target_depth in 1..=4 {
                let (dual, dual_enforced) = SparseDualL0::new(
                    target_depth,
                    log_d,
                    log_inv_rate,
                    &l0_codeword,
                    num_interleaved,
                    &lane_challenges,
                    &queries,
                    &alpha,
                );
                assert_eq!(dual_enforced, enforced);
                assert_eq!(dual.round_msg(&[]), round_msg_lsb(&f, &basis));
                let rs = ch.sample_f128_vec(target_depth);
                for depth in 1..=target_depth {
                    let folded_f = partial_eval_lsb(&f, &rs[..depth]);
                    let folded_basis = partial_eval_lsb(&basis, &rs[..depth]);
                    assert_eq!(
                        dual.round_msg(&rs[..depth]),
                        round_msg_lsb(&folded_f, &folded_basis),
                        "log_msg={log_msg}: target={target_depth} depth={depth} message"
                    );
                    if depth == target_depth {
                        assert_eq!(
                            dual.materialize_after_folds(&rs[..depth]),
                            folded_basis,
                            "log_msg={log_msg}: target={target_depth} materialization"
                        );
                    }
                }
            }
            let (mut dual, dual_enforced) = SparseDualL0::new(
                2,
                log_d,
                log_inv_rate,
                &l0_codeword,
                num_interleaved,
                &lane_challenges,
                &queries,
                &alpha,
            );
            assert_eq!(dual_enforced, enforced, "log_msg={log_msg}: enforced sum");
            assert_eq!(
                dual.round_msg(&[]),
                round_msg_lsb(&f, &basis),
                "log_msg={log_msg}: introduction message"
            );

            let rs = ch.sample_f128_vec(2);
            for depth in 1..=2 {
                let folded_f = partial_eval_lsb(&f, &rs[..depth]);
                let folded_basis = partial_eval_lsb(&basis, &rs[..depth]);
                assert_eq!(
                    dual.round_msg(&rs[..depth]),
                    round_msg_lsb(&folded_f, &folded_basis),
                    "log_msg={log_msg}: depth={depth} message"
                );
                if depth == 2 {
                    assert_eq!(
                        dual.materialize_after_folds(&rs[..depth]),
                        folded_basis,
                        "log_msg={log_msg}: depth={depth} materialization"
                    );
                }
            }

            let base_basis = ch.sample_f128_vec(f.len());
            let base_sum = f
                .iter()
                .zip(&base_basis)
                .map(|(&x, &y)| x * y)
                .fold(F128::ZERO, |x, y| x + y);
            let (mut dense_prover, _) =
                SumcheckProver::new(f.clone(), base_basis.clone(), base_sum);
            let (mut dual_prover, _) = SumcheckProver::new(f.clone(), base_basis, base_sum);
            let z = ch.sample_f128_vec(log_msg);
            let (_, dense_ood_sum) = dense_prover.introduce_new_ood_factorized(&z).unwrap();
            let (_, dual_ood_sum) = dual_prover.introduce_new_ood_factorized(&z).unwrap();
            assert_eq!(dual_ood_sum, dense_ood_sum);
            let ood_beta = ch.sample_f128();
            dense_prover.glue_factorized_ood(ood_beta);
            dual_prover.glue_factorized_ood(ood_beta);

            let intro = dense_prover.introduce_new(basis.clone(), enforced);
            assert_eq!(intro, dual.round_msg(&[]));
            dual_prover.introduce_sparse_dual(intro);
            let beta = ch.sample_f128();
            dense_prover.glue_deferred_into_factorized_ood_fold(beta);
            dual_prover.glue_sparse_dual(beta, enforced);
            dual.scale(beta);

            let fold_rs = ch.sample_f128_vec(3);
            for depth in 1..=2 {
                let want = dense_prover.fold(fold_rs[depth - 1]);
                let _base = dual_prover.fold(fold_rs[depth - 1]);
                let delta = dual.round_msg(&fold_rs[..depth]);
                let got = dual_prover.add_to_last_message(delta);
                assert_eq!(got, want, "log_msg={log_msg}: state depth={depth}");
            }
            dual_prover
                .defer_folded_sparse_dual(dual.materialize_after_folds(&fold_rs[..2]), F128::ONE);
            assert_eq!(dual_prover.fold(fold_rs[2]), dense_prover.fold(fold_rs[2]));
            assert_eq!(&*dual_prover.f, &*dense_prover.f);
            assert_eq!(&*dual_prover.combined_basis, &*dense_prover.combined_basis);
            assert_eq!(dual_prover.t_r, dense_prover.t_r);
            assert_eq!(dual_prover.transcript, dense_prover.transcript);
        }
    }

    #[test]
    fn sparse_dual_depth3_4_full_state_matches_dense() {
        use crate::challenger::Challenger;

        for target_depth in 3..=4 {
            let log_msg = 7usize;
            let log_d = log_msg + 1;
            let msg_cols = 1usize << log_msg;
            let block_len = 1usize << log_d;
            let log_interleaved = 2usize;
            let num_interleaved = 1usize << log_interleaved;
            let mut ch =
                crate::challenger::RandomChallenger::new(0xD0A1_3400 ^ target_depth as u64);
            let poly = ch.sample_f128_vec(msg_cols * num_interleaved);
            let lane_challenges = ch.sample_f128_vec(log_interleaved);
            let f = partial_eval_lsb(&poly, &lane_challenges);
            let ntt = AdditiveNttF128::standard(log_d);
            let mut l0_codeword = vec![F128::ZERO; block_len * num_interleaved];
            ntt.rs_encode_interleaved(&poly, &mut l0_codeword, num_interleaved);
            let queries = vec![1usize, 2, 3, 6, 7, 8, 9, 13, 31, 32, 33];
            if target_depth == 4 {
                let mut reduced: Vec<usize> =
                    queries.iter().map(|&query| query >> target_depth).collect();
                reduced.sort_unstable();
                assert!(
                    reduced.windows(2).any(|pair| pair[0] == pair[1]),
                    "k4 oracle must exercise reduced-position collisions"
                );
            }
            let opened_rows: Vec<Vec<F128>> = queries
                .iter()
                .map(|&q| l0_codeword[q * num_interleaved..(q + 1) * num_interleaved].to_vec())
                .collect();
            let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
            let (basis, enforced) = induce_sumcheck_poly_via_ntt(
                log_msg,
                1,
                &opened_rows,
                &lane_challenges,
                &queries,
                &alpha,
            );
            let (mut dual, dual_enforced) = SparseDualL0::new(
                target_depth,
                log_d,
                1,
                &l0_codeword,
                num_interleaved,
                &lane_challenges,
                &queries,
                &alpha,
            );
            assert_eq!(dual_enforced, enforced);

            let base_basis = ch.sample_f128_vec(f.len());
            let base_sum = f
                .iter()
                .zip(&base_basis)
                .map(|(&x, &y)| x * y)
                .fold(F128::ZERO, |x, y| x + y);
            let (mut dense, _) = SumcheckProver::new(f.clone(), base_basis.clone(), base_sum);
            let (mut sparse, _) = SumcheckProver::new(f, base_basis, base_sum);
            let z = ch.sample_f128_vec(log_msg);
            let (_, dense_ood_sum) = dense.introduce_new_ood_factorized(&z).unwrap();
            let (_, sparse_ood_sum) = sparse.introduce_new_ood_factorized(&z).unwrap();
            assert_eq!(sparse_ood_sum, dense_ood_sum);
            let ood_beta = ch.sample_f128();
            dense.glue_factorized_ood(ood_beta);
            sparse.glue_factorized_ood(ood_beta);

            let intro = dense.introduce_new(basis, enforced);
            assert_eq!(intro, dual.round_msg(&[]));
            sparse.introduce_sparse_dual(intro);
            let beta = ch.sample_f128();
            dense.glue_deferred_into_factorized_ood_fold(beta);
            sparse.glue_sparse_dual(beta, enforced);
            dual.scale(beta);

            let fold_rs = ch.sample_f128_vec(target_depth);
            for depth in 1..=target_depth {
                let want = dense.fold(fold_rs[depth - 1]);
                let _base = sparse.fold(fold_rs[depth - 1]);
                let got = sparse.add_to_last_message(dual.round_msg(&fold_rs[..depth]));
                assert_eq!(got, want, "target={target_depth} depth={depth}");
            }
            sparse.merge_folded_sparse_dual(dual.materialize_after_folds(&fold_rs));
            assert_eq!(&*sparse.f, &*dense.f);
            assert_eq!(&*sparse.combined_basis, &*dense.combined_basis);
            assert_eq!(sparse.t_r, dense.t_r);
            assert_eq!(sparse.transcript, dense.transcript);
        }
    }

    #[test]
    fn sparse_dual_k4_cross_level_proof_bytes_match_dense() {
        use crate::challenger::Challenger;

        let log_n = 12usize;
        let initial_k = 2usize;
        let (p_cfg, v_cfg) = ood_test_configs(
            log_n,
            initial_k,
            &[3, 2],
            vec![0, 1, 1],
            vec![0, 0, 0],
        );
        let mut rng = crate::challenger::RandomChallenger::new(0x5A2E_D0A1_0004);
        let poly = rng.sample_f128_vec(1usize << log_n);
        let z = rng.sample_f128_vec(log_n);
        let basis = build_eq_table(&z);
        let target = poly
            .iter()
            .zip(&basis)
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |x, y| x + y);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + p_cfg.log_inv_rates[0]);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            p_cfg.log_inv_rates[0],
            &ntt_0,
            p_cfg.merkle_hash,
        );
        let initial_root = wtns_0.root();

        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));
        let mut dense_ch = crate::challenger::FsChallenger::new(b"sparse-dual-k4-proof-byte");
        let dense = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut dense_ch,
        );

        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(4));
        let mut sparse_ch = crate::challenger::FsChallenger::new(b"sparse-dual-k4-proof-byte");
        let sparse = recursive_prover_with_basis(
            &p_cfg,
            poly,
            basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut sparse_ch,
        );
        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));

        assert_eq!(sparse, dense);
        assert_eq!(
            bincode::serialize(&sparse).expect("serialize sparse proof"),
            bincode::serialize(&dense).expect("serialize dense proof")
        );
        let mut verifier_ch =
            crate::challenger::FsChallenger::new(b"sparse-dual-k4-proof-byte");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &sparse,
            &basis,
            target,
            &initial_root,
            &mut verifier_ch,
        ));
    }

    #[test]
    fn sparse_dual_deep_levels_proof_bytes_match_dense() {
        use crate::challenger::Challenger;

        let log_n = 14usize;
        let initial_k = 2usize;
        let (p_cfg, v_cfg) = ood_test_configs(
            log_n,
            initial_k,
            &[3, 3, 2],
            vec![0, 1, 1, 1],
            vec![0, 0, 0, 0],
        );
        let mut rng = crate::challenger::RandomChallenger::new(0x5A2E_D0A1_0D33);
        let poly = rng.sample_f128_vec(1usize << log_n);
        let z = rng.sample_f128_vec(log_n);
        let basis = build_eq_table(&z);
        let target = poly
            .iter()
            .zip(&basis)
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |x, y| x + y);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + p_cfg.log_inv_rates[0]);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            p_cfg.log_inv_rates[0],
            &ntt_0,
            p_cfg.merkle_hash,
        );
        let initial_root = wtns_0.root();

        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));
        SPARSE_DUAL_DEEP_TEST_DEPTH.with(|value| value.set(0));
        let mut dense_ch = crate::challenger::FsChallenger::new(b"sparse-dual-deep-proof-byte");
        let dense = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut dense_ch,
        );

        SPARSE_DUAL_DEEP_TEST_DEPTH.with(|value| value.set(4));
        let mut deep_ch = crate::challenger::FsChallenger::new(b"sparse-dual-deep-proof-byte");
        let deep_only = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut deep_ch,
        );

        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(4));
        let mut sparse_ch = crate::challenger::FsChallenger::new(b"sparse-dual-deep-proof-byte");
        let sparse = recursive_prover_with_basis(
            &p_cfg,
            poly,
            basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut sparse_ch,
        );
        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));
        SPARSE_DUAL_DEEP_TEST_DEPTH.with(|value| value.set(0));

        assert_eq!(deep_only, dense, "deep-only sparse arm");
        assert_eq!(sparse, dense, "L0 + deep sparse arm");
        assert_eq!(
            bincode::serialize(&sparse).expect("serialize sparse proof"),
            bincode::serialize(&dense).expect("serialize dense proof")
        );
        let mut verifier_ch =
            crate::challenger::FsChallenger::new(b"sparse-dual-deep-proof-byte");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &sparse,
            &basis,
            target,
            &initial_root,
            &mut verifier_ch,
        ));
    }

    #[test]
    fn sparse_dual_exact_ranked_selector_positive_kill_and_negative() {
        let mut config = prover_config_for(25, 6, LigeritoProfile::Fast)
            .expect("exact ranked m32 Fast config");
        config.merkle_hash = HashKind::Blake3;
        let select = |cfg: &ProverConfig, sparse_disabled: bool, defer_disabled: bool| {
            ranked_sparse_dual_l0_depth_selected(
                cfg,
                25,
                19,
                1usize << 19,
                218,
                1,
                true,
                true,
                true,
                sparse_disabled,
                defer_disabled,
                4,
            )
        };
        assert_eq!(select(&config, false, false), Some(4));
        assert_eq!(select(&config, true, false), None, "sparse kill must restore dense");
        assert_eq!(select(&config, false, true), None, "defer kill must restore dense");

        let mut wrong_queries = config.clone();
        wrong_queries.queries[0] = 217;
        assert_eq!(select(&wrong_queries, false, false), None);
        let mut wrong_shape = config.clone();
        wrong_shape.recursive_ks[0] = 4;
        assert_eq!(select(&wrong_shape, false, false), None);
    }

    #[test]
    fn sparse_dual_ranked_depth2_default_rollback_and_explicit_priority() {
        assert_eq!(ranked_sparse_dual_l0_depth_setting(None, false), 2);
        assert_eq!(ranked_sparse_dual_l0_depth_setting(None, true), 4);
        for depth in 1..=4 {
            let value = depth.to_string();
            assert_eq!(
                ranked_sparse_dual_l0_depth_setting(Some(&value), false),
                depth
            );
            assert_eq!(
                ranked_sparse_dual_l0_depth_setting(Some(&value), true),
                depth,
                "explicit depth must override the depth2 rollback"
            );
        }
    }

    #[test]
    fn sparse_dual_deep_selector_covers_every_ranked_level() {
        let mut config = prover_config_for(25, 6, LigeritoProfile::Fast)
            .expect("exact ranked m32 Fast config");
        config.merkle_hash = HashKind::Blake3;
        let ladder: [(usize, usize, usize); 5] = [
            (19, 218, 1),
            (16, 106, 2),
            (13, 71, 3),
            (10, 53, 4),
            (7, 43, 5),
        ];
        for &(n_level, n_queries, rate) in &ladder {
            let select = |sparse_disabled: bool, defer_disabled: bool| {
                ranked_sparse_dual_l0_depth_selected(
                    &config,
                    25,
                    n_level,
                    1usize << n_level,
                    n_queries,
                    rate,
                    true,
                    true,
                    true,
                    sparse_disabled,
                    defer_disabled,
                    4,
                )
            };
            assert_eq!(select(false, false), Some(4), "level n={n_level}");
            assert_eq!(select(true, false), None, "sparse kill at n={n_level}");
            assert_eq!(select(false, true), None, "defer kill at n={n_level}");
            assert_eq!(
                ranked_sparse_dual_l0_depth_selected(
                    &config,
                    25,
                    n_level,
                    1usize << n_level,
                    n_queries,
                    rate + 1,
                    true,
                    true,
                    true,
                    false,
                    false,
                    4,
                ),
                None,
                "wrong rate at n={n_level}"
            );
        }
        assert_eq!(ranked_sparse_dual_deep_depth_for_level(19), None);
        let mut materializations = vec![SPARSE_DUAL_MAX_DEPTH - 1];
        for (level, n_level) in [16usize, 13, 10, 7].iter().copied().enumerate() {
            let depth = ranked_sparse_dual_deep_depth_for_level(n_level)
                .unwrap_or_else(|| panic!("no depth for ranked level n={n_level}"));
            assert!((2..=SPARSE_DUAL_MAX_DEPTH).contains(&depth));
            let remaining = 3 * (4 - level);
            assert!(depth <= remaining, "n={n_level}: depth {depth} > {remaining}");
            materializations.push(3 * (level + 1) + depth - 1);
        }
        let mut sorted = materializations.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            materializations.len(),
            "materialization folds collide: {materializations:?}"
        );
    }

    #[test]
    #[ignore = "~2 GiB exact-ranked oracle; run alone in release mode"]
    fn sparse_dual_exact_ranked_post_direct_fold8_messages_state_and_timings() {
        use crate::challenger::Challenger;
        use rayon::prelude::*;
        use std::hint::black_box;
        use std::time::Instant;

        let mut config = prover_config_for(25, 6, LigeritoProfile::Fast)
            .expect("exact ranked m32 Fast config");
        config.merkle_hash = HashKind::Blake3;
        assert_eq!(
            ranked_sparse_dual_l0_depth_selected(
                &config,
                25,
                19,
                1usize << 19,
                218,
                1,
                true,
                true,
                true,
                false,
                false,
                4,
            ),
            Some(4),
            "exact production selector must reach the sparse arm"
        );
        assert_eq!(
            ranked_sparse_dual_l0_depth_selected(
                &config,
                25,
                19,
                1usize << 19,
                218,
                1,
                true,
                true,
                true,
                true,
                false,
                4,
            ),
            None,
            "kill switch must restore the dense arm"
        );

        const LOG_N: usize = 25;
        const LANE_LOG: usize = 6;
        const LANES: usize = 1 << LANE_LOG;
        const LOG_MSG: usize = LOG_N - LANE_LOG;
        const LOG_D: usize = LOG_MSG + 1;
        const Q: usize = 218;
        let mut rng = crate::challenger::RandomChallenger::new(0xD12E_C7F8_0218_0064);
        let poly = rng.sample_f128_vec(1usize << LOG_N);
        let lane_challenges = rng.sample_f128_vec(LANE_LOG);
        let f = partial_eval_lsb(&poly, &lane_challenges);
        let ntt = AdditiveNttF128::standard(LOG_D);
        let mut l0_codeword = vec![F128::ZERO; (1usize << LOG_D) * LANES];
        let encode_start = Instant::now();
        ntt.rs_encode_interleaved(&poly, &mut l0_codeword, LANES);
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1e3;
        let mut queries: Vec<usize> = vec![1, 2];
        queries.extend(
            (0..Q - 2).map(|index| (index * 4093 + 17) & ((1usize << LOG_D) - 1)),
        );
        assert_eq!(queries.len(), Q);
        let mut reduced_k4: Vec<usize> = queries.iter().map(|&query| query >> 4).collect();
        reduced_k4.sort_unstable();
        assert!(
            reduced_k4.windows(2).any(|pair| pair[0] == pair[1]),
            "exact k4 oracle must exercise reduced-position collisions"
        );
        let opened_rows: Vec<Vec<F128>> = queries
            .iter()
            .map(|&query| l0_codeword[query * LANES..(query + 1) * LANES].to_vec())
            .collect();
        let alpha = rng.sample_f128_vec(ceil_log2(Q));

        let dense_start = Instant::now();
        let (dense_basis, enforced_sum) = induce_sumcheck_poly_via_ntt(
            LOG_MSG,
            1,
            &opened_rows,
            &lane_challenges,
            &queries,
            &alpha,
        );
        let dense_induce_ms = dense_start.elapsed().as_secs_f64() * 1e3;
        const INTRO_REPEATS: usize = 100;
        let dense_intro_start = Instant::now();
        for _ in 0..INTRO_REPEATS {
            black_box(round_msg_lsb(black_box(&f), black_box(&dense_basis)));
        }
        let dense_intro_ms =
            dense_intro_start.elapsed().as_secs_f64() * 1e3 / INTRO_REPEATS as f64;
        let all_fold_challenges = rng.sample_f128_vec(5);

        let mut depth4_dual = None;
        for depth in 2..=4 {
            let construct_start = Instant::now();
            let (dual, sparse_enforced) = SparseDualL0::new(
                depth,
                LOG_D,
                1,
                &l0_codeword,
                LANES,
                &lane_challenges,
                &queries,
                &alpha,
            );
            let construct_ms = construct_start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(sparse_enforced, enforced_sum);
            for prior in 0..=depth {
                assert_eq!(
                    dual.round_msg(&all_fold_challenges[..prior]),
                    round_msg_lsb(
                        &partial_eval_lsb(&f, &all_fold_challenges[..prior]),
                        &partial_eval_lsb(&dense_basis, &all_fold_challenges[..prior]),
                    ),
                    "exact ranked depth={depth} prior={prior} message"
                );
            }
            const MESSAGE_REPEATS: usize = 100;
            let messages_start = Instant::now();
            for _ in 0..MESSAGE_REPEATS {
                for prior in 0..=depth {
                    black_box(dual.round_msg(black_box(&all_fold_challenges[..prior])));
                }
            }
            let messages_ms = messages_start.elapsed().as_secs_f64() * 1e3
                / MESSAGE_REPEATS as f64;
            let material_start = Instant::now();
            let materialized = dual.materialize_after_folds(&all_fold_challenges[..depth]);
            let materialize_ms = material_start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                materialized,
                partial_eval_lsb(&dense_basis, &all_fold_challenges[..depth]),
                "exact ranked depth={depth} materialization"
            );
            let merge_ms = if depth == 3 {
                const MERGE_REPEATS: usize = 100;
                let mut dst = vec![F128::ZERO; materialized.len()];
                let merge_start = Instant::now();
                for _ in 0..MERGE_REPEATS {
                    dst.par_iter_mut()
                        .zip(materialized.par_iter())
                        .for_each(|(value, &added)| *value += added);
                    black_box(&dst);
                }
                merge_start.elapsed().as_secs_f64() * 1e3 / MERGE_REPEATS as f64
            } else {
                0.0
            };
            eprintln!(
                "[sparse-dual-ranked] k={depth} construct={construct_ms:.6} ms messages={messages_ms:.6} ms materialize={materialize_ms:.6} ms seam_merge={merge_ms:.6} ms replacement={:.6} ms",
                construct_ms + messages_ms + materialize_ms + merge_ms,
            );
            if depth == 4 {
                depth4_dual = Some(dual);
            }
        }

        let mut dual = depth4_dual.expect("k4 dual retained");
        let base_basis = rng.sample_f128_vec(f.len());
        let base_sum = f
            .iter()
            .zip(&base_basis)
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |x, y| x + y);
        let (mut dense, _) = SumcheckProver::new(f.clone(), base_basis.clone(), base_sum);
        let (mut sparse, _) = SumcheckProver::new(f, base_basis, base_sum);

        let z_l1 = rng.sample_f128_vec(LOG_MSG);
        let (_, dense_ood_sum) = dense.introduce_new_ood_factorized(&z_l1).unwrap();
        let (_, sparse_ood_sum) = sparse.introduce_new_ood_factorized(&z_l1).unwrap();
        assert_eq!(sparse_ood_sum, dense_ood_sum);
        let ood_beta_l1 = rng.sample_f128();
        dense.glue_factorized_ood(ood_beta_l1);
        sparse.glue_factorized_ood(ood_beta_l1);
        let dense_intro = dense.introduce_new(dense_basis, enforced_sum);
        let sparse_intro = dual.round_msg(&[]);
        assert_eq!(sparse_intro, dense_intro);
        sparse.introduce_sparse_dual(sparse_intro);
        let dual_beta = rng.sample_f128();
        dense.glue_deferred_into_factorized_ood_fold(dual_beta);
        sparse.glue_sparse_dual(dual_beta, enforced_sum);
        dual.scale(dual_beta);

        for depth in 1..=3 {
            let want = dense.fold(all_fold_challenges[depth - 1]);
            let _base = sparse.fold(all_fold_challenges[depth - 1]);
            let got = sparse.add_to_last_message(dual.round_msg(&all_fold_challenges[..depth]));
            assert_eq!(got, want, "exact ranked pre-boundary fold {depth}");
        }

        let z_l2 = rng.sample_f128_vec(LOG_MSG - 3);
        let (dense_ood_msg, dense_ood_sum) = dense.introduce_new_ood_factorized(&z_l2).unwrap();
        let (sparse_ood_msg, sparse_ood_sum) = sparse.introduce_new_ood_factorized(&z_l2).unwrap();
        assert_eq!((sparse_ood_msg, sparse_ood_sum), (dense_ood_msg, dense_ood_sum));
        let ood_beta_l2 = rng.sample_f128();
        dense.glue_factorized_ood(ood_beta_l2);
        sparse.glue_factorized_ood(ood_beta_l2);
        let b1 = rng.sample_f128_vec(1usize << (LOG_MSG - 3));
        let h1 = dense
            .f()
            .iter()
            .zip(&b1)
            .map(|(&x, &y)| x * y)
            .fold(F128::ZERO, |x, y| x + y);
        assert_eq!(dense.introduce_new(b1.clone(), h1), sparse.introduce_new(b1, h1));
        let beta1 = rng.sample_f128();
        dense.glue_deferred_into_factorized_ood_fold(beta1);
        sparse.glue_deferred_into_factorized_ood_fold(beta1);

        let want4 = dense.fold(all_fold_challenges[3]);
        let _base4 = sparse.fold(all_fold_challenges[3]);
        let got4 = sparse.add_to_last_message(dual.round_msg(&all_fold_challenges[..4]));
        assert_eq!(got4, want4);
        sparse.defer_folded_sparse_dual(
            dual.materialize_after_folds(&all_fold_challenges[..4]),
            F128::ONE,
        );
        assert_eq!(sparse.fold(all_fold_challenges[4]), dense.fold(all_fold_challenges[4]));
        assert_eq!(&*sparse.f, &*dense.f);
        assert_eq!(&*sparse.combined_basis, &*dense.combined_basis);
        assert_eq!(sparse.t_r, dense.t_r);
        assert_eq!(sparse.transcript, dense.transcript);
        eprintln!(
            "[sparse-dual-ranked] encode={encode_ms:.3} ms dense_induce={dense_induce_ms:.6} ms dense_intro={dense_intro_ms:.6} ms dense_removable={:.6} ms",
            dense_induce_ms + dense_intro_ms,
        );
    }

    #[test]
    #[ignore]
    fn sparse_dual_union_fold1_fold5_interaction_timings() {
        use crate::challenger::Challenger;
        use std::hint::black_box;
        use std::time::Instant;

        let mut rng = crate::challenger::RandomChallenger::new(0xD12E_F015_0005);
        let r1 = rng.sample_f128();
        let r5 = rng.sample_f128();
        let gamma = rng.sample_f128();
        let alpha = rng.sample_f128();

        const N1: usize = 1 << 19;
        const EQ_LO: usize = 2048;
        let f1 = rng.sample_f128_vec(N1);
        let b1 = rng.sample_f128_vec(N1);
        let d1 = rng.sample_f128_vec(N1);
        let eq_lo = rng.sample_f128_vec(EQ_LO);
        let eq_hi = rng.sample_f128_vec((N1 / 2) / EQ_LO);

        const N5: usize = 1 << 15;
        let f5 = rng.sample_f128_vec(N5);
        let b5 = rng.sample_f128_vec(N5);
        let d5 = rng.sample_f128_vec(N5);

        let fold1 = |basis: Option<(&[F128], F128)>| {
            black_box(fold_and_msg_lsb_inner(
                black_box(&f1),
                black_box(&b1),
                black_box(r1),
                None,
                Some((black_box(&eq_lo), black_box(&eq_hi), black_box(gamma))),
                basis,
            ))
        };
        let fold5 = |basis: Option<(&[F128], F128)>| {
            black_box(fold_and_msg_lsb_inner(
                black_box(&f5),
                black_box(&b5),
                black_box(r5),
                None,
                None,
                basis,
            ))
        };

        drop(fold1(None));
        drop(fold1(Some((&d1, alpha))));
        drop(fold5(None));
        drop(fold5(Some((&d5, alpha))));

        fn sample_ms(mut f: impl FnMut(), repeats: usize) -> f64 {
            let start = Instant::now();
            for _ in 0..repeats {
                f();
            }
            start.elapsed().as_secs_f64() * 1e3 / repeats as f64
        }

        const FOLD1_REPEATS: usize = 24;
        const FOLD5_REPEATS: usize = 160;
        let dense_fold1_ms = sample_ms(
            || drop(fold1(Some((black_box(&d1), black_box(alpha))))),
            FOLD1_REPEATS,
        );
        let sparse_fold1_ms = sample_ms(|| drop(fold1(None)), FOLD1_REPEATS);
        let dense_fold5_ms = sample_ms(|| drop(fold5(None)), FOLD5_REPEATS);
        let sparse_fold5_ms = sample_ms(
            || drop(fold5(Some((black_box(&d5), black_box(alpha))))),
            FOLD5_REPEATS,
        );
        let fold1_saved_ms = dense_fold1_ms - sparse_fold1_ms;
        let fold5_tax_ms = sparse_fold5_ms - dense_fold5_ms;
        eprintln!(
            "[sparse-dual-union-seams] dense_fold1={dense_fold1_ms:.6} ms sparse_fold1={sparse_fold1_ms:.6} ms fold1_saved={fold1_saved_ms:.6} ms dense_fold5={dense_fold5_ms:.6} ms sparse_fold5={sparse_fold5_ms:.6} ms fold5_tax={fold5_tax_ms:.6} ms net={:.6} ms",
            fold1_saved_ms - fold5_tax_ms,
        );
    }

    #[test]
    #[ignore]
    fn sparse_dual_round_msg_serial_vs_chunked_m4() {
        use crate::challenger::Challenger;
        use std::hint::black_box;
        use std::time::Instant;

        let depth = 4usize;
        let log_d = 20usize;
        let query_count = 218usize;
        let mut rng = crate::challenger::RandomChallenger::new(0x5E21_A1C4_0218);
        let queries: Vec<usize> = (0..query_count)
            .map(|index| (index * 4093 + 17) & ((1usize << log_d) - 1))
            .collect();
        let alpha_pows = rng.sample_f128_vec(query_count);
        let inverse_local_blocks = (0..query_count)
            .map(|_| {
                let values = rng.sample_f128_vec(32);
                let mut block = [F128::ZERO; 32];
                block.copy_from_slice(&values);
                block
            })
            .collect();
        let dual = SparseDualL0 {
            ntt: AdditiveNttF128::standard(log_d),
            log_d,
            log_inv_rate: 1,
            depth,
            cache_len: 32,
            queries,
            alpha_pows,
            w0_pows: Vec::new(),
            w2_pows: Vec::new(),
            folded_k: 0,
            inverse_local_blocks,
            rows: Vec::new(),
            round0_msg: SumcheckMessage {
                u_0: F128::ZERO,
                u_2: F128::ZERO,
            },
            snapshots: false,
            lane_x4: true,
        };
        let challenges = rng.sample_f128_vec(depth);
        for k in 0..=depth {
            assert_eq!(
                dual.round_msg(&challenges[..k]),
                dual.round_msg_chunked(&challenges[..k])
            );
        }

        const REPEATS: usize = 200;
        let serial_start = Instant::now();
        for _ in 0..REPEATS {
            for k in 0..=depth {
                black_box(dual.round_msg(black_box(&challenges[..k])));
            }
        }
        let serial = serial_start.elapsed();
        let chunked_start = Instant::now();
        for _ in 0..REPEATS {
            for k in 0..=depth {
                black_box(dual.round_msg_chunked(black_box(&challenges[..k])));
            }
        }
        let chunked = chunked_start.elapsed();
        eprintln!(
            "[sparse-dual-msg] serial={:.6} ms/proof chunked={:.6} ms/proof ratio={:.3}",
            serial.as_secs_f64() * 1e3 / REPEATS as f64,
            chunked.as_secs_f64() * 1e3 / REPEATS as f64,
            chunked.as_secs_f64() / serial.as_secs_f64(),
        );
    }

    #[test]
    fn transpose_sparse_matches_dense() {
        use crate::challenger::Challenger;
        for &log_d in &[6usize, 11, 12, 14, 16, 18] {
            for &nq in &[1usize, 5, 43, 218] {
                let n = 1usize << log_d;
                let nq = nq.min(n);
                let mut ch =
                    crate::challenger::RandomChallenger::new(0xC0DE ^ (log_d * 131 + nq) as u64);
                let ntt = AdditiveNttF128::standard(log_d);
                let mut positions: Vec<usize> = Vec::new();
                let mut values: Vec<F128> = Vec::new();
                while positions.len() < nq {
                    let p = (ch.sample_f128().lo as usize) % n;
                    if !positions.contains(&p) {
                        positions.push(p);
                        values.push(ch.sample_f128());
                    }
                }
                let mut dense = vec![F128::ZERO; n];
                for (&p, &v) in positions.iter().zip(&values) {
                    dense[p] += v;
                }
                transpose_forward_ntt(&ntt, &mut dense, log_d);
                let sparse = transpose_forward_ntt_sparse(&ntt, &positions, &values, log_d);
                assert_eq!(sparse, dense, "log_d={log_d}, nq={nq}");
            }
        }
    }

    #[test]
    fn singleton_window_matches_dense_for_every_position() {
        use crate::challenger::Challenger;
        const K: usize = 8;
        for &log_d in &[12usize, 18, 20] {
            let ntt = AdditiveNttF128::standard(log_d);
            let last_w = (1usize << (log_d - K)) - 1;
            for &w in &[0usize, 1, last_w] {
                let mut ch = crate::challenger::RandomChallenger::new(
                    0x51A6_1E70 ^ ((log_d * 257 + w) as u64),
                );
                for p in 0..(1usize << K) {
                    let value = if p == 0 { F128::ONE } else { ch.sample_f128() };
                    let mut dense = vec![F128::ZERO; 1usize << K];
                    dense[p] = value;
                    let want = transpose_forward_ntt_window_dense(&ntt, log_d, K, w, dense);
                    let got = transpose_forward_ntt_window_singleton(&ntt, log_d, K, w, p, value);
                    assert_eq!(got, want, "log_d={log_d} w={w} p={p}");
                }
            }
        }
    }

    fn sparse_dense_oracle(log_d: usize, positions: &[usize], values: &[F128]) -> Vec<F128> {
        let mut dense = vec![F128::ZERO; 1usize << log_d];
        for (&p, &v) in positions.iter().zip(values) {
            dense[p] += v;
        }
        let ntt = AdditiveNttF128::standard(log_d);
        transpose_forward_ntt(&ntt, &mut dense, log_d);
        dense
    }

    #[test]
    fn singleton_sparse_collision_falls_back_exactly() {
        use crate::challenger::Challenger;
        let log_d = 12usize;
        let w = 3usize;
        let positions = vec![(w << 8) + 1, (w << 8) + 7, (w << 8) + 201];
        let mut ch = crate::challenger::RandomChallenger::new(0xC011_1510);
        let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let want = sparse_dense_oracle(log_d, &positions, &values);
        let (got, stats) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(true),
            Some(8),
        );
        let (off, _) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(false),
            Some(8),
        );
        assert_eq!(got, want);
        assert_eq!(off, want);
        assert_eq!(stats.singleton_hits, 0);
        assert_eq!(stats.multi_hits, 1);
        assert_eq!(stats.collision_fallbacks, 0);
    }

    #[test]
    fn singleton_sparse_mixed_windows_match_dense() {
        use crate::challenger::Challenger;
        let log_d = 12usize;
        let positions = vec![
            (1usize << 8) + 2,
            (1usize << 8) + 91,
            (5usize << 8) + 17,
            (11usize << 8) + 233,
        ];
        let mut ch = crate::challenger::RandomChallenger::new(0x51A6_C011);
        let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_d);
        let want = sparse_dense_oracle(log_d, &positions, &values);
        let (got, stats) = transpose_forward_ntt_sparse_inner(
            &ntt,
            &positions,
            &values,
            log_d,
            Some(true),
            Some(8),
        );
        assert_eq!(got, want);
        assert_eq!(stats.singleton_hits, 2);
        assert_eq!(stats.multi_hits, 1);
        assert_eq!(stats.collision_fallbacks, 0);
    }

    #[test]
    fn induce_window_k_selects_documented_optima() {
        assert_eq!(induce_window_k(20, 218, 16, true), 12);
        assert_eq!(induce_window_k(18, 106, 16, true), 11);
        assert_eq!(induce_window_k(20, 218, 16, false), 8);
        assert_eq!(induce_window_k(18, 106, 16, false), 8);
        assert_eq!(induce_window_k(12, 218, 16, false), 8);
        assert_eq!(induce_window_k(11, 218, 16, true), 0);
        assert_eq!(induce_window_k(20, 0, 16, true), 0);
        assert_eq!(induce_window_k(20, 218, 256, true), 10);
        assert_eq!(induce_window_k(12, 106, 16, true), 8);
        assert_eq!(induce_window_k(20, 218, 4096, true), 8);
        assert_eq!(induce_window_k(20, 1, 1, true), 12);
        assert_eq!(induce_window_k(20, 100_000, 16, true), 8);
    }

    #[test]
    fn multi_singleton_matches_dense_across_threshold() {
        use crate::challenger::Challenger;
        let log_d = 14usize;
        for &k in &[8usize, 10, 12] {
            let ntt = AdditiveNttF128::standard(log_d);
            let wlen = 1usize << k;
            let last_w = (1usize << (log_d - k)) - 1;
            for &w in &[0usize, last_w] {
                let mut ch =
                    crate::challenger::RandomChallenger::new(0x0117_15EE ^ ((k * 8191 + w) as u64));
                for nnz in 1..=(k / 2 + 1) {
                    let mut ps: Vec<usize> = vec![0, wlen - 1];
                    ps.truncate(nnz);
                    while ps.len() < nnz {
                        let p = (ch.sample_f128().lo as usize) % wlen;
                        if !ps.contains(&p) {
                            ps.push(p);
                        }
                    }
                    let vs: Vec<F128> = ps.iter().map(|_| ch.sample_f128()).collect();
                    let mut dense = vec![F128::ZERO; wlen];
                    for (&p, &v) in ps.iter().zip(&vs) {
                        dense[p] += v;
                    }
                    let (_, want) = transpose_forward_ntt_window_dense(&ntt, log_d, k, w, dense);
                    let mut buf = take_singleton_buf(k);
                    expand_singleton_into(&ntt, log_d, k, w, ps[0], vs[0], &mut buf);
                    let mut scratch = take_singleton_buf(k);
                    for i in 1..nnz {
                        expand_singleton_into(&ntt, log_d, k, w, ps[i], vs[i], &mut scratch);
                        for (d, &s) in buf.iter_mut().zip(scratch.iter()) {
                            *d += s;
                        }
                    }
                    assert_eq!(buf, want, "k={k} w={w} nnz={nnz}");
                }
            }
        }
    }

    #[test]
    fn sparse_transpose_matches_dense_at_every_forced_k() {
        use crate::challenger::Challenger;
        for &log_d in &[12usize, 16, 18] {
            let n = 1usize << log_d;
            let ntt = AdditiveNttF128::standard(log_d);
            let mut ch = crate::challenger::RandomChallenger::new(0x7357_0A11 ^ log_d as u64);
            let mut positions: Vec<usize> = Vec::new();
            while positions.len() < 80.min(n / 4) {
                let p = (ch.sample_f128().lo as usize) % n;
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
            let values: Vec<F128> = positions.iter().map(|_| ch.sample_f128()).collect();
            let want = sparse_dense_oracle(log_d, &positions, &values);
            for k in 8..=12.min(log_d) {
                let (got, _) = transpose_forward_ntt_sparse_inner(
                    &ntt,
                    &positions,
                    &values,
                    log_d,
                    None,
                    Some(k),
                );
                assert_eq!(got, want, "log_d={log_d} k={k}");
            }
        }
    }

    #[test]
    fn densify_windows_fused_matches_two_pass() {
        use crate::challenger::Challenger;
        for &(log_d, k) in &[
            (12usize, 8usize),
            (16, 8),
            (20, 8),
            (13, 8),
            (20, 12),
            (18, 11),
            (16, 10),
        ] {
            let n = 1usize << log_d;
            let nwin = n >> k;
            for &active in &[0usize, 1, 214.min(nwin), nwin] {
                let mut ch = crate::challenger::RandomChallenger::new(
                    0x0DE5_1F17 ^ ((log_d * 8191 + active) as u64),
                );
                let mut processed: Vec<(usize, Vec<F128>)> = Vec::new();
                let mut seen = vec![false; nwin];
                while processed.len() < active {
                    let w = (ch.sample_f128().lo as usize) % nwin;
                    if seen[w] {
                        continue;
                    }
                    seen[w] = true;
                    processed.push((w, (0..(1usize << k)).map(|_| ch.sample_f128()).collect()));
                }
                let mut want = vec![F128::ZERO; n];
                for (w, buf) in &processed {
                    want[(w << k)..((w + 1) << k)].copy_from_slice(buf);
                }
                let got = densify_windows_fused(n, k, window_slots(nwin, processed));
                assert_eq!(got.len(), want.len(), "log_d={log_d} active={active}");
                assert!(got == want, "log_d={log_d} k={k} active={active}");
            }
        }
    }

    #[test]
    fn eq_split_matches_serial() {
        use crate::challenger::Challenger;
        for d in 0..=20usize {
            let mut ch = crate::challenger::RandomChallenger::new(0x3EE7_u64 ^ d as u64);
            let point: Vec<F128> = (0..d).map(|_| ch.sample_f128()).collect();
            assert_eq!(
                build_eq_table_split(&point),
                build_eq_table(&point),
                "d={d}"
            );
        }
    }

    #[test]
    fn factorized_ood_intro_matches_dense() {
        use crate::challenger::Challenger;
        for &d in &[4usize, 12, 19] {
            let mut ch = crate::challenger::RandomChallenger::new(0xFA17_00D0 ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..(1usize << d)).map(|_| ch.sample_f128()).collect();
            let dense = build_eq_table(&z);
            let want = round_msg_and_eval_lsb(&f, &dense);
            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let eq_lo = build_eq_table(&z[1..1 + split]);
            let eq_hi = build_eq_table(&z[1 + split..]);
            let got = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_eq!(got, want, "factorized OOD introduction d={d}");
        }
    }

    #[test]
    fn lazy_ood_corrected_fold_matches_dense_glue() {
        use crate::challenger::Challenger;
        let d = 13usize;
        let n = 1usize << d;
        let mut ch = crate::challenger::RandomChallenger::new(0x1A2E_00D5);
        let z = ch.sample_f128_vec(d);
        let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let beta = ch.sample_f128();
        let r = ch.sample_f128();

        let dense_eq = build_eq_table(&z);
        let mut dense_basis = b.clone();
        crate::field::f128_slice::add_scaled(&mut dense_basis, &dense_eq, beta);
        let (want_f, want_b, want_msg) = fold_and_msg_lsb(&f, &dense_basis, r, None);

        let eq_lo = build_eq_table(&z[1..12]);
        let eq_hi = build_eq_table(&z[12..]);
        assert_eq!(eq_lo.len(), 2048);
        let gamma = beta * (F128::ONE + z[0] + r);
        let (got_f, got_b, got_msg) =
            fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
        assert_eq!(&*got_f, &*want_f);
        assert_eq!(&*got_b, &*want_b);
        assert_eq!(got_msg, want_msg);
    }

    #[test]
    fn lazy_ood_state_machine_matches_dense_across_ordinary_glue() {
        use crate::challenger::Challenger;
        let d = 13usize;
        let n = 1usize << d;
        let mut ch = crate::challenger::RandomChallenger::new(0x57A7_E00D);
        let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let z = ch.sample_f128_vec(d);
        let ordinary: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
        let ordinary_claim = ch.sample_f128();
        let beta = ch.sample_f128();
        let alpha = ch.sample_f128();
        let r = ch.sample_f128();
        let target = ch.sample_f128();

        let (mut dense, dense_start) = SumcheckProver::new(f.clone(), b.clone(), target);
        let (mut lazy, lazy_start) = SumcheckProver::new(f.clone(), b.clone(), target);
        let (mut deferred, deferred_start) = SumcheckProver::new(f, b, target);
        assert_eq!(lazy_start, dense_start);
        assert_eq!(deferred_start, dense_start);

        let (dense_intro, dense_y) = dense.introduce_new_with_eval(build_eq_table(&z));
        let (lazy_intro, lazy_y) = lazy.introduce_new_ood_factorized(&z).unwrap();
        let (deferred_intro, deferred_y) = deferred.introduce_new_ood_factorized(&z).unwrap();
        assert_eq!((lazy_intro, lazy_y), (dense_intro, dense_y));
        assert_eq!((deferred_intro, deferred_y), (dense_intro, dense_y));
        dense.glue(beta);
        lazy.glue_factorized_ood(beta);
        deferred.glue_factorized_ood(beta);

        let dense_ordinary = dense.introduce_new(ordinary.clone(), ordinary_claim);
        let lazy_ordinary = lazy.introduce_new(ordinary.clone(), ordinary_claim);
        let deferred_ordinary = deferred.introduce_new(ordinary, ordinary_claim);
        assert_eq!(lazy_ordinary, dense_ordinary);
        assert_eq!(deferred_ordinary, dense_ordinary);
        dense.glue(alpha);
        lazy.glue(alpha);
        deferred.glue_deferred_into_factorized_ood_fold(alpha);
        assert!(deferred.pending_fold_basis.is_some());

        let dense_msg = dense.fold(r);
        assert_eq!(lazy.fold(r), dense_msg);
        assert_eq!(deferred.fold(r), dense_msg);
        assert!(deferred.pending_fold_basis.is_none());
        assert_eq!(&*lazy.f, &*dense.f);
        assert_eq!(&*lazy.combined_basis, &*dense.combined_basis);
        assert_eq!(lazy.t_r, dense.t_r);
        assert_eq!(lazy.transcript, dense.transcript);
        assert_eq!(&*deferred.f, &*dense.f);
        assert_eq!(&*deferred.combined_basis, &*dense.combined_basis);
        assert_eq!(deferred.t_r, dense.t_r);
        assert_eq!(deferred.transcript, dense.transcript);
    }

    #[test]
    fn ranked_lazy_ood_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (config, _) = security.to_prover_verifier_configs().unwrap();
        let selected = |log_n, n1, count, len, direct, platform, disabled| {
            ranked_l1_lazy_ood_eq_selected(
                &config, log_n, n1, count, len, direct, platform, disabled,
            )
        };
        assert!(selected(25, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(24, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 18, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 19, 2, 1 << 19, true, true, false));
        assert!(!selected(25, 19, 1, 1 << 18, true, true, false));
        assert!(!selected(25, 19, 1, 1 << 19, false, true, false));
        assert!(!selected(25, 19, 1, 1 << 19, true, false, false));
        assert!(!selected(25, 19, 1, 1 << 19, true, true, true));
    }

    #[test]
    fn ranked_deferred_induced_glue_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (mut config, _) = security.to_prover_verifier_configs().unwrap();
        config.merkle_hash = HashKind::Blake3;
        let selected = |config: &ProverConfig,
                        log_n,
                        n_level,
                        len,
                        queries,
                        rate,
                        lazy,
                        direct8,
                        platform,
                        disabled| {
            ranked_deferred_induced_glue_selected(
                config, log_n, n_level, len, queries, rate, lazy, direct8, platform, disabled,
            )
        };
        for &(n_level, queries, rate) in &[
            (19, 218, 1),
            (16, 106, 2),
            (13, 71, 3),
            (10, 53, 4),
            (7, 43, 5),
        ] {
            assert!(selected(
                &config,
                25,
                n_level,
                1 << n_level,
                queries,
                rate,
                true,
                true,
                true,
                false,
            ));
        }
        assert!(!selected(
            &config,
            25,
            4,
            1 << 4,
            36,
            6,
            true,
            true,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            false,
            true,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            false,
            true,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            false,
            false,
        ));
        assert!(!selected(
            &config,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            true,
            true,
        ));
        let mut wrong_ladder = config.clone();
        wrong_ladder.queries[1] += 1;
        assert!(!selected(
            &wrong_ladder,
            25,
            19,
            1 << 19,
            218,
            1,
            true,
            true,
            true,
            false,
        ));
    }

    #[test]
    fn deep_factorized_ood_intro_matches_dense() {
        use crate::challenger::Challenger;
        for &d in &[7usize, 10, 13, 16] {
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_00D0 ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..(1usize << d)).map(|_| ch.sample_f128()).collect();
            let dense = build_eq_table(&z);
            let want = round_msg_and_eval_lsb(&f, &dense);
            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let mut eq_lo = build_eq_table(&z[1..1 + split]);
            let eq_hi = build_eq_table(&z[1 + split..]);
            let got = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_eq!(got, want, "deep factorized OOD introduction d={d}");
            eq_lo[1] += F128::ONE;
            let bad = round_msg_and_eval_lsb_factorized_eq(&f, &eq_lo, &eq_hi, z[0]);
            assert_ne!(bad, want, "corrupted low factor went undetected d={d}");
        }
    }

    #[test]
    fn deep_lazy_ood_corrected_fold_matches_dense_glue() {
        use crate::challenger::Challenger;
        for &d in &[7usize, 10, 13, 16] {
            let n = 1usize << d;
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_F01D ^ d as u64);
            let z = ch.sample_f128_vec(d);
            let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let beta = ch.sample_f128();
            let r = ch.sample_f128();

            let dense_eq = build_eq_table(&z);
            let mut dense_basis = b.clone();
            crate::field::f128_slice::add_scaled(&mut dense_basis, &dense_eq, beta);
            let (want_f, want_b, want_msg) = fold_and_msg_lsb(&f, &dense_basis, r, None);

            let split = (d - 1).min(LAZY_OOD_EQ_SPLIT_LOW_LOG);
            let eq_lo = build_eq_table(&z[1..1 + split]);
            let mut eq_hi = build_eq_table(&z[1 + split..]);
            let gamma = beta * (F128::ONE + z[0] + r);
            let (got_f, got_b, got_msg) =
                fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
            assert_eq!(&*got_f, &*want_f, "folded witness d={d}");
            assert_eq!(&*got_b, &*want_b, "corrected basis d={d}");
            assert_eq!(got_msg, want_msg, "next-round message d={d}");

            eq_hi[0] += F128::ONE;
            let (_, bad_b, _) =
                fold_and_msg_lsb_inner(&f, &b, r, None, Some((&eq_lo, &eq_hi, gamma)), None);
            assert_ne!(
                &*bad_b, &*want_b,
                "corrupted high factor went undetected d={d}"
            );
        }
    }

    #[test]
    fn deep_lazy_ood_state_machine_matches_dense_across_ordinary_glue() {
        use crate::challenger::Challenger;
        for &d in &[10usize, 16] {
            let n = 1usize << d;
            let mut ch = crate::challenger::RandomChallenger::new(0xDEE9_57A7 ^ d as u64);
            let f: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let z = ch.sample_f128_vec(d);
            let ordinary: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
            let ordinary_claim = ch.sample_f128();
            let beta = ch.sample_f128();
            let alpha = ch.sample_f128();
            let r = ch.sample_f128();
            let target = ch.sample_f128();

            let (mut dense, dense_start) = SumcheckProver::new(f.clone(), b.clone(), target);
            let (mut lazy, lazy_start) = SumcheckProver::new(f, b, target);
            assert_eq!(lazy_start, dense_start);

            let (dense_intro, dense_y) = dense.introduce_new_with_eval(build_eq_table(&z));
            let (lazy_intro, lazy_y) = lazy.introduce_new_ood_factorized(&z).unwrap();
            assert_eq!((lazy_intro, lazy_y), (dense_intro, dense_y));
            dense.glue(beta);
            lazy.glue_factorized_ood(beta);

            let dense_ordinary = dense.introduce_new(ordinary.clone(), ordinary_claim);
            let lazy_ordinary = lazy.introduce_new(ordinary, ordinary_claim);
            assert_eq!(lazy_ordinary, dense_ordinary);
            dense.glue(alpha);
            lazy.glue(alpha);

            assert_eq!(lazy.fold(r), dense.fold(r), "d={d}");
            assert_eq!(&*lazy.f, &*dense.f, "d={d}");
            assert_eq!(&*lazy.combined_basis, &*dense.combined_basis, "d={d}");
            assert_eq!(lazy.t_r, dense.t_r, "d={d}");
            assert_eq!(lazy.transcript, dense.transcript, "d={d}");
        }
    }

    #[test]
    fn ranked_deep_lazy_ood_selector_is_exact() {
        let security = LigeritoSecurityConfig::from_toml_str(include_str!(
            "../../configs/ligerito/m32_fast.toml"
        ))
        .unwrap();
        let (config, _) = security.to_prover_verifier_configs().unwrap();
        let selected = |log_n, nl, count, len, direct, platform, disabled| {
            ranked_deep_lazy_ood_eq_selected(
                &config, log_n, nl, count, len, direct, platform, disabled,
            )
        };
        for &nl in &[16usize, 13, 10, 7] {
            assert!(selected(25, nl, 1, 1 << nl, true, true, false));
            assert!(!selected(24, nl, 1, 1 << nl, true, true, false));
            assert!(!selected(25, nl, 2, 1 << nl, true, true, false));
            assert!(!selected(25, nl, 1, 1 << (nl - 1), true, true, false));
            assert!(!selected(25, nl, 1, 1 << nl, false, true, false));
            assert!(!selected(25, nl, 1, 1 << nl, true, false, false));
            assert!(!selected(25, nl, 1, 1 << nl, true, true, true));
        }
        assert!(!selected(25, 19, 1, 1 << 19, true, true, false));
        assert!(!selected(25, 15, 1, 1 << 15, true, true, false));
        assert!(!selected(25, 4, 1, 1 << 4, true, true, false));
    }

    #[test]
    fn transpose_blocked_matches_per_layer() {
        use crate::challenger::Challenger;
        for &log_d in &[0usize, 1, 2, 5, 8, 12, 16, 18, 20] {
            let ntt = AdditiveNttF128::standard(log_d.max(1));
            for top in 0..=log_d {
                let n = 1usize << log_d;
                let mut ch =
                    crate::challenger::RandomChallenger::new(0x7A5E ^ ((log_d * 37 + top) as u64));
                let base: Vec<F128> = (0..n).map(|_| ch.sample_f128()).collect();
                let mut a = base.clone();
                let mut b = base;
                transpose_forward_ntt_dense_layers_per_layer(&ntt, &mut a, top);
                transpose_forward_ntt_dense_layers_blocked(&ntt, &mut b, top);
                assert_eq!(a, b, "log_d={log_d}, top={top}");
            }
        }
    }

    #[test]
    fn induce_sumcheck_poly_with_interleaving_and_v_challenges() {
        use crate::challenger::Challenger;
        let log_msg = 3; // msg_cols = 8
        let log_interleaved = 2; // num_interleaved = 4
        let log_inv_rate = 1; // block_len = 16
        let msg_cols = 1usize << log_msg;
        let num_interleaved = 1usize << log_interleaved;
        let block_len = msg_cols << log_inv_rate;
        let poly_len = msg_cols * num_interleaved;

        let mut ch = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        let poly: Vec<F128> = (0..poly_len).map(|_| ch.sample_f128()).collect();

        let v_challenges: Vec<F128> = (0..log_interleaved).map(|_| ch.sample_f128()).collect();
        let f_folded = partial_eval_lsb(&poly, &v_challenges);
        assert_eq!(f_folded.len(), msg_cols);

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.block_len, block_len);

        let num_queries = 5;
        let mut queries: Vec<usize> = Vec::new();
        while queries.len() < num_queries {
            let q = (ch.sample_f128().lo as usize) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&q| w.row(q).to_vec()).collect();

        let alpha = ch.sample_f128_vec(ceil_log2(queries.len()));
        let sks_vks = eval_sk_at_vks(log_msg);
        let (basis_poly, enforced_sum) = induce_sumcheck_poly(
            log_msg,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );

        let inner: F128 = f_folded
            .iter()
            .zip(basis_poly.iter())
            .map(|(&m, &b)| m * b)
            .fold(F128::ZERO, |a, v| a + v);
        assert_eq!(
            inner, enforced_sum,
            "folded-msg · basis_poly != enforced_sum (interleaved + v_challenges path)"
        );
    }

    #[test]
    fn ligerito_r1_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0; // unused — kept to silence the moved literal

        let mut rng = crate::challenger::RandomChallenger::new(0xCAFE_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();

        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let queries: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let grinding_bits = vec![0; log_inv_rates.len()];
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries,
            grinding_bits,
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let _ = num_queries; // queries derived per-level from log_inv_rates now

        let mut p_ch = crate::challenger::FsChallenger::new(b"test");
        let proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        let mut v_ch = crate::challenger::FsChallenger::new(b"test");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "verifier rejected a valid proof");
    }

    fn size_breakdown_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        use crate::challenger::Challenger;
        use std::time::Instant;
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);

        let r = recursive_ks.len();
        let mut recursive_log_msg_cols = Vec::with_capacity(r);
        let mut n_running = log_n - initial_k;
        for &k in &recursive_ks {
            assert!(n_running >= k);
            recursive_log_msg_cols.push(n_running - k);
            n_running -= k;
        }

        let mut rng = crate::challenger::RandomChallenger::new(0xBEEFCAFE);
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        eprintln!(
            "log_n={log_n}  initial_k={initial_k}  ks={:?}  log_inv_rates={:?}  queries={:?}",
            recursive_ks, log_inv_rates, queries_per_level
        );
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);
        drop(eq); // free 16 MB

        let grinding_bits = vec![0; log_inv_rates.len()];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: Default::default(),
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: recursive_ks.clone(),
            queries: queries_per_level,
            grinding_bits,
            fold_grinding_bits: vec![0; r + 1],
            ood_samples: vec![0; r + 1],
            merkle_hash: Default::default(),
        };

        let mut best = std::time::Duration::from_secs(3600);
        let mut proof = {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            recursive_prover(&cfg, &poly, &z, v, &mut p_ch)
        };
        for _ in 0..3 {
            let mut p_ch = crate::challenger::FsChallenger::new(b"size-test");
            let t = Instant::now();
            proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
            let el = t.elapsed();
            if el < best {
                best = el;
            }
        }
        eprintln!(
            "--- Ligerito proof: prover {:.2?} (best of 3), size: ---",
            best
        );
        proof.print_size_breakdown();

        let mut v_ch = crate::challenger::FsChallenger::new(b"size-test");
        assert!(recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch));
        proof.size_bytes()
    }

    #[test]
    fn ligerito_size_breakdown_m20_uniform_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 1, 1]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate() {
        size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
    }

    #[test]
    fn ligerito_size_breakdown_m20_decreasing_rate_thin() {
        size_breakdown_at(20, 4, vec![3, 3, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_aggressive() {
        size_breakdown_at(24, 3, vec![3, 3, 3, 3, 3], vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_uniform_rate() {
        size_breakdown_at(24, 5, vec![5, 4, 3], vec![1, 1, 1, 1]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m24_decreasing_rate() {
        size_breakdown_at(24, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22() {
        size_breakdown_at(22, 4, vec![4, 4, 3], vec![1, 2, 3, 4]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m22_initial_k6() {
        size_breakdown_at(22, 6, vec![3, 3, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    #[ignore]
    fn ligerito_size_breakdown_m23() {
        size_breakdown_at(23, 4, vec![4, 4, 3, 3], vec![1, 2, 3, 4, 5]);
    }

    fn multi_proof_num_siblings(positions: &[usize], num_leaves: usize) -> usize {
        let mut active: Vec<usize> = positions.to_vec();
        active.sort_unstable();
        active.dedup();
        let mut sib_count = 0usize;
        let mut level_len = num_leaves;
        while level_len > 1 {
            let mut next = Vec::with_capacity(active.len());
            let mut i = 0;
            while i < active.len() {
                let p = active[i];
                let sib_active = i + 1 < active.len() && active[i + 1] == (p ^ 1);
                if sib_active {
                    i += 2;
                } else {
                    sib_count += 1;
                    i += 1;
                }
                next.push(p >> 1);
            }
            active = next;
            level_len >>= 1;
        }
        sib_count
    }

    fn estimate_size_at(
        log_n: usize,
        initial_k: usize,
        recursive_ks: Vec<usize>,
        log_inv_rates: Vec<usize>,
    ) -> usize {
        const ELEM: usize = core::mem::size_of::<F128>();
        assert_eq!(log_inv_rates.len(), recursive_ks.len() + 1);
        let r = recursive_ks.len();
        let kb = |b: usize| {
            if b >= 1024 * 1024 {
                format!("{:.2} MB", b as f64 / 1024.0 / 1024.0)
            } else if b >= 1024 {
                format!("{:.1} KB", b as f64 / 1024.0)
            } else {
                format!("{} B", b)
            }
        };

        let mut log_num_interleaved: Vec<usize> = vec![initial_k];
        log_num_interleaved.extend_from_slice(&recursive_ks);
        let mut log_msg_cols: Vec<usize> = Vec::with_capacity(r + 1);
        let mut n_running = log_n;
        for i in 0..=r {
            assert!(
                n_running >= log_num_interleaved[i],
                "config infeasible at commit {i}: dim {n_running} < lanes {}",
                log_num_interleaved[i]
            );
            log_msg_cols.push(n_running - log_num_interleaved[i]);
            n_running -= log_num_interleaved[i]; // consumes initial_k or k_{i-1}
        }
        let yr_log_n = n_running; // = log_n - initial_k - Σ k_i
        let queries_per_level: Vec<usize> = log_inv_rates.iter().map(|&r| udr_queries(r)).collect();
        let log_block_len: Vec<usize> = log_msg_cols
            .iter()
            .zip(log_inv_rates.iter())
            .map(|(&m, &r)| m + r)
            .collect();

        eprintln!(
            "m={log_n}  initial_k={initial_k}  ks={:?}  rates={:?}  queries={:?}  yr_log={yr_log_n}",
            recursive_ks, log_inv_rates, queries_per_level
        );

        let mut ch = crate::challenger::FsChallenger::new(b"estimate");
        let mut total_opened = 0usize;
        let mut total_merkle = 0usize;
        for i in 0..=r {
            let bl = 1usize << log_block_len[i];
            let qn = queries_per_level[i];
            if qn > bl {
                eprintln!(
                    "  INFEASIBLE at commit {i}: queries ({qn}) > block_len ({bl}). Pick a higher rate (smaller bl) or smaller queries."
                );
                return usize::MAX;
            }
            let qs = sample_distinct_queries(&mut ch, bl, qn);
            let sib = multi_proof_num_siblings(&qs, bl);
            let opened = qn * (1usize << log_num_interleaved[i]) * ELEM;
            let merkle = sib * 32;
            let label = if i == 0 {
                "L0 (initial)"
            } else if i == r {
                "L{} (final)"
            } else {
                "L{} (recursive)"
            };
            eprintln!(
                "  {label} [bl=2^{}, lanes=2^{}, q={qn}]: opened={}  merkle={} ({} sibs)",
                log_block_len[i],
                log_num_interleaved[i],
                kb(opened),
                kb(merkle),
                sib,
            );
            total_opened += opened;
            total_merkle += merkle;
        }
        let yr_b = (1usize << yr_log_n) * ELEM;
        let roots_b = (r + 1) * 32;
        let sumcheck_msgs = 1 + r + recursive_ks.iter().sum::<usize>();
        let tx_b = sumcheck_msgs * 2 * ELEM;
        let total = total_opened + total_merkle + yr_b + roots_b + tx_b;
        eprintln!(
            "  TOTALS: opened={}  merkle={}  yr={}  roots={}  transcript={}  → GRAND={}",
            kb(total_opened),
            kb(total_merkle),
            kb(yr_b),
            kb(roots_b),
            kb(tx_b),
            kb(total),
        );
        total
    }

    #[test]
    fn estimator_matches_actual_m20() {
        let estimated = estimate_size_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        let actual = size_breakdown_at(20, 4, vec![4, 3], vec![1, 2, 4]);
        let diff = estimated.abs_diff(actual);
        eprintln!("estimator={estimated}  actual={actual}  diff={diff}");
        assert!(
            diff < actual / 20,
            "estimator drift too large: {diff} bytes"
        );
    }

    #[test]
    fn estimate_ligerito_m29() {
        eprintln!("\n=== Ligerito m=29 — decreasing rate (the real Ligerito design) ===");
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!(
            "\n=== Ligerito m=29 — uniform rate 1/2 (basefold-style baseline, infeasible at deepest level) ==="
        );
        estimate_size_at(29, 4, vec![4, 4, 4, 4, 3], vec![1, 1, 1, 1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — uniform rate, shallower (R=2) ===");
        estimate_size_at(29, 4, vec![10, 10], vec![1, 1, 1]);

        eprintln!("\n=== Ligerito m=29 — thinner lanes ===");
        estimate_size_at(
            29,
            3,
            vec![3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );
    }

    #[test]
    fn estimate_ligerito_m30() {
        eprintln!("\n=== Ligerito m=30 — decreasing rate ===");
        estimate_size_at(30, 4, vec![4, 4, 4, 4, 4, 3], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n=== Ligerito m=30 — thinner lanes ===");
        estimate_size_at(
            30,
            3,
            vec![3, 3, 3, 3, 3, 3, 3, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
    }

    #[test]
    fn estimate_ligerito_m29_initial_k6() {
        eprintln!(
            "\n=== Ligerito m=29 — initial_k=6 (matches basefold's 64-lane initial leaves) ==="
        );
        eprintln!("\n  Config A: thin recursive lanes, aggressive rate decrease");
        estimate_size_at(29, 6, vec![3, 3, 3, 3, 3, 2], vec![1, 2, 3, 4, 5, 6, 7]);

        eprintln!("\n  Config B: medium recursive lanes, fewer levels");
        estimate_size_at(29, 6, vec![4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6]);

        eprintln!("\n  Config C: 2x6-bit recursive lanes (= basefold's epoch leaves)");
        estimate_size_at(29, 6, vec![6, 6, 4, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn estimate_ligerito_m30_initial_k6() {
        eprintln!("\n=== Ligerito m=30 — initial_k=6 ===");
        eprintln!("\n  Config A: thin recursive lanes");
        estimate_size_at(
            30,
            6,
            vec![3, 3, 3, 3, 3, 3, 2],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        );

        eprintln!("\n  Config B: medium");
        estimate_size_at(30, 6, vec![4, 4, 4, 4, 3, 3], vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn ligerito_r2_roundtrip_accepts() {
        use crate::challenger::Challenger;
        let log_n = 18;
        let initial_k = 3;
        let k_0 = 3;
        let k_1 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xABCD_1234);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate; 3];
        let _ = num_queries;
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: Default::default(),
        };
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0, log_n - initial_k - k_0 - k_1],
            recursive_ks: vec![k_0, k_1],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0; 3],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);
        assert_eq!(proof.recursive_roots.len(), 2);
        assert_eq!(proof.recursive_proofs.len(), 1);

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-r2");
        let ok = recursive_verifier(&v_cfg, &proof, &z, v, &mut v_ch);
        assert!(ok, "R=2 verifier rejected valid proof");
    }

    #[test]
    fn ligerito_proof_bincode_roundtrip() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xDEED_F00D);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut p_ch = crate::challenger::FsChallenger::new(b"serde");
        let proof = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let bytes = bincode::serialize(&proof).expect("serialize");
        let proof2: LigeritoProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(proof, proof2);
        eprintln!("LigeritoProof bincode size: {} bytes", bytes.len());
    }

    #[test]
    fn recursive_prover_with_basis_roundtrip_single_claim() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_CAFE);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"basis-test");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "basis-based verifier rejected valid proof");
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn direct_fold8_four_map_gfni_matches_fold_one_slot_sum() {
        use crate::zerocheck::multilinear::kernels::x86_64::{
            build_row_fold_mats, gfni_fold64_four_maps_staged,
        };
        let mut state = 0xB6F1_6408_5EED_191Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let tables: Vec<Vec<F128>> = (0..4)
            .map(|_| {
                let generators: Vec<F128> = (0..128)
                    .map(|_| F128 {
                        lo: next(),
                        hi: next(),
                    })
                    .collect();
                super::super::ring_switch::build_fold_byte_table(&generators)
            })
            .collect();
        let values: Vec<Vec<u64>> = (0..4).map(|_| (0..64).map(|_| next()).collect()).collect();
        let mats: Vec<[u64; 128]> = tables
            .iter()
            .map(|table| build_row_fold_mats(&table[..8 * 256]))
            .collect();
        let mut got = vec![F128::ZERO; 64];
        let mut planes = vec![F128::ZERO; 64];
        unsafe {
            gfni_fold64_four_maps_staged(
                values[0].as_ptr().cast::<u8>(),
                &mats[0],
                values[1].as_ptr().cast::<u8>(),
                &mats[1],
                values[2].as_ptr().cast::<u8>(),
                &mats[2],
                values[3].as_ptr().cast::<u8>(),
                &mats[3],
                got.as_mut_ptr(),
                planes.as_mut_ptr().cast(),
            );
        }
        for i in 0..64 {
            let expect = (0..4).fold(F128::ZERO, |acc, map| {
                acc + super::super::ring_switch::fold_one_slot(
                    F128::new(values[map][i], 0),
                    &tables[map],
                )
            });
            assert_eq!(got[i], expect);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn direct_fold4_two_claim_gfni_matches_composed_table_oracle() {
        use crate::zerocheck::multilinear::kernels::x86_64::{
            build_row_fold_mats_from_cols, gfni_fold64_four_maps_staged,
        };
        use core::arch::x86_64::_mm512_setzero_si512;

        let mut state = 0xD1CE_F004_5EED_191Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let direct_tables: Vec<Vec<F128>> = (0..2)
            .map(|_| {
                let generators: Vec<F128> = (0..128).map(|_| F128::new(next(), next())).collect();
                super::super::ring_switch::build_fold_byte_table(&generators)
            })
            .collect();
        let eq_hi = [F128::new(next(), next()), F128::new(next(), next())];
        let rows: Vec<(Vec<u64>, Vec<u64>)> = (0..2)
            .map(|_| {
                (
                    (0..64).map(|_| next()).collect(),
                    (0..64).map(|_| next()).collect(),
                )
            })
            .collect();
        let cols0 = super::super::ring_switch::compose_block_cols(&direct_tables[0], eq_hi[0]);
        let cols1 = super::super::ring_switch::compose_block_cols(&direct_tables[1], eq_hi[1]);
        let mats0_lo = build_row_fold_mats_from_cols(&cols0[..64]);
        let mats0_hi = build_row_fold_mats_from_cols(&cols0[64..]);
        let mats1_lo = build_row_fold_mats_from_cols(&cols1[..64]);
        let mats1_hi = build_row_fold_mats_from_cols(&cols1[64..]);
        let mut got = vec![F128::ZERO; 64];
        let mut planes = unsafe { [_mm512_setzero_si512(); 16] };
        unsafe {
            gfni_fold64_four_maps_staged(
                rows[0].0.as_ptr().cast::<u8>(),
                &mats0_lo,
                rows[0].1.as_ptr().cast::<u8>(),
                &mats0_hi,
                rows[1].0.as_ptr().cast::<u8>(),
                &mats1_lo,
                rows[1].1.as_ptr().cast::<u8>(),
                &mats1_hi,
                got.as_mut_ptr(),
                planes.as_mut_ptr(),
            );
        }
        let mut composed0 = vec![F128::ZERO; super::super::ring_switch::FOLD_TABLE_TOTAL];
        let mut composed1 = vec![F128::ZERO; super::super::ring_switch::FOLD_TABLE_TOTAL];
        super::super::ring_switch::compose_block_table(&direct_tables[0], eq_hi[0], &mut composed0);
        super::super::ring_switch::compose_block_table(&direct_tables[1], eq_hi[1], &mut composed1);
        for slot in 0..64 {
            let expect = super::super::ring_switch::fold_one_slot(
                F128::new(rows[0].0[slot], rows[0].1[slot]),
                &composed0,
            ) + super::super::ring_switch::fold_one_slot(
                F128::new(rows[1].0[slot], rows[1].1[slot]),
                &composed1,
            );
            assert_eq!(got[slot], expect, "slot {slot}");
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn direct_fold8_composed_mats_match_column_builder() {
        use crate::zerocheck::multilinear::kernels::x86_64::build_row_fold_mats_from_cols;

        let mut state = 0xD1CE_F008_5EED_191Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..3 {
            let generators: Vec<F128> = (0..128)
                .map(|_| F128::new(next(), next()))
                .collect();
            let table = super::super::ring_switch::build_fold_byte_table(&generators);
            let e_hi = F128::new(next(), next());
            let map =
                super::super::ring_switch::build_gfni_direct_fold_map_from_generators(&generators);
            let (got_lo, got_hi) =
                super::super::ring_switch::compose_block_mats_gfni(&map, e_hi);
            let cols = super::super::ring_switch::compose_block_cols(&table, e_hi);
            assert_eq!(
                got_lo,
                build_row_fold_mats_from_cols(&cols[..64]),
                "low matrix mismatch"
            );
            assert_eq!(
                got_hi,
                build_row_fold_mats_from_cols(&cols[64..]),
                "high matrix mismatch"
            );
        }
    }


    #[test]
    fn direct_fold4_gfni_gate_is_ranked_shape_only() {
        assert!(direct_fold4_b_gfni_shape(2, false, 64));
        assert!(direct_fold4_b_gfni_shape(2, false, 8192));
        assert!(!direct_fold4_b_gfni_shape(1, false, 64));
        assert!(!direct_fold4_b_gfni_shape(3, false, 64));
        assert!(!direct_fold4_b_gfni_shape(2, true, 64));
        assert!(!direct_fold4_b_gfni_shape(2, false, 60));
    }

    #[test]
    fn direct_fold8_sparse_k4_cross_level_proof_bytes_match_ordinary() {
        use crate::challenger::Challenger;

        let log_n = 12;
        let initial_k = 6;
        let recursive_ks = vec![3usize, 2];
        let log_inv_rate = 1;
        let mut rng = crate::challenger::RandomChallenger::new(0xD1CE_F008);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let suffix: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let scaled_rdp: Vec<F128> = build_eq_table(
            &(0..crate::pcs::LOG_PACKING)
                .map(|_| rng.sample_f128())
                .collect::<Vec<_>>(),
        );
        let combined_basis =
            super::super::ring_switch::fold_b128_elems(&build_eq_table(&suffix), &scaled_rdp);
        let target = poly
            .iter()
            .zip(combined_basis.iter())
            .map(|(&f, &b)| f * b)
            .fold(F128::ZERO, |acc, value| acc + value);

        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let tail_eq = build_eq_table(&suffix[6..]);
        let table = super::super::ring_switch::build_fold_byte_table(&scaled_rdp);
        let low_eq = build_eq_table(&suffix[..6]);
        let mut a_state = vec![F128::ZERO; 64 * n_packed];
        for e in 0..64 {
            let strided: Vec<F128> = (0..poly.len() / 64)
                .map(|rest| poly[64 * rest + e])
                .collect();
            let bank = super::super::ring_switch::fold_1b_rows_naive(&strided, &tail_eq);
            let transposed = super::super::ring_switch::tensor_algebra_transpose(&bank);
            for (bit, value) in transposed.into_iter().enumerate() {
                a_state[bit * 64 + e] = value;
            }
        }
        let mut w_state = vec![F128::ZERO; 64 * n_packed];
        for d_low in 0..64 {
            let mut basis_product = low_eq[d_low];
            w_state[d_low] = super::super::ring_switch::fold_one_slot(basis_product, &table);
            for bit in 1..n_packed {
                basis_product = crate::field::mul_by_x(basis_product);
                w_state[bit * 64 + d_low] =
                    super::super::ring_switch::fold_one_slot(basis_product, &table);
            }
        }
        let mut round0 = (F128::ZERO, F128::ZERO);
        for i in (0..a_state.len()).step_by(2) {
            let a0 = a_state[i];
            let a1 = a_state[i + 1];
            let b0 = w_state[i];
            let b1 = w_state[i + 1];
            round0.0 += a0 * b0;
            round0.1 += (a0 + a1) * (b0 + b1);
        }
        let (eq_lo, eq_hi) =
            super::super::ring_switch::build_eq_split(&suffix[6..], (log_n - 6) / 2);
        let direct = vec![super::super::ring_switch::DirectFold8Factors {
            eq_lo,
            eq_hi,
            a_state,
            w_state,
            round0,
        }];

        let log_inv_rates = vec![1usize, 2, 3];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![3, 1],
            recursive_ks: recursive_ks.clone(),
            queries: vec![8; 3],
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0, 1, 1],
            merkle_hash: Default::default(),
        };
        let ntt_0 = AdditiveNttF128::standard(log_n - initial_k + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_n - initial_k,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );

        let mut ordinary_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));
        let ordinary = recursive_prover_with_basis_precomputed_round0(
            &cfg,
            poly.clone(),
            combined_basis.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            None,
            &mut ordinary_challenger,
        );
        let mut direct_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(4));
        let got = recursive_prover_with_basis_direct_fold8(
            &cfg,
            poly,
            Vec::new(),
            direct,
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            round0,
            None,
            &mut direct_challenger,
        );
        SPARSE_DUAL_TEST_DEPTH.with(|value| value.set(0));

        assert_eq!(got, ordinary);
        assert_eq!(
            bincode::serialize(&(got.clone(), target)).expect("serialize direct-fold8 proof/claim"),
            bincode::serialize(&(ordinary, target)).expect("serialize ordinary proof/claim"),
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 2,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![3, 1],
            recursive_ks,
            queries: vec![8; 3],
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 3],
            ood_samples: vec![0, 1, 1],
            merkle_hash: Default::default(),
        };
        let mut verifier_challenger =
            crate::challenger::FsChallenger::new(b"direct-fold8-proof-byte-oracle");
        assert!(recursive_verifier_with_basis(
            &v_cfg,
            &got,
            &combined_basis,
            target,
            &wtns_0.root(),
            &mut verifier_challenger,
        ));
    }

    #[test]
    fn induce_sumcheck_evaluate_at_residual_matches_dense() {
        use crate::challenger::Challenger;
        let log_msg_cols = 6;
        let yr_log_n = 2;
        let prefix_len = log_msg_cols - yr_log_n;
        let num_interleaved = 4;
        let log_num_interleaved = 2;
        let num_queries = 5;

        let mut rng = crate::challenger::RandomChallenger::new(0x2017_5052);
        let queries: Vec<usize> = (0..num_queries).map(|i| (i * 7 + 3) % (1 << 8)).collect();
        let opened_rows: Vec<Vec<F128>> = (0..num_queries)
            .map(|_| (0..num_interleaved).map(|_| rng.sample_f128()).collect())
            .collect();
        let v_challenges: Vec<F128> = (0..log_num_interleaved)
            .map(|_| rng.sample_f128())
            .collect();
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();
        let ris_for_basis: Vec<F128> = (0..prefix_len).map(|_| rng.sample_f128()).collect();
        let sks_vks = eval_sk_at_vks(log_msg_cols);

        let (basis_dense, dense_enforced_sum) = induce_sumcheck_poly(
            log_msg_cols,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            &alpha,
        );
        let dense_residual = partial_eval_lsb(&basis_dense, &ris_for_basis);

        let succinct_enforced_sum =
            induce_sumcheck_enforced_sum(&opened_rows, &v_challenges, &queries, &alpha);
        let succinct_residual = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &ris_for_basis,
            yr_log_n,
        );

        assert_eq!(
            succinct_enforced_sum, dense_enforced_sum,
            "enforced_sum mismatch"
        );
        assert_eq!(
            succinct_residual.len(),
            dense_residual.len(),
            "residual length mismatch"
        );
        for (i, (s, d)) in succinct_residual
            .iter()
            .zip(dense_residual.iter())
            .enumerate()
        {
            assert_eq!(s, d, "residual mismatch at y={i}");
        }
    }

    #[test]
    fn final_level_binding_pins_yr_to_committed_codeword() {
        use crate::challenger::Challenger;
        let log_msg_cols = 5; // yr has 32 entries (within the shipped yr_log_n range)
        let log_inv_rate = 1;
        let num_queries = 20;
        let msg_cols = 1usize << log_msg_cols;
        let block_len = msg_cols << log_inv_rate;

        let mut rng = crate::challenger::RandomChallenger::new(0xB19D_1235);
        let yr: Vec<F128> = (0..msg_cols).map(|_| rng.sample_f128()).collect();
        let ntt = AdditiveNttF128::standard(log_msg_cols + log_inv_rate);
        let wtns = ligero_commit(&yr, log_msg_cols, 0, log_inv_rate, &ntt, HashKind::Sha256);

        let mut queries: Vec<usize> = Vec::new();
        let mut q = 1usize;
        while queries.len() < num_queries {
            q = (q * 73 + 41) % block_len;
            if !queries.contains(&q) {
                queries.push(q);
            }
        }
        let opened_rows: Vec<Vec<F128>> = queries.iter().map(|&p| wtns.row(p).to_vec()).collect();

        let level_rs: Vec<F128> = Vec::new(); // num_interleaved = 1
        let alpha: Vec<F128> = (0..ceil_log2(num_queries))
            .map(|_| rng.sample_f128())
            .collect();

        let enforced_sum = induce_sumcheck_enforced_sum(&opened_rows, &level_rs, &queries, &alpha);
        let sks_vks = eval_sk_at_vks(log_msg_cols);
        let induced_basis = induce_sumcheck_evaluate_at_residual(
            log_msg_cols,
            &sks_vks,
            &queries,
            &alpha,
            &[],
            log_msg_cols,
        );
        let inner = |v: &[F128]| -> F128 {
            v.iter()
                .zip(induced_basis.iter())
                .map(|(&a, &b)| a * b)
                .fold(F128::ZERO, |s, x| s + x)
        };

        assert_eq!(
            inner(&yr),
            enforced_sum,
            "honest yr must satisfy ⟨yr, induced_basis⟩ == enforced_sum"
        );

        let jnz = induced_basis
            .iter()
            .position(|b| !b.is_zero())
            .expect("induced basis must not be identically zero");
        let mut yr_bad = yr.clone();
        yr_bad[jnz] += F128::ONE;
        assert_ne!(
            inner(&yr_bad),
            enforced_sum,
            "a forged yr must break the final-level proximity tie"
        );
    }

    #[test]
    fn recursive_verifier_with_basis_succinct_matches_dense() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0x52CC_2017);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut v_ch = crate::challenger::FsChallenger::new(b"succ-cmp");
        let dense_ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(dense_ok, "dense verifier must accept");

        let mut v_ch2 = crate::challenger::FsChallenger::new(b"succ-cmp");
        let eval_b_residual = |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
            let yr_len = 1usize << yr_log_n;
            let mut point = ris.to_vec();
            point.resize(ris.len() + yr_log_n, F128::ZERO);
            (0..yr_len)
                .map(|y| {
                    for j in 0..yr_log_n {
                        point[ris.len() + j] = if (y >> j) & 1 == 1 {
                            F128::ONE
                        } else {
                            F128::ZERO
                        };
                    }
                    crate::zerocheck::multilinear::eq_eval(&z, &point)
                })
                .collect()
        };
        let succ_ok = recursive_verifier_with_basis_succinct(
            &v_cfg,
            &proof,
            log_n,
            target,
            &initial_root,
            eval_b_residual,
            &mut v_ch2,
        );
        assert!(succ_ok, "succinct verifier must accept");
    }

    fn ood_test_configs(
        log_n: usize,
        initial_k: usize,
        ks: &[usize],
        ood_samples: Vec<usize>,
        fold_grinding_bits: Vec<usize>,
    ) -> (ProverConfig, VerifierConfig) {
        let r = ks.len();
        let log_inv_rates: Vec<usize> = (0..=r).map(|i| 1 + i).collect();
        let mut recursive_log_msg_cols = Vec::new();
        let mut dim = log_n - initial_k;
        for &k in ks {
            recursive_log_msg_cols.push(dim - k);
            dim -= k;
        }
        let queries = vec![20usize; r + 1];
        let grinding_bits = vec![0usize; r + 1];
        let p = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: recursive_log_msg_cols.clone(),
            recursive_ks: ks.to_vec(),
            queries: queries.clone(),
            grinding_bits: grinding_bits.clone(),
            fold_grinding_bits: fold_grinding_bits.clone(),
            ood_samples: ood_samples.clone(),
            merkle_hash: Default::default(),
        };
        let v = VerifierConfig {
            log_inv_rates,
            recursive_steps: r,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols,
            recursive_ks: ks.to_vec(),
            queries,
            grinding_bits,
            fold_grinding_bits,
            ood_samples,
            merkle_hash: Default::default(),
        };
        (p, v)
    }

    #[test]
    fn ligerito_ood_and_fold_grinding_roundtrip_and_tamper() {
        use crate::challenger::Challenger;
        let log_n = 12;
        let initial_k = 2;
        let ks = [2usize, 2];
        let (p_cfg, v_cfg) = ood_test_configs(log_n, initial_k, &ks, vec![0, 2, 2], vec![3, 3, 3]);

        let mut rng = crate::challenger::RandomChallenger::new(0x00D_7E57);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"ood-test");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        assert_eq!(proof.ood_values.len(), 4, "2 OOD samples each at L1 and L2");
        assert_eq!(proof.fold_grinding_nonces.len(), initial_k + ks[0] + ks[1]);

        let dense = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis(&v_cfg, proof, &b, target, &initial_root, &mut ch)
        };
        let eval_b_residual = {
            let z = z.clone();
            move |ris: &[F128], yr_log_n: usize| -> Vec<F128> {
                let yr_len = 1usize << yr_log_n;
                let mut point = ris.to_vec();
                point.resize(ris.len() + yr_log_n, F128::ZERO);
                (0..yr_len)
                    .map(|y| {
                        for j in 0..yr_log_n {
                            point[ris.len() + j] = if (y >> j) & 1 == 1 {
                                F128::ONE
                            } else {
                                F128::ZERO
                            };
                        }
                        crate::zerocheck::multilinear::eq_eval(&z, &point)
                    })
                    .collect()
            }
        };
        let succinct = |proof: &LigeritoProof| {
            let mut ch = crate::challenger::FsChallenger::new(b"ood-test");
            recursive_verifier_with_basis_succinct(
                &v_cfg,
                proof,
                log_n,
                target,
                &initial_root,
                &eval_b_residual,
                &mut ch,
            )
        };

        assert!(dense(&proof), "dense verifier must accept OOD proof");
        assert!(succinct(&proof), "succinct verifier must accept OOD proof");

        let mut bad_ood = proof.clone();
        bad_ood.ood_values[0] += F128::ONE;
        assert!(!dense(&bad_ood), "dense must reject tampered OOD value");
        assert!(
            !succinct(&bad_ood),
            "succinct must reject tampered OOD value"
        );

        let mut bad_nonce = proof.clone();
        bad_nonce.fold_grinding_nonces[0] ^= 0xDEAD_BEEF;
        assert!(!dense(&bad_nonce), "dense must reject tampered fold nonce");
        assert!(
            !succinct(&bad_nonce),
            "succinct must reject tampered fold nonce"
        );
    }

    #[test]
    fn ligerito_fast_profile_m22_roundtrip() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");
        assert!(p_cfg.ood_samples.iter().skip(1).any(|&s| s > 0));
        assert!(p_cfg.fold_grinding_bits.iter().any(|&g| g > 0));

        let mut rng = crate::challenger::RandomChallenger::new(0xFA57_0022);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-fast");
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "m22 fast profile proof must verify"
        );
    }

    #[test]
    #[ignore]
    fn open_sumcheck_kernel_e2e_transcript_digest() {
        use crate::challenger::Challenger;
        use std::hash::{Hash as _, Hasher as _};
        let m = 29usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m29 fast prover config");
        let mut v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m29 fast verifier config");
        p_cfg.merkle_hash = HashKind::Blake3;
        v_cfg.merkle_hash = HashKind::Blake3;

        let mut rng = crate::challenger::RandomChallenger::new(0x50A0_0AC1E_u64);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rate_0 = p_cfg.log_inv_rates[0];
        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate_0);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate_0,
            &ntt_0,
            HashKind::Blake3,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::with_hash(b"soa-oracle", HashKind::Blake3);
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{proof:?}").hash(&mut h);
        eprintln!("PROOF_DIGEST {:016x}", h.finish());

        let mut v_ch = crate::challenger::FsChallenger::with_hash(b"soa-oracle", HashKind::Blake3);
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "m29 blake3 proof must verify"
        );
    }

    #[test]
    fn ligerito_m22_roundtrip_under_blake3() {
        use crate::challenger::Challenger;
        let m = 22usize;
        let log_n = m - crate::pcs::LOG_PACKING;
        let initial_k = 6;
        let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast prover config");
        let mut v_cfg = verifier_config_for(log_n, initial_k, LigeritoProfile::Fast)
            .expect("m22 fast verifier config");
        assert_eq!(p_cfg.merkle_hash, HashKind::Sha256);
        p_cfg.merkle_hash = HashKind::Blake3;
        v_cfg.merkle_hash = HashKind::Blake3;

        let mut rng = crate::challenger::RandomChallenger::new(0xB1A5_E300);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let b = build_eq_table(&z);
        let target: F128 = poly
            .iter()
            .zip(b.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            1,
            &ntt_0,
            HashKind::Blake3,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        let proof = recursive_prover_with_basis(
            &p_cfg,
            poly,
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let mut v_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch),
            "blake3 Merkle proof must verify"
        );

        let mut wrong_cfg = v_cfg.clone();
        wrong_cfg.merkle_hash = HashKind::Sha256;
        let mut w_ch = crate::challenger::FsChallenger::new(b"m22-blake3");
        assert!(
            !recursive_verifier_with_basis(
                &wrong_cfg,
                &proof,
                &b,
                target,
                &initial_root,
                &mut w_ch
            ),
            "a sha256-configured verifier must reject a blake3 proof"
        );
    }

    #[test]
    fn ligerito_m22_roundtrip_over_hash_matrix() {
        use crate::challenger::Challenger;
        const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];
        let log_n = 22usize - crate::pcs::LOG_PACKING;
        let initial_k = 6;

        for merkle_hash in KINDS {
            for fs_hash in KINDS {
                let mut p_cfg = prover_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                let mut v_cfg =
                    verifier_config_for(log_n, initial_k, LigeritoProfile::Fast).unwrap();
                p_cfg.merkle_hash = merkle_hash;
                v_cfg.merkle_hash = merkle_hash;

                let mut rng = crate::challenger::RandomChallenger::new(0x4A11_0000);
                let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
                let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
                let b = build_eq_table(&z);
                let target: F128 = poly
                    .iter()
                    .zip(b.iter())
                    .map(|(&a, &c)| a * c)
                    .fold(F128::ZERO, |a, x| a + x);

                let log_msg_cols_0 = log_n - initial_k;
                let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + 1);
                let wtns_0 =
                    ligero_commit(&poly, log_msg_cols_0, initial_k, 1, &ntt_0, merkle_hash);
                let initial_root = wtns_0.root();

                let mut p_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                let proof = recursive_prover_with_basis(
                    &p_cfg,
                    poly,
                    b.clone(),
                    target,
                    &wtns_0.mat,
                    &wtns_0.tree,
                    &mut p_ch,
                );

                let mut v_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", fs_hash);
                assert!(
                    recursive_verifier_with_basis(
                        &v_cfg,
                        &proof,
                        &b,
                        target,
                        &initial_root,
                        &mut v_ch
                    ),
                    "merkle={merkle_hash} fs={fs_hash} must verify"
                );

                let other_fs = match fs_hash {
                    HashKind::Sha256 => HashKind::Blake3,
                    HashKind::Blake3 => HashKind::Sha256,
                };
                let mut w_ch = crate::challenger::FsChallenger::with_hash(b"m22-matrix", other_fs);
                assert!(
                    !recursive_verifier_with_basis(
                        &v_cfg,
                        &proof,
                        &b,
                        target,
                        &initial_root,
                        &mut w_ch
                    ),
                    "merkle={merkle_hash}: an {other_fs} transcript must reject an {fs_hash} proof"
                );
            }
        }
    }

    #[test]
    fn recursive_prover_with_basis_roundtrip_batched_claims() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xBA51_BA51);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z1: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let z2: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let g1 = rng.sample_f128();
        let g2 = rng.sample_f128();
        let b1 = build_eq_table(&z1);
        let b2 = build_eq_table(&z2);
        let b: Vec<F128> = b1
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| g1 * a + g2 * c)
            .collect();
        let v1: F128 = poly
            .iter()
            .zip(b1.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let v2: F128 = poly
            .iter()
            .zip(b2.iter())
            .map(|(&a, &c)| a * c)
            .fold(F128::ZERO, |a, x| a + x);
        let target = g1 * v1 + g2 * v2;

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let wtns_0 = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let initial_root = wtns_0.root();

        let mut p_ch = crate::challenger::FsChallenger::new(b"batched");
        let proof = recursive_prover_with_basis(
            &cfg,
            poly.clone(),
            b.clone(),
            target,
            &wtns_0.mat,
            &wtns_0.tree,
            &mut p_ch,
        );

        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"batched");
        let ok =
            recursive_verifier_with_basis(&v_cfg, &proof, &b, target, &initial_root, &mut v_ch);
        assert!(ok, "batched-basis verifier rejected valid proof");
    }

    #[test]
    fn recursive_prover_with_l0_matches_full() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;

        let mut rng = crate::challenger::RandomChallenger::new(0xACED_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_a = recursive_prover(&cfg, &poly, &z, v, &mut p_ch);

        let log_msg_cols_0 = log_n - initial_k;
        let ntt_0 = AdditiveNttF128::standard(log_msg_cols_0 + log_inv_rate);
        let mut wtns_0_external = ligero_commit(
            &poly,
            log_msg_cols_0,
            initial_k,
            log_inv_rate,
            &ntt_0,
            HashKind::Sha256,
        );
        let mut p_ch_b = crate::challenger::FsChallenger::new(b"l0-test");
        let proof_b = recursive_prover_with_l0(
            &cfg,
            &poly,
            std::mem::take(&mut wtns_0_external.mat),
            std::mem::take(&mut wtns_0_external.tree),
            &z,
            v,
            &mut p_ch_b,
        );

        assert_eq!(proof_a.initial_root, proof_b.initial_root);
        assert_eq!(proof_a.recursive_roots, proof_b.recursive_roots);
        assert_eq!(proof_a.final_proof.yr, proof_b.final_proof.yr);
        assert_eq!(
            proof_a.sumcheck_transcript.len(),
            proof_b.sumcheck_transcript.len()
        );
        for (ma, mb) in proof_a
            .sumcheck_transcript
            .iter()
            .zip(proof_b.sumcheck_transcript.iter())
        {
            assert_eq!(ma.u_0, mb.u_0);
            assert_eq!(ma.u_2, mb.u_2);
        }
        let v_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let mut v_ch = crate::challenger::FsChallenger::new(b"l0-test");
        assert!(recursive_verifier(&v_cfg, &proof_b, &z, v, &mut v_ch));
    }

    #[test]
    fn ligerito_r1_rejects_mutated_yr() {
        use crate::challenger::Challenger;
        let log_n = 14;
        let initial_k = 3;
        let k_0 = 2;
        let log_inv_rate = 1;
        let num_queries = 0;

        let mut rng = crate::challenger::RandomChallenger::new(0xDEAD_BEEF);
        let poly: Vec<F128> = (0..(1usize << log_n)).map(|_| rng.sample_f128()).collect();
        let z: Vec<F128> = (0..log_n).map(|_| rng.sample_f128()).collect();
        let eq = build_eq_table(&z);
        let v: F128 = poly
            .iter()
            .zip(eq.iter())
            .map(|(&a, &b)| a * b)
            .fold(F128::ZERO, |a, x| a + x);

        let log_inv_rates = vec![log_inv_rate, log_inv_rate];
        let _ = num_queries;
        let prover_cfg = ProverConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };
        let verifier_cfg = VerifierConfig {
            log_inv_rates: log_inv_rates.clone(),
            recursive_steps: 1,
            initial_log_msg_cols: log_n - initial_k,
            initial_log_num_interleaved: initial_k,
            initial_k,
            recursive_log_msg_cols: vec![log_n - initial_k - k_0],
            recursive_ks: vec![k_0],
            queries: log_inv_rates.iter().map(|&r| udr_queries(r)).collect(),
            grinding_bits: vec![0; log_inv_rates.len()],
            fold_grinding_bits: vec![0; 2],
            ood_samples: vec![0; 2],
            merkle_hash: Default::default(),
        };

        let mut p_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let mut proof = recursive_prover(&prover_cfg, &poly, &z, v, &mut p_ch);

        proof.final_proof.yr[0] += F128::ONE;

        let mut v_ch = crate::challenger::FsChallenger::new(b"test-mut");
        let ok = recursive_verifier(&verifier_cfg, &proof, &z, v, &mut v_ch);
        assert!(!ok, "verifier accepted a proof with mutated yr");
    }

    #[test]
    fn ligero_commit_encoding_roundtrips_via_inv_ntt() {
        let log_msg = 4; // msg_cols = 16
        let log_interleaved = 3; // num_interleaved = 8
        let log_inv_rate = 1; // block_len = 32
        let msg_cols = 1 << log_msg;
        let num_interleaved = 1 << log_interleaved;
        let block_len = msg_cols << log_inv_rate;

        let poly: Vec<F128> = (0..num_interleaved * msg_cols)
            .map(|i| {
                F128::new(
                    (i as u64).wrapping_mul(0x9E3779B97F4A7C15),
                    0x1234 ^ i as u64,
                )
            })
            .collect();

        let ntt = AdditiveNttF128::standard(log_msg + log_inv_rate);
        let w = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.block_len, block_len);
        assert_eq!(w.num_interleaved, num_interleaved);
        assert_eq!(w.mat.len(), block_len * num_interleaved);

        for lane in 0..num_interleaved {
            let mut col: Vec<F128> = (0..block_len)
                .map(|pos| w.mat[pos * num_interleaved + lane])
                .collect();
            ntt.inverse_transform(&mut col);
            for col_idx in 0..msg_cols {
                assert_eq!(
                    col[col_idx],
                    poly[col_idx * num_interleaved + lane],
                    "lane {lane} col_idx {col_idx} mismatch",
                );
            }
            for col_idx in msg_cols..block_len {
                assert_eq!(
                    col[col_idx],
                    F128::ZERO,
                    "lane {lane} pad position {col_idx} not zero",
                );
            }
        }

        let w2 = ligero_commit(
            &poly,
            log_msg,
            log_interleaved,
            log_inv_rate,
            &ntt,
            HashKind::Sha256,
        );
        assert_eq!(w.root(), w2.root());
    }

    #[test]
    fn gpu_open_merkle_tree_matches_cpu() {
        if !crate::gpu::merkle::available() {
            eprintln!("SKIP gpu_open_merkle_tree_matches_cpu: Metal unavailable");
            return;
        }
        for log_leaves in [16usize, 18] {
            let n_leaves = 1usize << log_leaves;
            let leaf_size = 128usize;
            let data: Vec<F128> = (0..n_leaves * leaf_size / 16)
                .map(|i| F128::new(i as u64, (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                .collect();
            let bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 16) };
            let mut busy_ms = 0.0f64;
            let gpu_tree = gpu_merkle_tree_for_open(bytes, n_leaves, leaf_size, &mut busy_ms)
                .expect("GPU open-merkle session must complete when Metal is available");
            let cpu_tree = merkle::merkle_tree(bytes, n_leaves, HashKind::Blake3);
            assert_eq!(
                gpu_tree, cpu_tree,
                "GPU open-merkle tree != CPU at 2^{log_leaves} leaves"
            );
        }
    }
    #[test]
    fn transpose_pass_b_zero_leaf_matches_general() {
        use crate::challenger::Challenger;
        assert!(tnt_pass_b_zero_selected(false));
        assert!(!tnt_pass_b_zero_selected(true));
        let ntt = AdditiveNttF128::standard(12);
        let mut ch = crate::challenger::RandomChallenger::new(0xB10C_0000_7E80);
        for &(stride, layer) in &[(1usize, 3usize), (2, 2), (4, 1), (8, 0)] {
            assert_eq!(ntt.twiddle(layer, 0), F128::ZERO);
            for &tile in &[1usize, 3, 4, 7, 16, 65] {
                let base: Vec<Vec<F128>> = (0..2 * stride)
                    .map(|_| (0..tile).map(|_| ch.sample_f128()).collect())
                    .collect();
                let mut got = base.clone();
                let mut want = base.clone();
                {
                    let mut cols: Vec<&mut [F128]> =
                        got.iter_mut().map(Vec::as_mut_slice).collect();
                    transpose_pass_b_block_zero_xor(&mut cols, stride, layer, &ntt);
                }
                {
                    let mut cols: Vec<&mut [F128]> =
                        want.iter_mut().map(Vec::as_mut_slice).collect();
                    transpose_pass_b_block_zero_general(&mut cols, stride, layer, &ntt);
                }
                assert_eq!(got, want, "stride={stride} tile={tile}");
                for u in 0..stride {
                    for i in 0..tile {
                        assert_eq!(got[u][i], base[u][i] + base[u + stride][i]);
                        assert_eq!(got[u + stride][i], base[u + stride][i]);
                    }
                }
            }
        }
    }
}



