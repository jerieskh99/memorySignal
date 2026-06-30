//! Family D / change-location: WHERE in the page the change sits (the intra-page mask).

#[derive(Clone, Default, Debug)]
pub struct ChangeLocation {
    pub changed_runs: u16,        // number of contiguous changed-byte segments
    pub change_span: u16,         // last - first changed offset (inclusive)
    pub change_centroid: f32,     // mean changed-byte offset
    pub longest_changed_run: u16, // largest contiguous changed block
    pub change_density: f32,      // changed bytes / span (compactness of the edit)
}

pub fn compute(p: &[u8], q: &[u8]) -> ChangeLocation {
    let mut first: Option<usize> = None;
    let mut last = 0usize;
    let mut runs: u16 = 0;
    let mut cur_run: u16 = 0;
    let mut longest: u16 = 0;
    let mut changed: u32 = 0;
    let mut sum_idx: u64 = 0;
    let mut in_run = false;
    for (i, (&a, &b)) in p.iter().zip(q.iter()).enumerate() {
        if a != b {
            changed += 1;
            sum_idx += i as u64;
            if first.is_none() {
                first = Some(i);
            }
            last = i;
            if !in_run {
                runs += 1;
                in_run = true;
                cur_run = 0;
            }
            cur_run += 1;
            if cur_run > longest {
                longest = cur_run;
            }
        } else {
            in_run = false;
        }
    }
    match first {
        Some(f) => {
            let span = (last - f + 1) as u16;
            ChangeLocation {
                changed_runs: runs,
                change_span: span,
                change_centroid: sum_idx as f32 / changed as f32,
                longest_changed_run: longest,
                change_density: changed as f32 / span as f32,
            }
        }
        None => ChangeLocation::default(), // no change (only reached for unchanged pages)
    }
}
