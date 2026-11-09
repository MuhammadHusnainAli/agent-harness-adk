//! PCM in plain Rust: just enough signal handling for a voice agent.
//!
//! Everything here is 16-bit mono little-endian PCM — what microphones,
//! telephony bridges and every speech API settle on. A 20 ms frame is a few
//! hundred samples; looping over that is microseconds.

use crate::errors::{Error, Result};

/// PCM bytes as signed 16-bit samples. A trailing odd byte is dropped.
pub fn samples(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect()
}

/// Samples back to little-endian PCM bytes.
pub fn to_bytes(data: &[i16]) -> Vec<u8> {
    data.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// How loud a stretch of audio is: 0 is silence, 32768 is full scale.
pub fn rms(pcm: &[u8]) -> f64 {
    let data = samples(pcm);
    if data.is_empty() {
        return 0.0;
    }
    (data.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / data.len() as f64).sqrt()
}

pub fn duration_ms(pcm: &[u8], rate: u32) -> f64 {
    pcm.len() as f64 / 2.0 / rate as f64 * 1000.0
}

/// Changes the sample rate of a stream, a chunk at a time.
///
/// It remembers where one chunk ended, so the next begins exactly there —
/// resampling each chunk on its own leaves a click at every join, and a voice
/// that clicks fifty times a second does not sound like a voice.
#[derive(Debug, Clone)]
pub struct Resampler {
    pub src: u32,
    pub dst: u32,
    made: i64,
    taken: i64,
    last: Option<i16>,
    odd: Vec<u8>,
}

impl Resampler {
    pub fn new(src: u32, dst: u32) -> Result<Self> {
        if src == 0 || dst == 0 {
            return Err(Error::config("sample rates must be positive"));
        }
        Ok(Self { src, dst, made: 0, taken: 0, last: None, odd: vec![] })
    }

    pub fn feed(&mut self, pcm: &[u8]) -> Vec<u8> {
        if self.src == self.dst {
            return pcm.to_vec();
        }
        let mut joined = std::mem::take(&mut self.odd);
        joined.extend_from_slice(pcm);
        if joined.len() % 2 == 1 {
            self.odd = vec![joined.pop().expect("odd length")];
        }
        let data = samples(&joined);
        if data.is_empty() {
            return vec![];
        }
        // The last sample of the previous chunk sits at index -1 of this one.
        let previous = self.last.unwrap_or(data[0]) as f64;
        let (src, dst, count) = (self.src as i64, self.dst as i64, data.len() as i64);
        let mut out = Vec::new();
        let mut made = self.made;
        loop {
            // Where output sample `made` falls among the input samples — kept
            // in whole numbers, so the answer does not depend on how the
            // stream happened to be cut into chunks.
            let position = made * src;
            let index = position / dst - self.taken;
            if index >= count - 1 {
                break;
            }
            let left = if index < 0 { previous } else { data[index as usize] as f64 };
            let right = data[(index + 1) as usize] as f64;
            out.push((left + (right - left) * ((position % dst) as f64 / dst as f64)) as i16);
            made += 1;
        }
        self.made = made;
        self.taken += count;
        self.last = data.last().copied();
        to_bytes(&out)
    }
}

/// A whole recording at another sample rate.
pub fn resample(pcm: &[u8], src: u32, dst: u32) -> Result<Vec<u8>> {
    Ok(Resampler::new(src, dst)?.feed(pcm))
}

/// PCM wrapped as a WAV file, for an API that wants a file.
pub fn wav(pcm: &[u8], rate: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + pcm.len());
    let data_len = pcm.len() as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// (mono 16-bit PCM, sample rate) from a WAV file. Stereo is mixed down.
pub fn unwav(data: &[u8]) -> Result<(Vec<u8>, u32)> {
    let bad = |m: &str| Error::config(format!("not a WAV file: {m}"));
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(bad("no RIFF/WAVE header"));
    }
    let (mut channels, mut rate, mut bits) = (0u16, 0u32, 0u16);
    let mut pcm: Option<&[u8]> = None;
    let mut at = 12;
    while at + 8 <= data.len() {
        let id = &data[at..at + 4];
        let size = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]) as usize;
        let body = &data[at + 8..(at + 8 + size).min(data.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                channels = u16::from_le_bytes([body[2], body[3]]);
                rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                bits = u16::from_le_bytes([body[14], body[15]]);
            }
            b"data" => pcm = Some(body),
            _ => {}
        }
        at += 8 + size + (size % 2);
    }
    let pcm = pcm.ok_or_else(|| bad("no data chunk"))?;
    if bits != 16 {
        return Err(Error::config(format!("only 16-bit WAV is read here, not {bits}-bit")));
    }
    if channels > 1 {
        let all = samples(pcm);
        let c = channels as usize;
        let mixed: Vec<i16> = all.chunks_exact(c).map(|frame| (frame.iter().map(|s| *s as i32).sum::<i32>().div_euclid(c as i32)) as i16).collect();
        return Ok((to_bytes(&mixed), rate));
    }
    Ok((pcm.to_vec(), rate))
}

// ---- G.711 µ-law: what a phone line carries ---------------------------------
const BIAS: i32 = 0x84;
const CLIP: i32 = 32635;

fn ulaw_sample(byte: u8) -> i16 {
    let value = !byte as i32 & 0xFF;
    let mut sample = (((value & 0x0F) << 3) + BIAS) << ((value & 0x70) >> 4);
    sample -= BIAS;
    (if value & 0x80 != 0 { -sample } else { sample }) as i16
}

/// 8-bit µ-law to 16-bit PCM.
pub fn ulaw_decode(data: &[u8]) -> Vec<u8> {
    to_bytes(&data.iter().map(|b| ulaw_sample(*b)).collect::<Vec<_>>())
}

/// 16-bit PCM to 8-bit µ-law.
pub fn ulaw_encode(pcm: &[u8]) -> Vec<u8> {
    samples(pcm)
        .into_iter()
        .map(|sample| {
            let s = sample as i32;
            let sign = if s < 0 { 0x80 } else { 0 };
            let magnitude = s.abs().min(CLIP) + BIAS;
            let bit_length = 32 - magnitude.leading_zeros() as i32;
            let exponent = (bit_length - 8).max(0);
            let mantissa = (magnitude >> (exponent + 3)) & 0x0F;
            (!(sign | (exponent << 4) | mantissa) & 0xFF) as u8
        })
        .collect()
}

// ---- signals to test with -----------------------------------------------------
/// A steady tone — a stand-in for a voice, for tests and examples.
pub fn tone(ms: f64, rate: u32, frequency: f64, amplitude: i32) -> Vec<u8> {
    let count = (rate as f64 * ms / 1000.0) as usize;
    let data: Vec<i16> = (0..count).map(|n| (amplitude as f64 * (2.0 * std::f64::consts::PI * frequency * n as f64 / rate as f64).sin()) as i16).collect();
    to_bytes(&data)
}

/// A 220 Hz tone at 16 kHz — Python's `tone(ms)` defaults.
pub fn voice_like(ms: f64) -> Vec<u8> {
    tone(ms, 16_000, 220.0, 9_000)
}

/// Quiet, optionally with a little room noise in it.
pub fn silence(ms: f64, rate: u32, noise: i32) -> Vec<u8> {
    let count = (rate as f64 * ms / 1000.0) as usize;
    if noise == 0 {
        return vec![0; count * 2];
    }
    // A fixed pattern rather than randomness, so a test hears the same room twice.
    let data: Vec<i16> = (0..count).map(|n| ((n as i64 * 7919) % (2 * noise as i64 + 1) - noise as i64) as i16).collect();
    to_bytes(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_formats_and_rates() {
        assert_eq!(rms(&silence(20.0, 16_000, 0)), 0.0);
        assert!(rms(&voice_like(20.0)) > 6_000.0);
        assert_eq!(duration_ms(&voice_like(250.0), 16_000), 250.0);
        let pcm = voice_like(100.0);
        let (back, rate) = unwav(&wav(&pcm, 16_000)).unwrap();
        assert_eq!((back, rate), (pcm.clone(), 16_000));
        // µ-law round trips to within its quantisation.
        let round = samples(&ulaw_decode(&ulaw_encode(&pcm)));
        assert!(samples(&pcm).iter().zip(&round).all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1 + (*a as i32).abs() / 16));
        // Chunked resampling is the same as resampling the whole.
        let whole = resample(&pcm, 16_000, 24_000).unwrap();
        let mut r = Resampler::new(16_000, 24_000).unwrap();
        let mut pieces = Vec::new();
        for chunk in pcm.chunks(321) {
            pieces.extend(r.feed(chunk));
        }
        assert_eq!(pieces, whole);
        assert!(Resampler::new(0, 1).is_err());
    }

    #[test]
    fn matches_python_byte_for_byte() {
        // ulaw_encode(tone(1)) and resample(tone(2), 16000, 8000) in Python.
        assert_eq!(ulaw_encode(&voice_like(1.0)), PY_ULAW);
        assert_eq!(samples(&resample(&voice_like(2.0), 16_000, 8_000).unwrap()), PY_RESAMPLED);
        assert_eq!(samples(&silence(1.0, 16_000, 3)), PY_NOISE);
    }

    const PY_ULAW: &[u8] = &[255, 211, 197, 188, 183, 177, 174, 171, 169, 166, 164, 162, 161, 159, 159, 158];
    const PY_RESAMPLED: &[i16] = &[0, 1547, 3048, 4459, 5736, 6843, 7746, 8418, 8840, 8998, 8889, 8514, 7886, 7023, 5951, 4702];
    const PY_NOISE: &[i16] = &[-3, -1, 1, 3, -2, 0, 2, -3, -1, 1, 3, -2, 0, 2, -3, -1];
}
