//! Voice activity detection: when someone starts speaking, and when they stop.
//!
//! ```
//! use agent_harness::voice::{EnergyVAD, VAD, VadKind};
//! let mut vad = EnergyVAD::new(16_000);
//! for event in vad.feed(&agent_harness::voice::audio::voice_like(400.0)) {
//!     if event.kind == VadKind::Start { /* they began — stop talking over them */ }
//! }
//! ```
//!
//! This is what makes a voice agent feel like a conversation. Stopping too
//! early cuts people off mid-sentence; too late, and every reply arrives after
//! an awkward pause. `silence_ms` is that dial.
//!
//! [`EnergyVAD`] listens to loudness against the room, which it keeps
//! measuring as the quietest tenth of the last few seconds — speech has gaps in
//! it, a fan does not — so it needs nothing installed and copes with a steady
//! hum. It can be fooled by a door slamming. For a noisy room, anything that
//! implements [`VAD`] — a neural detector such as Silero — plugs in instead.

use std::collections::VecDeque;

use crate::errors::{Error, Result};
use crate::voice::audio::rms;

/// What a detector heard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadKind {
    /// Speech began.
    Start,
    /// Speech stopped; the event carries the utterance.
    End,
    /// What started turned out to be a noise, not a turn.
    Cancel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadEvent {
    pub kind: VadKind,
    /// Position in the stream.
    pub at_ms: f64,
    /// The whole utterance, on `End`.
    pub audio: Vec<u8>,
}

/// Anything that can be fed audio and say when speech starts and ends.
pub trait VAD: Send {
    fn rate(&self) -> u32;
    fn feed(&mut self, pcm: &[u8]) -> Vec<VadEvent>;
    /// The stream has ended: whoever was mid-sentence has finished it.
    fn flush(&mut self) -> Vec<VadEvent>;
    fn reset(&mut self);
    fn speaking(&self) -> bool;
}

/// Python's `round()`: halves go to the even neighbour.
fn py_round(x: f64) -> usize {
    let r = x.round();
    let r = if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 { r - x.signum() } else { r };
    r.max(0.0) as usize
}

/// How an [`EnergyVAD`] listens. Every field has a working default.
#[derive(Debug, Clone)]
pub struct VadOptions {
    /// How finely it listens.
    pub frame_ms: u32,
    /// How much continuous speech it takes to believe someone has started.
    pub start_ms: u32,
    /// How long a pause ends the turn. 500 suits conversation.
    pub silence_ms: u32,
    /// An "utterance" shorter than this is a noise, and dropped.
    pub min_speech_ms: u32,
    /// Audio kept from just before speech was detected, so the first consonant is not clipped.
    pub prefix_ms: u32,
    /// Nobody is left talking for ever; past this the turn ends.
    pub max_utterance_ms: u32,
    /// How many times louder than the room speech has to be.
    pub ratio: f64,
    /// The quietest level that can count as speech at all.
    pub floor: f64,
    /// How far back it listens to judge how loud the room is.
    pub room_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self { frame_ms: 20, start_ms: 80, silence_ms: 500, min_speech_ms: 160, prefix_ms: 240, max_utterance_ms: 30_000, ratio: 3.0, floor: 250.0, room_ms: 3000 }
    }
}

/// Speech by loudness, against a noise floor that adapts.
#[derive(Debug, Clone)]
pub struct EnergyVAD {
    pub rate: u32,
    pub frame_ms: u32,
    pub frame_bytes: usize,
    start_frames: usize,
    silence_frames: usize,
    min_speech_frames: usize,
    max_frames: usize,
    pub ratio: f64,
    pub floor: f64,
    prefix_cap: usize,
    levels_cap: usize,
    rest: Vec<u8>,
    noise: f64,
    levels: VecDeque<f64>,
    frames: u64,
    speaking: bool,
    loud: usize,
    quiet: usize,
    voiced: usize,
    utterance: Vec<Vec<u8>>,
    prefix: VecDeque<Vec<u8>>,
}

impl EnergyVAD {
    /// With the defaults, at this sample rate.
    pub fn new(rate: u32) -> Self {
        Self::with_options(rate, VadOptions::default()).expect("the defaults are valid")
    }

    pub fn with_options(rate: u32, o: VadOptions) -> Result<Self> {
        if rate == 0 || o.frame_ms == 0 {
            return Err(Error::config("rate and frame_ms must be positive"));
        }
        let per = |ms: u32| py_round(ms as f64 / o.frame_ms as f64);
        Ok(Self {
            rate,
            frame_ms: o.frame_ms,
            frame_bytes: (rate as u64 * o.frame_ms as u64 / 1000) as usize * 2,
            start_frames: per(o.start_ms).max(1),
            silence_frames: per(o.silence_ms).max(1),
            min_speech_frames: per(o.min_speech_ms).max(1),
            max_frames: per(o.max_utterance_ms).max(1),
            ratio: o.ratio,
            floor: o.floor,
            prefix_cap: per(o.prefix_ms).max(1),
            levels_cap: per(o.room_ms).max(10),
            rest: vec![],
            noise: 0.0,
            levels: VecDeque::new(),
            frames: 0,
            speaking: false,
            loud: 0,
            quiet: 0,
            voiced: 0,
            utterance: vec![],
            prefix: VecDeque::new(),
        })
    }

    /// How loud a frame has to be, right now, to count as speech.
    pub fn threshold(&self) -> f64 {
        self.floor.max(self.noise * self.ratio)
    }

    fn frame(&mut self, frame: Vec<u8>) -> Vec<VadEvent> {
        self.frames += 1;
        let level = rms(&frame);
        // The room is the quietest tenth of what was recently heard. Speech is
        // full of gaps that fall back to it; a hum has none, so the hum *is*
        // the room.
        if self.levels.len() == self.levels_cap {
            self.levels.pop_front();
        }
        self.levels.push_back(level);
        let mut ranked: Vec<f64> = self.levels.iter().copied().collect();
        ranked.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.noise = ranked[ranked.len() / 10];
        let loud = level > self.threshold();
        let mut events = vec![];
        if !self.speaking {
            if self.prefix.len() == self.prefix_cap {
                self.prefix.pop_front();
            }
            self.prefix.push_back(frame);
            self.loud = if loud { self.loud + 1 } else { 0 };
            if self.loud >= self.start_frames {
                self.speaking = true;
                self.quiet = 0;
                self.voiced = self.loud;
                self.utterance = self.prefix.drain(..).collect();
                events.push(VadEvent { kind: VadKind::Start, at_ms: (self.frames - self.loud as u64) as f64 * self.frame_ms as f64, audio: vec![] });
            }
            return events;
        }
        self.utterance.push(frame);
        if loud {
            self.voiced += 1;
            self.quiet = 0;
        } else {
            self.quiet += 1;
        }
        if self.quiet >= self.silence_frames || self.utterance.len() >= self.max_frames {
            events.extend(self.finish());
        }
        events
    }

    fn finish(&mut self) -> Vec<VadEvent> {
        let voiced = self.voiced;
        let frames = std::mem::take(&mut self.utterance);
        self.speaking = false;
        self.loud = 0;
        self.quiet = 0;
        self.voiced = 0;
        let at_ms = self.frames as f64 * self.frame_ms as f64;
        if voiced < self.min_speech_frames {
            // A click, a cough: not a turn — and whoever stopped to listen
            // should be told they can carry on.
            return vec![VadEvent { kind: VadKind::Cancel, at_ms, audio: vec![] }];
        }
        // The trailing silence that ended the turn is not part of what was
        // said, bar a short tail so the last word is not clipped.
        let keep = frames.len() - self.tail(&frames).min(frames.len());
        vec![VadEvent { kind: VadKind::End, at_ms, audio: frames[..keep].concat() }]
    }

    /// How many frames of the closing silence to drop (keeping 100 ms of it).
    fn tail(&self, frames: &[Vec<u8>]) -> usize {
        let threshold = self.threshold();
        let quiet = frames.iter().rev().take_while(|f| rms(f) <= threshold).count();
        quiet.saturating_sub(5)
    }
}

impl VAD for EnergyVAD {
    fn rate(&self) -> u32 {
        self.rate
    }

    fn feed(&mut self, pcm: &[u8]) -> Vec<VadEvent> {
        let mut data = std::mem::take(&mut self.rest);
        data.extend_from_slice(pcm);
        let size = self.frame_bytes.max(2);
        let whole = data.len() - data.len() % size;
        self.rest = data[whole..].to_vec();
        let mut events = vec![];
        for chunk in data[..whole].chunks(size) {
            events.extend(self.frame(chunk.to_vec()));
        }
        events
    }

    fn flush(&mut self) -> Vec<VadEvent> {
        let events = if self.speaking { self.finish() } else { vec![] };
        self.rest.clear();
        events
    }

    fn reset(&mut self) {
        self.rest.clear();
        self.noise = 0.0;
        self.levels.clear();
        self.frames = 0;
        self.speaking = false;
        self.loud = 0;
        self.quiet = 0;
        self.voiced = 0;
        self.utterance.clear();
        self.prefix.clear();
    }

    fn speaking(&self) -> bool {
        self.speaking
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::audio::{silence, voice_like};

    #[test]
    fn a_turn_starts_ends_and_a_cough_is_cancelled() {
        let mut vad = EnergyVAD::new(16_000);
        let mut events = vad.feed(&silence(300.0, 16_000, 40));
        events.extend(vad.feed(&voice_like(600.0)));
        assert_eq!(events.iter().map(|e| e.kind).collect::<Vec<_>>(), vec![VadKind::Start]);
        assert_eq!(events[0].at_ms, 300.0);
        let ended = vad.feed(&silence(700.0, 16_000, 40));
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].kind, VadKind::End);
        // The utterance: the prefix, the speech, and 100 ms of the closing quiet.
        let ms = crate::voice::audio::duration_ms(&ended[0].audio, 16_000);
        assert!((600.0..=1000.0).contains(&ms), "{ms}");
        let mut events = vad.feed(&voice_like(100.0));
        events.extend(vad.feed(&silence(600.0, 16_000, 40)));
        assert_eq!(events.iter().map(|e| e.kind).collect::<Vec<_>>(), vec![VadKind::Start, VadKind::Cancel]);
        vad.feed(&voice_like(400.0));
        assert!(vad.speaking());
        assert_eq!(vad.flush()[0].kind, VadKind::End);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
    }
}
