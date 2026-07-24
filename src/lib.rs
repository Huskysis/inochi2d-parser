//! Typed parser and intermediate representation (IR) for the Inochi2D INP/INX
//! puppet format, plus the INR engine-facing runtime format (feature `inr`).
//!
//! ```no_run
//! use inochi2d_parser::prelude::Puppet;
//!
//! let puppet = Puppet::open("path/to/file.inp")?;
//! let root = &puppet.nodes;
//! println!("{} top-level children", root.children.len());
//! # Ok::<(), std::io::Error>(())
//! ```

#![warn(missing_docs)]

#[cfg(feature = "inr")]
pub mod inr;
/// Typed accessors on top of the `json` crate's `JsonValue`, used by the INX/INP parser.
pub mod json_extra;
/// The IR: [`owned::Puppet`] and its node/param/animation types.
pub mod owned;
/// Serializes an [`owned::Puppet`] back to INX/INP JSON bytes.
pub mod serialize;

/// Re-exports the IR types ([`owned::Puppet`] and friends) for glob import.
pub mod prelude {
    pub use crate::owned::*;
}
