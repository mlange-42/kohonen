//! Self-organizing maps / Kohonen maps with an arbitrary amount of layers (Super-SOMs).

pub mod calc;
pub mod cli;
pub mod data;
pub mod map;
pub mod proc;
pub mod ui;

use core::fmt;

/// Error type for failed parsing of `String`s to `enum`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseEnumError(String);

impl fmt::Display for ParseEnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
