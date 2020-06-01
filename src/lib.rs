//! Self-organizing maps / Kohonen maps with an arbitrary amount of layers (Super-SOMs).

pub mod calc;
pub mod cli;
pub mod data;
pub mod map;
pub mod proc;
pub mod ui;

use crate::cli::CliParsed;
use crate::map::som::Som;
use crate::proc::Processor;
use core::fmt;
use std::fs::File;
use std::io::Write;
/*
pub trait EnumFromString {
    /// Parses a string to an `enum`.
    fn from_string(str: &str) -> Result<Self, ParseEnumError>
    where
        Self: std::marker::Sized;
}
*/
/// Error type for failed parsing of `String`s to `enum`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseEnumError(String);

impl fmt::Display for ParseEnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Error type for wrong data type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTypeError(String);

impl fmt::Display for DataTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub fn write_output(parsed: &CliParsed, proc: &Processor, som: &Som) {
    if let Some(out) = &parsed.output {
        let units_file = format!("{}-units.csv", &out);
        proc.write_som_units(&som, &units_file, true).unwrap();
        let data_file = format!("{}-out.csv", &out);
        proc.write_data_nearest(&som, proc.data(), &data_file)
            .unwrap();
        let norm_file = format!("{}-norm.csv", &out);
        proc.write_normalization(&som, &norm_file).unwrap();

        let som_file = format!("{}-som.json", &out);
        let serialized = serde_json::to_string_pretty(&(som, proc.denorm())).unwrap();
        let mut file = File::create(som_file).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }
}
