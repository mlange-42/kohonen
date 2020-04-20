# kohonen
A Rust Self-organizing Maps (SOM) / Kohonen networks library and command line tool.

_Warning:_ This project is in a very experimental state.

## Self-organizing maps

* [SOM on Wikipedia](https://en.wikipedia.org/wiki/Self-organizing_map)
* For Super-SOMs, see [this paper](https://www.jstatsoft.org/article/view/v021i05) about a respective R package.

## Command line tool

### Installation

* Download the [latest binaries](https://github.com/mlange-42/kohonen/releases).
* Unzip somewhere with write privileges (only required for running examples in place).

### Usage

* Try the examples in sub-directory [`/cmd_examples`](https://github.com/mlange-42/chrono-photo/tree/master/cmd_examples).
* To view the full list of options, run `kohonen --help`

## Library / crate

To use this crate as a library, add the following to your `Cargo.toml` dependencies section:
```
kohonen = { git = "https://github.com/mlange-42/kohonen.git" }
```
See the included examples for usage details.

## Development version

For the latest development version, see branch [`dev`](https://github.com/mlange-42/kohonen/tree/dev).