//! The command line. There is not much of one: the game opens a window and
//! takes no arguments, and everything else is about running a control
//! script (see [`crate::scripting`]), in the window or without one.

use std::path::PathBuf;

use crate::scripting::Source;

pub const USAGE: &str = "\
Usage: sandy-3 [--headless] [SCRIPT.lua | -e CODE | -]

With no arguments, opens the window.

  SCRIPT.lua    run this control script, in the window unless --headless
  -e CODE       run CODE as the script
  -             read the script from standard input
  --headless    no window: run the script and exit, 1 if it failed
  -h, --help    show this";

/// What the command line asked for.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    /// Open the window, and run a script in it if one was given.
    Window {
        script: Option<Source>,
    },
    /// Run a script with no window.
    Headless {
        script: Source,
    },
    Help,
}

/// Read the arguments, without the program's own name.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut headless = false;
    let mut script: Option<Source> = None;
    let mut args = args.into_iter();
    let mut set = |source: Source| -> Result<(), String> {
        if script.is_some() {
            return Err("only one script can be run".to_string());
        }
        script = Some(source);
        Ok(())
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--headless" => headless = true,
            "-e" => {
                let code = args.next().ok_or("-e needs the code to run after it")?;
                set(Source::Inline(code))?;
            }
            "-" => set(Source::Stdin)?,
            other if other.starts_with('-') => return Err(format!("unknown option '{other}'")),
            path => set(Source::File(PathBuf::from(path)))?,
        }
    }
    match (headless, script) {
        (true, Some(script)) => Ok(Command::Headless { script }),
        (true, None) => Err("--headless needs a script to run".to_string()),
        (false, script) => Ok(Command::Window { script }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn the_arguments_say_what_to_run_and_where() {
        assert_eq!(parse_all(&[]).unwrap(), Command::Window { script: None });
        assert_eq!(
            parse_all(&["demo.lua"]).unwrap(),
            Command::Window {
                script: Some(Source::File(PathBuf::from("demo.lua")))
            }
        );
        assert_eq!(
            parse_all(&["--headless", "demo.lua"]).unwrap(),
            Command::Headless {
                script: Source::File(PathBuf::from("demo.lua"))
            }
        );
        assert_eq!(
            parse_all(&["-e", "sim.step()", "--headless"]).unwrap(),
            Command::Headless {
                script: Source::Inline("sim.step()".to_string())
            }
        );
        assert_eq!(
            parse_all(&["--headless", "-"]).unwrap(),
            Command::Headless {
                script: Source::Stdin
            }
        );
        assert_eq!(parse_all(&["-h"]).unwrap(), Command::Help);
        assert_eq!(parse_all(&["demo.lua", "--help"]).unwrap(), Command::Help);
    }

    #[test]
    fn a_bad_command_line_says_why() {
        assert!(parse_all(&["--headless"]).unwrap_err().contains("script"));
        assert!(parse_all(&["-e"]).unwrap_err().contains("-e"));
        assert!(parse_all(&["--wat"]).unwrap_err().contains("--wat"));
        assert!(parse_all(&["a.lua", "b.lua"]).unwrap_err().contains("one"));
        assert!(parse_all(&["-e", "x", "-"]).unwrap_err().contains("one"));
    }
}
