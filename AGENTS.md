# Sandy 3

## Overview

A GPU sand simulation.

## Tests

Always write tests when needed (only when needed though; don't write tests for something as trivial as, say, a hello world).

## Code Quality

Always lint and format:

```bash
cargo fmt && cargo clippy
```

If a clippy error already existed, you don't need to fix it, but do note it.

## Toolchain

You are permitted to write nightly Rust, and note that the dev build uses the Cranelift backend.

## Outdated packages

PLEASE, PLEASE do not use a package that is old or deprecated. When possible use the latest version.

## Keep things human

Do not use em dashes or other special symbols not normally found in writing. Do not word things in a weird way. Keep it looking human.

## Plugins

Prefer to write new tools, brushes, materials, or worlds in JavaScript as a built-in plugin when possible.

## README (IMPORTANT!!!)

Always update the README if necessary (again, not if unnecessary) when changing or adding code. If the README already contain outdated information, you don't need to fix it, but do note it.