//! A small, safe expression language for policy conditions.
//!
//! ```
//! use agent_harness::governance::compile_expr;
//! use serde_json::json;
//!
//! let test = compile_expr("args.amount > 100 and 'payments' in tool.tags").unwrap();
//! let ctx = json!({"args": {"amount": 250}, "tool": {"tags": ["payments"]}});
//! assert_eq!(test.eval(ctx.as_object().unwrap()), json!(true));
//! ```
//!
//! Compliance people read these, so they look like Python and behave like it,
//! but nothing is ever executed as code: there is no access to anything
//! starting with an underscore, and no function can be called except the
//! listed ones. Two rules make policies safe to write against loose data:
//!
//! * a missing name or field is `null`, never an error;
//! * a comparison that cannot be made (`null > 100`) is `false`.
//!
//! The language itself lives in [`crate::expr`] (workflows use it too); this
//! module is where governance names it.

use std::sync::Arc;

pub use crate::expr::{Ctx, Expr, FUNCTIONS, truthy};

use crate::errors::Result;

/// A policy condition that does not parse, or uses something not allowed. In
/// Rust it is an `Error::Configuration`, as Python's `ExpressionError` is a
/// `ConfigurationError`.
pub type ExpressionError = crate::errors::Error;

/// Parse and compile once. The result is evaluated with `.eval(&ctx)`.
pub fn compile_expr(source: &str) -> Result<Arc<Expr>> {
    crate::expr::compile(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expressions_read_loose_data_safely() {
        let ctx = json!({"args": {"amount": 250, "to": "x"}, "tool": {"tags": ["payments"]}});
        let ctx = ctx.as_object().unwrap();
        for source in [
            "args.amount > 100 and 'payments' in tool.tags",
            "args['amount'] >= 250",
            "missing.field is None",
            "len(tool.tags) == 1 and lower('AB') == 'ab'",
            "matches(args.to, '^x$') and not false",
            "1 < args.amount < 300",
        ] {
            assert!(truthy(&compile_expr(source).unwrap().eval(ctx)), "{source}");
        }
        assert_eq!(compile_expr("missing.field > 5").unwrap().eval(ctx), json!(false));
    }

    #[test]
    fn expressions_refuse_anything_dangerous() {
        for source in ["__import__('os')", "args.__class__", "open('x')", "(lambda: 1)()", "[x for x in args]", "args.amount if"] {
            assert!(compile_expr(source).is_err(), "{source}");
        }
    }
}
