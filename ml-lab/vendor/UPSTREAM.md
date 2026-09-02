# Frozen CoreField reference

- (a) Source repository: read-only sibling `CoreField Startup`.
- (a) Source commit: `8219c99088645b7df984752e099a3f873bae773b` (`field-validation`, 1 Sep 2026).
- (a) Copy scope: the thirteen top-level Python modules in `corefield/`, plus the upstream Apache-2.0 `LICENSE` and `NOTICE`.
- (a) Copy method: byte-for-byte copy; no source file was edited. SHA-256 values are in `manifest.json` and are verified by the harness tests.
- (a) The sibling repository remained clean after copying.
- (c) Treat this directory as read-only. Experimental code belongs under `src/corefield_ml_lab/`.

The upstream implementation warns that its detailed IEC 60076-7 equations/constants were mirror-sourced and remain unverified against a licensed copy. Vendoring preserves that warning; it does not convert the implementation into a standards-compliance claim.

