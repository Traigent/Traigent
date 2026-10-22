# Historical Git hook templates

These files are the exact bytes of every hook script that earlier versions of
`traigent/hooks/installer.py` could have written. `tests/unit/hooks` uses them
to check that `traigent hooks install` upgrades them and `traigent hooks
uninstall` removes them. A hook counts as Traigent's only when its bytes equal
one that Traigent generated. The tests never use the installer's own
classifier to decide what a fixture is.

Do not edit, re-encode or normalise these files. The tests pin their SHA-256
digests, so a line-ending or whitespace rewrite fails loudly.

## Extraction

Every commit that touched the installer was listed with
`git log --all --format=%H -- traigent/hooks/installer.py`. That is 9 commits.
For each commit, `git show <commit>:traigent/hooks/installer.py` was parsed
with `ast` (no historical code was imported or run). The script took the string
constants assigned to `PRE_PUSH_HOOK_SCRIPT` and `PRE_COMMIT_HOOK_SCRIPT` and
encoded them as UTF-8. Every literal is ASCII. The extraction ran as:

```
python3 -I -S <extract-templates-script> <repo> --write-fixtures tests/fixtures/hooks
```

The script is kept with the change's local evidence, not in this repository.
Regenerating these fixtures does not need it: the procedure above is the whole
specification, and the digests below pin the result either way.

Result: exactly two template pairs across the SDK's history. The first of
them is still what the installer ships today; the second is superseded.

| File | Kind | Bytes | SHA-256 | Revisions with this literal |
|---|---|---|---|---|
| `pre-push.18bfcfdfe7e3.lf.hook` | pre-push | 924 | `18bfcfdfe7e36238286e4aeeb65bdef3e04660c58b477f0856a40e640e26095c` | `45c067ff`, `0320568b`, `c59cada2`, `c517a91f`, `729eea82` |
| `pre-commit.b7b26bf4e881.lf.hook` | pre-commit | 454 | `b7b26bf4e881ce5923c3d2d506488e6e0deec538afde892b58333f3d28319ee0` | same five |
| `pre-push.bdf3be4f4ae4.lf.hook` | pre-push | 924 | `bdf3be4f4ae44aa15100ecc6a3409b448ea442b1a048bb5816c0fc0d5a8a069c` | `26902cf5`, `1d8ca986`, `f6be5a66`, `7489260a` |
| `pre-commit.105f68f2146a.lf.hook` | pre-commit | 454 | `105f68f2146a23e5a0cfae9ad5459d9721f55e1cd2127a66c26e7fb5e07c96e9` | same four |

Full revision ids:

- `45c067ff18284dc3c705f448e6a34daa4a04ebd4`
- `0320568b2e7c740419425683c00c437c85ab3bb7`
- `c59cada23d911dee8959b3efc265d6830bcb1d0d`
- `c517a91fd06181188f5d3f880111d8a03fba484e`
- `729eea8203db2654fcd8c29b71ac23c484801be7`
- `26902cf557a7cec241a5b2c85e240a867ddb7673`
- `1d8ca9867808c842758387a88d9fc64f955b76eb`
- `f6be5a667fccd2090d88bb342f82f2dd9e8f0872`
- `7489260abb965d902124cb3200796c75b336d12d`

Every `v*` tag that contains the installer was checked by AST-extracting
`git show <tag>:traigent/hooks/installer.py`. Tags `v0.9.0` through `v0.30.0`
ship the `18bfcfdf…`/`b7b26bf4…` pair. The pre-release tag `v0.8.0-rc1` ships
the `bdf3be4f…`/`105f68f2…` pair, which has the `TraiGent` header spelling. No
tag ships any other template.

## CRLF derivations

Both historical writers used text mode with the default newline handling:

- `7489260a` used `hook_path.write_text(content)`.
- `45c067ff` and later used `safe_write_text`, which calls
  `Path.write_text(content, encoding=...)`.

Per the Python documentation for `open()`
(<https://docs.python.org/3.11/library/functions.html#open>), when text is
written with `newline=None`, every `"\n"` is translated to the platform
separator `os.linesep`. On Windows that separator is `"\r\n"`, so an earlier
SDK on Windows wrote the CRLF form of these exact templates.

Each `*.crlf.hook` file is exactly `lf_bytes.replace(b"\n", b"\r\n")`, and the
LF sources contain no `\r`. No other normalisation was applied.

| File | Derived from | Bytes | SHA-256 |
|---|---|---|---|
| `pre-push.18bfcfdfe7e3.crlf.hook` | `pre-push.18bfcfdfe7e3.lf.hook` | 960 | `b9d8398357c53b8b128ba0ca6d275a804cd0f674f94037a84333f673605d5b72` |
| `pre-commit.b7b26bf4e881.crlf.hook` | `pre-commit.b7b26bf4e881.lf.hook` | 472 | `9ddbae8ca0016f02820662bb5a18af416c938db7fab74614f5b98c515ab4ef18` |
| `pre-push.bdf3be4f4ae4.crlf.hook` | `pre-push.bdf3be4f4ae4.lf.hook` | 960 | `5ea26895d95b571ca10ee0625d0e7b44f1d312f378d92cd0c80255cd4dd2b092` |
| `pre-commit.105f68f2146a.crlf.hook` | `pre-commit.105f68f2146a.lf.hook` | 472 | `da32c293794c1a030111120ab5a85950c8268bf61bd5ff65673f123a9db96c82` |

The CRLF forms of BOTH pairs are listed, including the pair the installer
still ships. That is not a contradiction: the current installer writes raw LF
bytes, but the historical text-mode writers described above emitted the CRLF
derivation of those same bytes on Windows, so those bytes are genuinely ours
to recognise.

What the installer's `_LEGACY_DIGESTS` deliberately omits is the **LF** form
of the current pair. Those two digests live in `_CURRENT_TEMPLATES`, which the
classifier consults first; listing them in both places would make "installed
(traigent)" and "installed (traigent, outdated)" describe the same bytes.

## Current templates

The current pair IS the `18bfcfdf…` / `b7b26bf4…` pair above, so it is stored
here like the rest of the history.

| Kind | Bytes | SHA-256 |
|---|---|---|
| pre-push | 924 | `18bfcfdfe7e36238286e4aeeb65bdef3e04660c58b477f0856a40e640e26095c` |
| pre-commit | 454 | `b7b26bf4e881ce5923c3d2d506488e6e0deec538afde892b58333f3d28319ee0` |

Changing a template requires adding the previous bytes here, with their LF and
CRLF digests and revisions, and in the installer's known set. The pinned
digest test in `tests/unit/hooks/test_installer.py` fails until both are
updated.
