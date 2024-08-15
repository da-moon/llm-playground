# Reason for This Change

Elaborate on the reasons for this change. If applicable link it to the related
issue (like `Closes #<issue>`).

## Description of Changes

Overview of what was changed. While on your **feature branch**:

- run the following snippet to get a list of changed files and provide a short
  description on what changed in each file

```bash
git diff --name-only  "$(git merge-base $(git symbolic-ref --short refs/remotes/origin/HEAD) "$(git for-each-ref --format='%(upstream:short)' "$(git symbolic-ref -q HEAD)")").."$(git for-each-ref --format='%(upstream:short)' "$(git symbolic-ref -q HEAD)")""
```

- detect changes between each file that was listed with previous snippet and
  the **default** branch:

```bash
git diff  "$(git for-each-ref --format='%(upstream:short)' "$(git symbolic-ref -q HEAD)")" "$(git symbolic-ref --short refs/remotes/origin/HEAD)" -- "<file-to-check>"
```

- Give explanation of those changes so PR reviewer knows why/how a change was
  made

For running **diffs** between files you can use the following tools as Git's
default differ leaves a lot to be desired:

- [`git-delta`][1]: a great general purpose differ. Use the following snippet
  after installing it for setting some sane defaults

```bash
git config --global pager.diff delta ;
git config --global pager.log delta ;
git config --global pager.reflog delta ;
git config --global pager.show delta ;
git config --global interactive.difffilter "delta --color-only --features=interactive" ;
git config --global delta.features "side-by-side line-numbers decorations" ;
git config --global delta.whitespace-error-style "22 reverse" ;
git config --global delta.decorations.commit-decoration-style "bold yellow box ul" ;
git config --global delta.decorations.file-style "bold yellow ul" ;
git config --global delta.decorations.file-decoration-style "none" ;
git config --global delta.decorations.commit-style "raw" ;
git config --global delta.decorations.hunk-header-decoration-style "blue box" ;
git config --global delta.decorations.hunk-header-file-style "red" ;
git config --global delta.decorations.hunk-header-line-number-style "#067a00" ;
git config --global delta.decorations.hunk-header-style "file line-number syntax" ;
git config --global delta.interactive.keep-plus-minus-markers "false" ;
```

- For diffing Terraform files, you can use a differ with `tree-sitter`
  integration instead of using [`git-delta`][1]. These differs **understand**
  HCL/Terraform file `AST` and show you exactly what has changed instead of a
  generic side-by-side diff that you get out of [`git-delta`][1]:
  - [diffsitter][2]
  - [difftastic][3]

[1]: https://dandavison.github.io/delta
[2]: https://github.com/afnanenayet/diffsitter
[3]: https://github.com/Wilfred/difftastic
