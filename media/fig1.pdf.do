# -*- mode: sh -*-
set -e

exec >&2


redo-ifchange fig1.tex ../bin/texlive

TMPDIR=$(mktemp --directory)
OLDPWD=$PWD
COLORDEFS=$(grep "^%%.*definecolor" fig1.tex | tr -d '% ')
cd $TMPDIR

cat > fig1.tex <<EOF
\documentclass{standalone}
%% Make sure the required packages are loaded in your preamble
\usepackage{pgf}
%%
%% Also ensure that all the required font packages are loaded; for instance,
%% the lmodern package is sometimes necessary when using math font.
\usepackage{lmodern}

\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
$COLORDEFS
\ifdefined\pdftexversion\else  % non-pdftex case.
  \usepackage{fontspec}
  \setsansfont{RobotoCondensed-Regular.ttf}[Path=\detokenize{/home/jnb/dev/berenslab/graphs/media/fonts/ttf/roboto/}]
  \setmonofont{Iosevka-Regular.ttf}[Path=\detokenize{/home/jnb/dev/berenslab/graphs/media/fonts/ttf/iosevka/}]
\fi

\begin{document}
\input{$OLDPWD/fig1.tex}
\end{document}
EOF

$OLDPWD/../bin/tex/texlive/2025/bin/x86_64-linux/xelatex fig1.tex > /dev/null 2>/dev/null
cd $OLDPWD
mv $TMPDIR/fig1.pdf $3
