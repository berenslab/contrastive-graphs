# -*- mode: sh -*-
exec >&2


redo-ifchange fig1.tex ../bin/texlive

TMPDIR=$(mktemp --directory)
OLDPWD=$PWD
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
\definecolor{myred1}{RGB}{255, 0, 0}

\begin{document}
\input{$OLDPWD/fig1.tex}
\end{document}
EOF

$OLDPWD/../bin/tex/texlive/2025/bin/x86_64-linux/pdflatex fig1.tex
cd $OLDPWD
mv $TMPDIR/fig1.pdf $3
