"""Insert Supplementary Section S11 into supplementary.tex.

Idempotent: if the section heading already exists, the block is replaced.

Structure (three tables, no separate interpretation subsection):
    S11.1  design and metrics
    Fig. S11.1 / S11.2
    Table S11.1  per-scene metrics
    Table S11.2  (a) native vs alpha, (b) tolerance sensitivity
    Table S11.3  statistical analysis
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
SUP = os.path.join(_ROOT, "Computers & Geosciences", "supplementary", "supplementary.tex")
TAB = os.path.join(_HERE, "supp_tables_s11.tex")

HEAD = r"""
\section{Boundary-quality Evaluation on Controlled Geometries}
\label{sec:supp_boundary}
\setcounter{figure}{0}
\setcounter{table}{0}
\setcounter{equation}{0}

\subsection{Experimental design and boundary metrics}

Six controlled scenes isolate three sources of geometric difficulty: non-convex
shape (\textit{nonconvex\_c}, \textit{nonconvex\_u}), narrow inter-cluster gaps
(\textit{gap\_4}, \textit{gap\_2}) and heterogeneous density
(\textit{density\_3}, \textit{density\_8}). Each scene contains
$K_{\mathrm{gt}}=2$ clusters on a normalised $100\times100$ domain and was run
with 20 test seeds (0--19), giving 120 matched scene--seed observations per
method. All hyperparameters were fixed on an independent validation seed (9001)
and held constant for the test runs.

Every method is scored on the labels it produces. IZGC contributes two
representations: its accepted Marching Squares contours (\emph{native}) and the
same predicted labels re-polygonised with the alpha-shape applied to the
point-label baselines (\emph{alpha}). The alpha-shape discards Delaunay triangles
whose circumradius exceeds $1/\alpha$; $\alpha=0.05$, that is a threshold of 20
spatial units, was fixed before the test runs as a common geometric parameter
rather than tuned per scene. Because $\mathrm{IoU}_{\mathrm{pen}}$ is computed
from the rasterised labels, it is invariant to the polygonisation, so $\alpha$
was examined against boundary F1 on the validation seed instead. Averaged over
the five point-label baselines implemented in Python (K-means++, GMM, DBSCAN,
CLIQUE and BANG; DPC-FWSN runs through a MATLAB engine and was not part of this
scan), the mean BF was 0.870, 0.867, 0.864, 0.809 and 0.683 for
$\alpha=0.02$, 0.05, 0.10, 0.20 and 0.35. The first three differ by less than
0.01, so the evaluation is insensitive to the exact value in this range, whereas
coarser hulls deteriorate rapidly.

Boundary F1 is computed on discrete boundary point sets, sampled at about one
vertex per grid step (1.0 unit) along each polygon ring, with a matching
tolerance $\tau=1.5$ spatial units, that is 1.5\% of the scene side length:

\begin{equation}
\begin{aligned}
P_B &= \frac{|\{p\in B_p: d(p,B_g)\leq\tau\}|}{|B_p|}, &
R_B &= \frac{|\{q\in B_g: d(q,B_p)\leq\tau\}|}{|B_g|},\\
\mathrm{BF} &= \frac{2P_BR_B}{P_B+R_B},
\end{aligned}
\end{equation}

where $B_p$ and $B_g$ are the predicted and reference boundary point sets. A
predicted point counts as matched when its nearest reference-boundary point lies
within $\tau$, and recall applies the same nearest-neighbour criterion in the
reverse direction; matching is therefore distance-based rather than one-to-one,
so several predicted points may map to the same reference point. The pooled
95th-percentile bidirectional boundary distance ($\mathrm{HD95}$) combines the
nearest-neighbour distances of both directions before taking the 95th percentile
and is reported as a percentage of the scene diagonal. The region metric is

\begin{equation}
\mathrm{IoU}_{\mathrm{pen}}=
\frac{\sum_{(i,j)\in M}|G_i\cap P_j|}
{\sum_i|G_i|+\sum_{j\notin M}|P_j|},
\end{equation}

where $G_i$ and $P_j$ denote ground-truth and predicted regions and $M$ the
Hungarian-matched pairs. Every unmatched predicted region enters the denominator,
and missed ground-truth regions remain there through $\sum_i|G_i|$, so both error
types are penalised.

Table~\ref{tab:S11_scene_metrics} reports the per-scene metrics of the eight
methods, Table~\ref{tab:S11_native_alpha} compares the two boundary
representations of IZGC and the sensitivity of BF to the matching tolerance, and
Table~\ref{tab:S11_significance} collects the complete statistical analysis.
IZGC-Full is competitive on the gap and heterogeneous-density scenes, whereas the
U-shaped configuration is its weakest case: occasional over-segmentation lowered
the clustering consistency ($\mathrm{ARI}=0.708\pm0.423$, mean predicted cluster
number 2.90) and DBSCAN performed best there. The gap between the two boundary
representations is quantified in Table~\ref{tab:S11_native_alpha}(a) and
discussed in the main text.

"""

FIG = r"""
\begin{figure}[tbp]
	\centering
	\begin{subfigure}{0.82\textwidth}
		\centering
		\includegraphics[width=\linewidth]{Figure_S11.1(a).png}
		\caption{Ground-truth cluster memberships and reference boundaries.}
	\end{subfigure}

	\vspace{4pt}
	\begin{subfigure}{0.82\textwidth}
		\centering
		\includegraphics[width=\linewidth]{Figure_S11.1(b).png}
		\caption{Predicted cluster memberships of IZGC with the reference
		boundaries overlaid as black dashed lines.}
	\end{subfigure}
	\caption{Ground-truth geometries and illustrative IZGC clustering results on
	the six controlled scenes, ordered \textit{nonconvex\_c}, \textit{nonconvex\_u},
	\textit{gap\_4}, \textit{gap\_2}, \textit{density\_3} and \textit{density\_8}.
	All scenes contain two ground-truth clusters. The two non-convex scenes show a
	single additional cluster of 11--12 points in this realization
	($K_{\mathrm{pred}}=3$), a minor over-segmentation that leaves the ARI at 0.963
	and 0.966. Panels are single runs (seed 0) and illustrate the method rather
	than averages over the test seeds.}
	\label{fig:S11_geometry}
\end{figure}

\begin{figure}[tbp]
	\centering
	\includegraphics[width=0.96\textwidth]{Figure_S11.2.png}
	\caption{Native Marching Squares contours produced by IZGC on the six
	controlled scenes. Grey points are the input observations, coloured solid
	curves are the extracted cluster contours and black dashed curves are the
	reference boundaries. The contours are shown before alpha-shape
	polygonisation; the stair-stepped segments on the non-convex and
	\textit{density\_8} scenes illustrate the effect of uniform-grid
	discretisation.}
	\label{fig:S11_contours}
\end{figure}

"""

TAIL = "\n"


def main():
    text = open(SUP, encoding="utf-8").read()
    tables = open(TAB, encoding="utf-8").read()
    block = HEAD + tables + FIG + TAIL

    start = text.find(r"\section{Boundary-quality Evaluation on Controlled Geometries}")
    if start != -1:
        end = text.rfind(r"\end{document}")
        text = text[:start] + block + text[end:]
    else:
        end = text.rfind(r"\end{document}")
        text = text[:end] + block + "\n" + text[end:]
    open(SUP, "w", encoding="utf-8").write(text)
    print("S11 inserted; file now", len(text), "chars")


if __name__ == "__main__":
    main()
