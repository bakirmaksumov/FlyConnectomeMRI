# Правки раздела Methods (B–F): проверенные по коду фрагменты LaTeX

*Собрано 2026-11-02. Каждый фрагмент сверен с кодом проекта, с замороженными файлами графов и с `docs/RESEARCH_PLAN.md`. Новых экспериментов не было, результаты не менялись.*

**Статус:** черновик для утверждения. В `docs/paper/latex/sections/03_methods.tex` эти правки **ещё не внесены**, ZIP для Overleaf не пересобран.

| Подраздел | Что заменяет в `03_methods.tex` |
|---|---|
| B. Image Preprocessing | старый пункт о предобработке и формулы нормализации |
| C. Patient Grouping | абзац «Conservative grouping of possibly related examinations» |
| D. Biological Graph Construction | «Source and extraction» и «Selection rule for strategy B». Обоснование выбора B, свойства графа и абзац про EB остаются отдельно |
| E. Matched Graph Controls | подраздел «Control Graphs» |
| F. Model Architecture | описание модели и формулы (3)–(6) |
| Limitations | новое предложение о seeds контролей |

---

## B. Image Preprocessing

### Что проверено в коде

| Шаг | Факт | Где в коде |
|---|---|---|
| Min-max | Каждый срез масштабируется отдельно по своим min и max. **Константы ε нет.** Если срез однородный, программа останавливается с ошибкой `DatasetError`. Все 2 134 среза прошли без этой ошибки | [figshare.py:74-80](../../src/flyconnectome_mri/datasets/figshare.py#L74-L80) |
| Resize | До 128×128, билинейная интерполяция со сглаживанием (`antialias=True`). Большинство срезов 512×512, 15 срезов исходно 256×256 | [transforms.py:52](../../src/flyconnectome_mri/datasets/transforms.py#L52) |
| Нормализация | Одно скалярное μ и одно σ по всем пикселям обучающих срезов **после resize**. σ — стандартное отклонение генеральной совокупности. В Phase 10 пересчитываются для каждого фолда по очищенной обучающей части (фолд 1: μ = 0.143685, σ = 0.16231) | [transforms.py:63-81](../../src/flyconnectome_mri/datasets/transforms.py#L63-L81) |
| Аугментация | Нет | `RESEARCH_PLAN.md` |

### LaTeX

```latex
\subsection{Image Preprocessing}
Each MRI slice $x$ was independently min--max scaled using its own
intensity range:
\begin{equation}
\tilde{x} = \frac{x - \min(x)}{\max(x) - \min(x)} .
\end{equation}
Slices with constant intensity ($\max(x)=\min(x)$) would be rejected
by the pipeline with an error rather than rescaled; no such slice
occurred in the dataset. The scaled image was then resized to
$128 \times 128$ pixels using bilinear interpolation with
antialiasing:
\begin{equation}
z = \operatorname{Resize}_{128 \times 128}(\tilde{x}) .
\end{equation}
Finally, the resized images were standardized:
\begin{equation}
\hat{x} = \frac{z - \mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}} ,
\end{equation}
where $\mu_{\mathrm{train}}$ and $\sigma_{\mathrm{train}}$ are a single
scalar mean and (population) standard deviation computed over all
pixels of the resized training images only. In cross-validation they
were recomputed for each outer fold from that fold's training set.
The same values were applied unchanged to the validation and test
images. No data augmentation was used.
```

---

## C. Patient Grouping

### Что проверено

- **Подсчёт групп:** 171 ID; 42 связанных ID объединены в 18 групп. 171 − 42 + 18 = **147 групп**. Все 42 связанных ID относятся к глиоме.
- **Правило:** `^(?P<base>MR\d{6})[A-Z]?$`.
- **Агрегация по пациенту:** вероятности срезов усредняются по **исходному** `patient_id`, а не по ключу группы ([metrics.py:97-101](../../src/flyconnectome_mri/training/metrics.py#L97-L101)).
- **Как группы используются при разделении:**
  - В fixed split группы целиком распределяются по train, val и test.
  - В CV официальные фолды **не перестраивались**. Родственники тестового фолда **исключались** из train и validation. Поэтому последнее предложение абзаца сформулировано так, чтобы подходить к обоим случаям.

### LaTeX

```latex
\subsection{Patient Grouping}
Some patient identifiers shared the same MR-prefixed base identifier
but differed by a trailing letter (e.g., MR040240 and MR040240B).
Whether these identifiers represented repeated examinations of the
same individual was unknown. As a conservative precaution against
potential patient overlap, identifiers matching
\texttt{MR} followed by six digits and an optional single trailing
letter were assigned a common grouping key defined by the
\texttt{MR} base number. This procedure grouped 42 identifiers into
18 groups, yielding 147 groups across the 171 patient identifiers in
the binary dataset; all 42 grouped identifiers belonged to the glioma
class. The 147 groups should not be interpreted as 147 confirmed
independent individuals. The original patient identifiers were
retained for patient-level prediction and evaluation. The grouping
keys were used only to prevent potentially related examinations from
appearing in both the training/validation data and the test data of
the same evaluation; how this was enforced in the fixed split and in
cross-validation is described below.
```

---

## D. Biological Graph Construction

### Что проверено

Источники: [drosophila.py:144-233](../../src/flyconnectome_mri/connectome/drosophila.py#L144-L233), [loader.py](../../src/flyconnectome_mri/connectome/loader.py), `data/connectome/drosophila_visual_128*_metadata.json`, [configs/malecns_visual.yaml](../../configs/malecns_visual.yaml).

| Шаг | Факт |
|---|---|
| Источник | MaleCNS v1.0 через neuPrint (`male-cns:v1.0`), база данных uuid `4b2087c0fbe046bfaf0d60bc970e3e5d`, neuprint-python 0.6.3 |
| Кандидаты | 11 типов клеток; есть синапсы в ME(R); `assignedOlHex1/2` не null. Всего **9 727** кандидатов |
| Центр | Медиана (hex1, hex2) по каждой координате отдельно среди **817** полных колонок, с отбрасыванием дробной части. Получается (18, 20). Код проверяет, что центр сам является полной колонкой |
| Расстояние | max(\|Δ1\|, \|Δ2\|), если Δ1·Δ2 ≥ 0; иначе \|Δ1\| + \|Δ2\| |
| Отбор | Сортировка: расстояние → hex1 → hex2 → порядок типов → bodyId. Берутся первые 128; максимальное расстояние 2 |
| Рёбра | `fetch_adjacencies(..., omit_rois=True)`: синапсы считаются **по всей нервной системе**, а не только внутри ME |
| Направление | Источник — пресинаптический нейрон, цель — постсинаптический |
| Порог | ≥ 1 синапса; получается 989 рёбер |
| Петли | Политика «drop»; найдено **0** |
| Изолированные узлы | 0 |
| Порядок узлов | По возрастанию bodyId; так же в контрольных графах |
| Не используется | Число синапсов сверх порога, медиатор и знак связи (ACh 80, Glu 24, GABA 24), типы клеток |

### LaTeX

```latex
\subsection{Biological Graph Construction}
We extracted a directed subgraph of the \textit{Drosophila} visual
system from the MaleCNS v1.0 connectome~\cite{malecns}, accessed
through neuPrint~\cite{neuprint} (dataset \texttt{male-cns:v1.0}).
The subgraph is a local retinotopic patch of the input stage of the
ON/OFF motion pathways in the right medulla. It contains 128 nodes and
989 directed edges. Each node corresponds to one biological neuron,
and the graph connectivity is represented by a binary adjacency
matrix. This matrix defines which connections are permitted in the
graph-constrained layers; the strengths of these connections are
learned during MRI classification training.

\textbf{Neuron selection.} Candidate neurons were those of the
11 columnar cell types
$\mathcal{T}=\{$L1, L2, L3, Mi1, Mi4, Mi9, C3, Tm1, Tm2, Tm4, Tm9$\}$
that have synapses in ME(R) and an assigned retinotopic column
$(h_1,h_2)$ (\texttt{assignedOlHex1/2}); this gave 9,727 candidates.
Let $\mathcal{K}$ be the set of columns containing all 11 types
(817 columns). The central column was
\begin{equation}
c = \bigl(\lfloor \operatorname{median}_{\mathcal{K}} h_1 \rfloor,\;
          \lfloor \operatorname{median}_{\mathcal{K}} h_2 \rfloor\bigr)
  = (18, 20),
\end{equation}
and the hexagonal distance of a neuron in column $(h_1,h_2)$ from it,
with $\Delta_k = h_k - c_k$, was
\begin{equation}
d(\Delta_1,\Delta_2) =
\begin{cases}
\max(|\Delta_1|,|\Delta_2|), & \Delta_1\Delta_2 \ge 0,\\
|\Delta_1|+|\Delta_2|, & \text{otherwise}.
\end{cases}
\end{equation}
Candidates were sorted by $d$, then by $h_1$, $h_2$, the type order
of $\mathcal{T}$ and the neuron identifier (bodyId), and the first
128 were kept (maximum distance 2). T4/T5 and Tm3 have no column
assignment in MaleCNS and were therefore not included; no columns
were inferred.

\textbf{Edges.} For the selected neurons we retrieved all directed
connections among them. Let $s_{ij}$ be the total number of synapses
from presynaptic neuron $i$ to postsynaptic neuron $j$, summed over
the whole dataset (not only within ME). The binary adjacency matrix is
\begin{equation}
A_{ij} =
\begin{cases}
1, & s_{ij} \ge 1 \text{ and } i \ne j,\\
0, & \text{otherwise}.
\end{cases}
\end{equation}
Self-loops were to be removed; none were present. No neuron was
isolated. Nodes were indexed in ascending order of bodyId, and the
same indexing was used for all control graphs.

\textbf{Information not used.} The model uses only the presence and
direction of connections. Synapse counts (beyond the inclusion
threshold), neurotransmitter identity and therefore the sign of a
connection, and cell-type labels are not used. The assignment of
encoder feature $i$ to node $i$ follows the bodyId order and carries
no functional correspondence between image features and particular
neurons; the experiment therefore tests the connection topology,
not the function of individual cell types.
```

---

## E. Matched Graph Controls

### Что проверено

Источники: [random_graphs.py](../../src/flyconnectome_mri/connectome/random_graphs.py), [controls.py](../../src/flyconnectome_mri/connectome/controls.py), `data/connectome/*_visual128_raw_seed*.csv` и `*_metadata.json`, NetworkX 3.7 из `.venv`.

| Контроль | Факты |
|---|---|
| **Rewired** | Направленная двойная перестановка рёбер (Maslov–Sneppen). 10m = **9 890** попыток; пары рёбер выбираются равномерно с возвращением. Отклоняется: одно и то же ребро дважды, петля (a = d или c = b), уже существующее ребро a→d или c→b. Принято по seeds 0–4: **7 570, 7 536, 7 620, 7 544, 7 608** (76–77 %). Совпадение с биологическими рёбрами: 12.3 / 14.7 / 12.9 / 13.9 / 11.3 % (в среднем 13.0 %); приближённое ожидание первого порядка ≈ 13.3 %. Равномерность выборки и сходимость **не проверялись** |
| **WS-type** | Кольцо в порядке индексов узлов (по bodyId). Цели по порядку i+1, i−1, i+2, i−2, … по модулю 128. Выходная степень 7; первые 93 узла получают 8 (989 = 7·128 + 93). С вероятностью p = 0.1 ребро сохраняет источник и получает новую цель, равномерно среди узлов, которые не совпадают с источником и не являются его целями. Выходные степени 7–8 сохраняются; входные получаются 4–12. Совпадение с биологическими рёбрами — 0 |
| **ER** | m различных упорядоченных пар, выбранных равномерно из n(n−1) |
| **BA-type** | k = 9: наименьшее k, при котором k(n−k) ≥ m. NetworkX `barabasi_albert_graph` с начальным графом-звездой на k+1 узлах даёт 1 071 неориентированное ребро. Направление каждого ребра случайно (вероятность 0.5); равномерно без возвращения оставляются 989 (82 удаляются). Seed NetworkX берётся из потока BA-контроля |
| **Seeds** | `default_rng([seed, stream])`; номер потока: ER 1, WS 2, BA 3, rewired 4. Обучающий seed s ↔ граф seed s. В CV граф seed s используется во всех пяти фолдах |
| **Степени биологического графа** | Выходные 0–20, входные 0–38. 7 узлов без входящих рёбер, 3 без исходящих |
| **Параметры** | 44 411 во всех графовых условиях |

### LaTeX

```latex
\subsection{Matched Graph Controls}
For the 128-node visual graph, the primary topology comparison used
the biological graph and degree-preserving rewired controls.
A Watts--Strogatz-type control provided an additional comparison in
the five-fold evaluation. Erd\H{o}s--R\'enyi (ER) and
Barab\'asi--Albert-type (BA) controls were also evaluated in the
preceding single-split visual-graph experiment.

All visual-graph controls contained 128 nodes and 989 directed edges,
without self-loops or duplicate edges. They were evaluated using the
same classifier architecture, with 44,411 trainable parameters in each
graph condition. All controls matched the biological graph in node and
edge counts; only the degree-preserving controls additionally matched
every node's in-degree and out-degree.

\textbf{Degree-preserving rewiring.} Starting from the biological graph,
we applied directed double-edge swaps:
\begin{equation}
(a\rightarrow b,\;c\rightarrow d)
\longmapsto
(a\rightarrow d,\;c\rightarrow b).
\end{equation}
Each accepted swap preserved the in-degree and out-degree of every
node. Pairs of edges were sampled uniformly with replacement. An
attempt was rejected if the same edge was selected twice, if either
proposed edge was a self-loop, or if either proposed edge already
existed.

For a graph with $m=989$ edges, the procedure performed
$10m=9{,}890$ attempts, including rejected attempts. Across the five
graph seeds, 7,536--7,620 swaps were accepted. The rewired graphs
shared 11.3--14.7\% (mean 13.0\%) of their edges with the biological
graph, close to the first-order expectation
$\sum_{(i,j)\in\mathcal{E}} k_i^{\mathrm{out}} k_j^{\mathrm{in}} / m^2
\approx 13.3\%$ for this degree sequence. Uniform sampling over all
graphs with the specified degree sequence and convergence of the
rewiring procedure were not established. Rewiring reduced reciprocity
from 0.51 to 0.08 and clustering from 0.38 to 0.10 (means over seeds).
This control compared the biological connectivity with alternative
configurations preserving the directed degree sequence; it did not
isolate the effects of individual higher-order graph properties.

\textbf{Watts--Strogatz-type control.} We used a custom directed
construction inspired by the Watts--Strogatz model. Nodes were
arranged on a ring in ascending \texttt{bodyId} order; this ring
ordering did not represent biological spatial coordinates. For each
node $i$, initial targets were selected in the order
$i+1,i-1,i+2,i-2,\ldots$, modulo $n=128$.

Each node was assigned $\lfloor m/n\rfloor=7$ outgoing edges, with one
additional outgoing edge assigned to each of the first $m\bmod n=93$
nodes. This produced exactly 989 directed edges. Each edge was
selected for target rewiring with probability $p=0.1$, retaining its
source. The replacement target was sampled uniformly from nodes other
than the source and its existing targets.

The procedure preserved the number of edges and each node's assigned
out-degree of 7 or 8, while allowing in-degrees to change (4--12).
It did not preserve the biological graph's in-degree or out-degree
sequence (biological out-degrees 0--20, in-degrees 0--38). No WS edge
coincided with a biological edge.

\textbf{ER and BA-type controls.} The directed ER control was generated
by sampling $m$ distinct ordered node pairs uniformly from the
$n(n-1)$ possible non-self connections. The BA-type control was built
from an undirected Barab\'asi--Albert graph generated with
NetworkX~3.7 (\texttt{barabasi\_albert\_graph}), using the default
initial star graph on $k+1$ nodes and $k$ edges per added node. We
used the smallest $k$ with $k(n-k)\ge m$, i.e.\ $k=9$, giving
$9\cdot 119=1{,}071$ undirected edges. Each edge was assigned a random
direction with probability $0.5$, and $m=989$ of the directed edges
were retained uniformly at random without replacement (82 removed).

\textbf{Graph instances and training seeds.} Five instances of each
control type were generated using graph seeds 0--4 and separate random
streams for each control type. Training seed $s$ was paired with graph
seed $s$. In the five-fold evaluation, the same graph instance for
seed $s$ was reused across all five outer folds. The biological graph
remained fixed across all training seeds and folds.
```

**Ссылка:** записи для NetworkX в `references.bib` нет. Если на неё ссылаться, нужна новая запись с пометкой [CITATION NEEDED]; выходные данные не придумывать.

---

## F. Model Architecture

### Что проверено

Источники: [encoder.py](../../src/flyconnectome_mri/models/encoder.py), [cnn_baseline.py](../../src/flyconnectome_mri/models/cnn_baseline.py), [connectome_block.py](../../src/flyconnectome_mri/models/connectome_block.py), [graph_classifier.py](../../src/flyconnectome_mri/models/graph_classifier.py). Число параметров закреплено тестом `tests/test_frozen_decisions.py`.

**Цепочка размеров:** (B,1,128,128) → Conv 3×3 1→16 + ReLU + MaxPool → (B,16,64,64) → Conv 16→32 + ReLU + MaxPool → (B,32,32,32) → Conv 32→128 + ReLU → (B,128,32,32) → глобальный средний пуллинг → (B,128).

| Часть | Формула | Параметров |
|---|---|---|
| Conv1 | 1·16·9 + 16 | 160 |
| Conv2 | 16·32·9 + 32 | 4 640 |
| Conv3 | 32·128·9 + 128 | 36 992 |
| **Энкодер** | | **41 792** |
| BatchNorm1d | γ, β: 2·128 | 256 |
| Графовый блок | 2·(989 + 128) | 2 234 |
| Linear | 128 + 1 | 129 |
| **Графовая модель (visual)** | | **44 411** |
| **CNN baseline** | 41 792 + 129 | **41 921** |

**Ключевые факты:**
- Энкодер у обеих моделей одинаковый. В нём нет BatchNorm, dropout и предобучения.
- Признак i становится начальным состоянием узла i. Сопоставление определяется порядком bodyId и функционального смысла не несёт.
- BatchNorm1d при обучении использует статистику батча, при оценке — накопленную (running) статистику.
- Параметры есть только у существующих рёбер. Это эквивалентно (W⊙A)ᵀh. Два слоя используют одни и те же рёбра, но у каждого свои веса.
- Инициализация весов: U(−1/√k_j, 1/√k_j), где k_j — входящая степень узла-цели. Смещения равны нулю. Числа синапсов не используются.
- Обходных путей нет: классификатор видит только 128 состояний после второго слоя.
- **Узлы без входящих рёбер** (7 в биологическом и rewired-графах, 0 в WS) после первого слоя равны ReLU(b_j) и от изображения не зависят. Свой исходный признак они передают дальше по исходящим рёбрам.

### LaTeX

```latex
\subsection{Model Architecture}
Both model types share the same convolutional encoder $\phi$:
three $3\times3$ convolutions (1$\rightarrow$16, 16$\rightarrow$32,
32$\rightarrow$128 channels, padding 1), each followed by ReLU, with
$2\times2$ max pooling after the first two, and global average pooling,
giving $f=\phi(x)\in\mathbb{R}^{128}$. The encoder has no batch
normalization, dropout or pretraining.

\textbf{CNN baseline.} The logit is $z = v^{\top} f + c$.

\textbf{Graph-constrained model.} The features are batch-normalized and
feature $i$ initializes the scalar state of node $i$:
\begin{equation}
h^{(0)} = \operatorname{BN}(f) \in \mathbb{R}^{128}.
\end{equation}
Two propagation layers $\ell = 1, 2$ then update the node states along
the directed edge set $\mathcal{E}$:
\begin{equation}
h^{(\ell)}_j = \operatorname{ReLU}\Bigl(
\sum_{i:\,(i\rightarrow j)\in\mathcal{E}} w^{(\ell)}_{ij}\, h^{(\ell-1)}_i
+ b^{(\ell)}_j \Bigr),
\end{equation}
equivalently $h^{(\ell)} = \operatorname{ReLU}\bigl((W^{(\ell)}\odot A)^{\top}
h^{(\ell-1)} + b^{(\ell)}\bigr)$ with the binary adjacency matrix $A$.
Only existing edges have parameters; the two layers share $\mathcal{E}$
but have separate weights. Weights were initialized from
$\mathcal{U}(-1/\sqrt{k_j},\,1/\sqrt{k_j})$, where $k_j$ is the
in-degree of the target node, and biases to zero; synapse counts were
not used. The logit is computed from all 128 final node states,
\begin{equation}
z = v^{\top} h^{(2)} + c, \qquad p = \sigma(z),
\end{equation}
with no pooling and no skip connection, so all information reaching the
classifier passes through the graph layers. Nodes without incoming
edges (7 in the biological and rewired graphs) receive only their bias
after the first layer and therefore carry no image-dependent signal in
$h^{(1)}$ and $h^{(2)}$, although they pass their initial state on along their outgoing edges.

The graph models have 44,411 trainable parameters (encoder 41,792,
batch normalization 256, graph layers $2\times(989+128)$, classifier
129); the CNN baseline has 41,921. The CNN--graph comparison therefore
differs in both architecture and parameter count, whereas all graph
conditions have identical parameter counts.
```

**Открытый вопрос:** куда поместить абзац «Development note» о первой версии модели с mean pooling, которая схлопнулась, — оставить здесь или перенести в Discussion.

---

## Добавление в Limitations

**Почему:** у контрольных графов seed s меняет и граф, и случайность обучения. У биологического графа меняется только случайность обучения. Поэтому разброс между seeds у этих условий измеряет разное. Это касается и наблюдения Phase 9 о меньшем разбросе у Drosophila.

```latex
For control graphs, the seed determined both the graph instance and
the training randomness, so seed-to-seed variation of the controls
combines graph variation and optimisation variation, whereas for the
fixed biological graph it reflects optimisation variation only; the
design does not separate these two sources.
```

---

## Новые числа, которых не было в прежнем черновике

Все получены из замороженных файлов, без новых запусков:
- Число кандидатов (9 727) и полных колонок (817) — из метаданных графа.
- Принятые перестановки (7 536–7 620) — из метаданных rewired-графов.
- Ожидаемое совпадение рёбер ≈ 13.3 % — посчитано по `drosophila_visual_128_raw.csv`.
- Степени WS-графов (выходные 7–8, входные 4–12) и их совпадение с биологическими рёбрами (0) — посчитаны по `ws_visual128_raw_seed*.csv`.
- Степени биологического графа (выходные 0–20, входные 0–38; 7 узлов без входящих рёбер, 3 без исходящих) — посчитаны по CSV.
- k = 9 и 1 071 ребро для BA — из функции `_ba_attachment_count` и NetworkX 3.7.

## Что дальше

1. Утвердить фрагменты B–F.
2. Внести их в `03_methods.tex`, а предложение из Limitations — в `07_limitations.tex`.
3. Решить вопрос о ссылке на NetworkX.
4. Пересобрать `FlyConnectomeMRI_Overleaf.zip`.
