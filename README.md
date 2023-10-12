# ABC Grounding

This folder contains the code of the LLM symbol grounding system for ABC, which is written in Python.

To determine the dummy constants and predicate resulted in the repaired theory, we use some Large Language Models (LLMs) to conduct symbol grounding on those dummy items. Given a repaired theory, the system find the axioms with dummy items, determine the possible grounding, and let users to choose which one(s) is/are suitable in the theory.

## How to run the code

### Pre-requisite

Before running the code, your folder hierarchy should be like this:

```bash
.
├── Folder "code" from ABC_Datalog
├── code_LLM (Grounding code from this repository)
├── (LLM folder)
├── (Unrepaired theories folder)
├── (Repaired theories folder)
├── requirements.txt
└── README.md
```

1. Code from ABC: Clone from [ABC_Datalog](https://github.com/Xuerli/ABC_Datalog). After that, make the following amendment in the "code" folder so that the program can return the number of repair plans:

- In main.pl:
  - Near line 30: Change from "abc:-" to "abc(**FaultNum**):-"
  - Near line 52: Append a new argument **FaultNum** at the end of the function "output", that is, change from "output(AllRepStates, ExecutionTime)" to "output(AllRepStates, ExecutionTime, **FaultNum**)"
- In fileOps.pl:
  - Near line 54 and 117: Append a new argument **FaultNum** at the end of the function "output".
  - Near line 107, 177, 200 and 313: Append a new argument **FaultNum** at the end of the function "write_Spec".
  - Near line 320: After "spec(faultsNum(InsuffNum, IncompNum)),", add "**FaultNum is InsuffNum + IncompNum,**".

2. Download the LLMs. We experimented with different LLMs without further fine-tuning, including GPT models from OpenAI [[1]](#1)[[2]](#2), T5 models by Google [[3]](#3), OpenLLaMA models from OpenLM Research [[4]](#4), and Dolly 2.0 models by Databricks [[5]](#5). Before use, manage the setting of grounding in main.py, and place your OpenAI API key in LLM.py.

3. Create a virtual environment and download all the required library with "requirements.txt". You can comment out some library based on your LLM choices.

4. Prepare the theory input file. It has to include a Datalog theory given by _axiom([...])_, and the preferred structure given by _trueSet([...])_ and _falseSet([...])_. Then one can put the items to protect from being changed in _protect([...])_, and heuristics to apply in _heuristics([...])._. Examples are in the [evaluation theories folder in ABC_Datalog](https://github.com/Xuerli/ABC_Datalog/tree/main/evaluation%20theories).

### Running ABC and Grounding

Suppose your folder hierarchy is like this:

```bash
.
├── Folder "code" from ABC_Datalog
|    └── ...
├── code_LLM (Grounding code from this repository)
|    └── ...
├── t5-xl-ssm-nq (An LLM folder)
|    └── ...
├── new_theories (Unrepaired theories folder)
|    └── theory1.pl
├── repaired_theories (Repaired theories folder)
├── requirements.txt
└── README.md
```

1. Activate the virtual environment. Go to the folder "code_LLM".

2. To run the ABC in Python:

   > python run_abc.py .. theory_dir export_folder_dir

   For instance,

   > python run_abc.py .. new_theories/theory1.pl repaired_theories

The output files include _abc_..._faultFree.txt_ which contains the repair solutions; _abc_..._record.txt_ which has the log information of ABC's procedure, and _abc_..._repNum.txt_ which is the pruned sub-optimal.

3. To run the LLM grounding:

   > python main.py file_dir model_name ans_num

   For instance,

   > python main.py repaired_theories/theory2.pl t5-xl-ssm-nq 1

   will use t5-xl-ssm-nq model to search for 1 grounding per each dummy item in file "repaired_theories/theory2.pl".

4. The grounding outputs are in "grounding_result" folder.

5. Repeat Step 3 and 4 if the grounded theory is faulty again.

## Evaluation Theories

We constructed theories automatically from two knowledge bases, enriched WebNLG dataset [[6]](#6) and excerpt of DART [[7]](#7), and replaced some items with dummy names. These theories serve as simulations of the generated repaired theories, with both assertions and rules.

In the selected corpora, each knowledge contains several propositions, with exactly 2 constants’ names and 1 predicate name. Thanks to the binary property, we construct a knowledge graph (KG) represents all knowledge, where the graph is directed and with parallel edges, vertices represent constant, and directed labelled edges represent the predicate and the subject-object relation. We construct a rule by leveraging the VF2 matching algorithm [[8]](#8) that solves subgraph isomorphism problem [[8]](#8)[[9]](#9).

During theory construction, we keep a subgraph for rule construction, and we try to extend the subgraph by adding an edge with two considerations - (1) after adding that edge, there is another distinct subgraph which is isomorphic to the resulting subgraph, and (2) we prioritize edges that make the new subgraph cyclic. The former consideration ensures the rule constructed is applicable on more knowledge, while the latter one avoids that the probability of getting constraint axioms is way greater than getting rules. We extend the subgraph a maximum of 3 times to avoid the rules becoming too lengthy.

After having a list of subgraph isomorphisms in KG, we select two to three for rule construction. We then rename the constant names in all subgraphs to variable names, unless these isomorphisms map to the same constant name. Next, we look for a set of knowledge that includes all propositions involved in one rule and include all propositions in that knowledge set in the theory, except for those that are deducible from rules.

Once we have the ground truth theories, we can mask constants or predicates randomly to have a test case theory. There are 4 Prolog files for each test case. Variation 1 replaces a constant name with a dummy name once only, while variation 2 replaces all the same constant names with that dummy name. Variations 3 and 4 act similarly except they replace a predicate name. We prioritize selecting items with two or more appearances in the theory so that there are differences between these variations.

## References

<a id="1">[1]</a>
T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, D. Amodei, Language Models are Few-Shot Learners, Advances in neural information processing systems 33 (2020) 1877–1901. URL: http://arxiv.org/abs/2005.14165.

<a id="2">[2]</a>
OpenAI, GPT-4 Technical Report, arXiv preprint arXiv:2303.08774 (2023). URL: http://arxiv.org/abs/2303.08774.

<a id="3">[3]</a>
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, P. J. Liu, Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, Journalof Machine Learning Research 21 (2020) 1–67. URL: http://jmlr.org/papers/v21/20-074.html.

<a id="4">[4]</a>
X. Geng, H. Liu, OpenLLaMA: An Open Reproduction of LLaMA, 2023. URL: https://github.com/openlm-research/open_llama.

<a id="5">[5]</a>
M. Conover, M. Hayes, A. Mathur, X. Meng, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, R. Xin, Free Dolly: Introducing the World’s First Truly Open Instruction-Tuned LLM, 2023. URL: https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm.

<a id="6">[6]</a>
T. C. Ferreira, D. Moussallem, S. Wubben, E. Krahmer, Enriching the WebNLG corpus, in: Proceedings of the 11th International Conference on Natural Language Generation, 2018, pp. 171–176. URL: http://data.statmt.org/wmt17_systems.

<a id="7">[7]</a>
L. Nan, D. Radev, R. Zhang, A. Rau, A. Sivaprasad, C. Hsieh, X. Tang, A. Vyas, N. Verma, P. Krishna, Y. Liu, N. Irwanto, J. Pan, F. Rahman, A. Zaidi, M. Mutuma, Y. Tarabar, A. Gupta, T. Yu, Y. C. Tan, X. V. Lin, C. Xiong, R. Socher, N. F. Rajani, DART: Open-Domain Structured Data Record to Text Generation, in: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, Online, 2021, pp. 432–447. URL: https://aclanthology.org/2021.naacl-main.37. doi:10.18653/v1/2021.naacl- main.37.

<a id="8">[8]</a>
L. P. Cordella, P. Foggia, C. Sansone, M. Vento, A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs, Pattern Analysis and Machine Intelligence, IEEE Transactions on 26 (2004) 1367–1372. URL: http://amalfi.dis.unina.it/graph/.

<a id="9">[9]</a>
S. A. Cook, The Complexity of Theorem-Proving Procedures, in: Proceedings of the Third Annual ACM Symposium on Theory of Computing, Association for Computing Machinery, New York, NY, USA, 1971, pp. 151–158.
