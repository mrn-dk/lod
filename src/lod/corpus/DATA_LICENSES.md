# Training-data licences

The corpus the Lod models are trained on is built from third-party sources, each under its own terms, plus data our own generators write. This file lists every source of the release corpus: its licence as the source states it, the source-table rows it feeds, and how many questions it contributes to training and to the held-out dev/test splits.

The model weights and this code are released under Apache-2.0. Some training sources are licensed for non-commercial or research use only, or carry share-alike or unclear terms (flagged below). Whether such terms extend to weights trained on the data is unsettled; if you plan commercial use of the models, review the flagged sources first.

## Summary

- **Training questions:** 537,899, of which 256,830 come from Lod's own generators (no third-party data).
- **non-commercial:** 8,396 training questions from 12 sources.
- **research use:** 7,618 training questions from 2 sources.
- **share-alike:** 35,993 training questions from 19 sources.
- **no-derivatives:** 91 training questions from 1 source.
- **copyleft:** 92 training questions from 1 source.
- **unclear:** 14,226 training questions from 26 sources.
- Sources with no training questions (*eval only*) are used only in the held-out dev and test splits.
- Hub-sweep sources (row 1) record no licence of their own; the licence shown is the one on the dataset's Hub card, looked up for this file.

## Flagged sources in training

| source | licence | flags | rows | train q |
|---|---|---|---|---|
| sick | CC-BY-NC-SA-3.0 | non-commercial, share-alike | 50 | 4,000 |
| coastalcph/lex_glue | CC-BY-NC-SA-4.0 (LexGLUE ECtHR, Chalkidis et al. 2019/2022; HUDOC case facts) -- non-commercial project, attribution owed | non-commercial, share-alike | 51 | 1,178 |
| docs.manifold.markets/api | Manifold Markets API data; non-commercial use only for AI training (Manifold terms) -- not for commercial model training | non-commercial | 28 | 789 |
| Larxel/PHAENIX-1 | Hub card: cc-by-nc-4.0 | non-commercial | 1 | 644 |
| Tobi-Bueck/customer-support-tickets | Hub card: cc-by-nc-4.0 | non-commercial | 1 | 460 |
| NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026 | Hub card: cc-by-nc-sa-4.0 | non-commercial, share-alike | 1 | 406 |
| scbirlab/thomas-2018-spark-wt | Hub card: cc-by-nc-4.0 | non-commercial | 1 | 368 |
| nhop/OpenReview | Hub card: cc-by-nc-4.0 | non-commercial | 1 | 184 |
| maastrichtlawtech/bsard | Hub card: cc-by-nc-sa-4.0 | non-commercial, share-alike | 1 | 92 |
| J0nasW/science-datalake | Hub card: cc0-1.0,cc-by-4.0,cc-by-sa-4.0,cc-by-nc-sa-4.0,cc-by-nc-4.0 | non-commercial, share-alike | 1 | 92 |
| PKU-Alignment/BeaverTails | Hub card: cc-by-nc-4.0 | non-commercial | 1 | 92 |
| sled-umich/SDN | Hub card: cc-by-nc-nd-4.0 | no-derivatives, non-commercial | 1 | 91 |
| cardiffnlp/tweet_eval | undefined per the TweetEval card (research release; Twitter ToS applies) | research use, unclear | 44 | 5,401 |
| github.com/logpai/loghub | Loghub research-use (research/academic only; attribution to github.com/logpai/loghub + ISSRE'23 citation required; not OSI) | research use | 14 | 2,217 |
| youngfish42/PaperVault | Hub card: gpl-3.0 | copyleft | 1 | 92 |
| cardiffnlp/tweet_sentiment_multilingual | CC-BY-3.0 (SemEval tweets) + Twitter ToS | unclear | 46 | 2,499 |
| nlp.stanford.edu/sentiment | no licence stated (Stanford Sentiment Treebank v1.0, nlp.stanford.edu/sentiment) | unclear | 44 | 2,000 |
| tensorfeed/ai-ecosystem-daily | Hub card: other | unclear | 1 | 920 |
| AlgorithmicResearchGroup/s2orc-cs-enriched | Hub card: none | unclear | 1 | 552 |
| einrafh/hnm-fashion-recommendations-data | Hub card: none | unclear | 1 | 368 |
| toxigen/toxigen-data | Hub card: none | unclear | 1 | 276 |
| saattrupdan/womens-clothing-ecommerce-reviews | Hub card: none | unclear | 1 | 276 |
| ai4privacy/open-pii-masking-500k-ai4privacy | Hub card: other | unclear | 1 | 184 |
| data-is-better-together/fineweb-c | Hub card: none | unclear | 1 | 184 |
| almanach/Biomed-Enriched | Hub card: none | unclear | 1 | 184 |
| data-is-better-together/10k_prompts_ranked | Hub card: other | unclear | 1 | 183 |
| M-A-D/Mixed-Arabic-Datasets-Repo | Hub card: none | unclear | 1 | 92 |
| allenai/art | Hub card: unknown | unclear | 1 | 92 |
| kareenamehta/ccnews | Hub card: none | unclear | 1 | 92 |
| stanford-oval/ccnews | Hub card: none | unclear | 1 | 92 |
| SetFit/bbc-news | Hub card: none | unclear | 1 | 92 |
| ai4bharat/indic_glue | Hub card: other | unclear | 1 | 92 |
| ai4privacy/pii-masking-400k | Hub card: other | unclear | 1 | 92 |
| bastao/VeraCruz_PT-BR | Hub card: none | unclear | 1 | 92 |
| cw1521/ember2018-malware | Hub card: none | unclear | 1 | 92 |
| hezarai/sentiment-dksf | Hub card: none | unclear | 1 | 92 |
| medalpaca/medical_meadow_health_advice | Hub card: none | unclear | 1 | 92 |
| papluca/language-identification | Hub card: none | unclear | 1 | 92 |
| tdavidson/hate_speech_offensive | Hub card: unknown | unclear | 1 | 92 |
| claritystorm/cfpb-consumer-complaints | Hub card: other | unclear | 1 | 3 |
| tals/vitaminc | CC BY-SA 3.0 (VitaminC; Wikipedia-derived) — share-alike | share-alike | 34 | 10,000 |
| Davlan/sib200 | CC-BY-SA-4.0 | share-alike | 46 | 6,000 |
| dl.fbaipublicfiles.com/mtop/mtop.zip | CC-BY-SA-4.0 (MTOP release, LICENSE.txt) | share-alike | 45 | 3,999 |
| copenlu/fever_gold_evidence | CC BY-SA 3.0 (FEVER claims; Wikipedia evidence) — share-alike | share-alike | 34 | 2,000 |
| stanfordnlp/snli | CC-BY-SA-4.0 | share-alike | 24 | 2,000 |
| DeepPavlov/hwu64 | CC-BY-SA-3.0 (Liu et al. 2019, NLU-Evaluation-Data) | share-alike | 50 | 1,999 |
| fancyzhx/dbpedia_14 | CC-BY-SA-3.0 (Wikipedia / DBpedia abstracts) | share-alike | 45 | 1,999 |
| strickvl/isafpressreleases | Hub card: cc-by-sa-4.0 | share-alike | 1 | 846 |
| arxiv.org/help/oa | CC-BY-4.0 / CC-BY-SA-4.0 / CC0-1.0 (per document, in meta) | share-alike | 7 | 542 |
| adugeen/personal-facts-msc | Hub card: cc-by-sa-4.0 | share-alike | 1 | 460 |
| derek-thomas/ScienceQA | Hub card: cc-by-sa-4.0 | share-alike | 1 | 184 |
| AI4Math/IneqMath | Hub card: cc-by-sa-4.0 | share-alike | 1 | 92 |
| google/boolq | Hub card: cc-by-sa-3.0 | share-alike | 1 | 92 |
| joey234/nan-nli | Hub card: cc-by-sa-4.0 | share-alike | 1 | 12 |

## Generated data

Rows 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 51 are written by the generators in `src/lod/corpus/services/sources/synth/` and contain no third-party data (256,830 training questions). generated by src/lod/corpus/services/sources/synth/toolgate.py; CLI half seeded by command and flag names from the local man pages (no man-page text); HTTP half seeded by method and path templates from Apache-2.0 / MIT OpenAPI specs attributed in src/lod/corpus/services/sources/synth/toolgate_http.json

## All sources

| source | licence | rows | train q | dev/test q |
|---|---|---|---|---|
| [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache-2.0 | 20 | 32,817 | 7,139 |
| [gharchive.org](https://www.gharchive.org/) | CC-BY-4.0 (GH Archive aggregation of public GitHub events) | 5 | 19,550 | 7,162 |
| [tasksource/bigbench](https://huggingface.co/datasets/tasksource/bigbench) | Apache-2.0 | 2 | 17,980 | 5,613 |
| [Muennighoff/natural-instructions](https://huggingface.co/datasets/Muennighoff/natural-instructions) | Apache-2.0 | 3 | 14,992 | 2,339 |
| [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | CC-BY-4.0 | 21 | 13,880 | 3,470 |
| [nvd.nist.gov/developers/vulnerabilities](https://nvd.nist.gov/developers/vulnerabilities) | public domain (NIST NVD) | 9 | 11,242 | 4,828 |
| [tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc) | CC BY-SA 3.0 (VitaminC; Wikipedia-derived) — share-alike | 34 | 10,000 | 0 |
| [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) | CC0-1.0 | 13 | 8,003 | 3,780 |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | CC-BY-4.0 | 16, 46 | 7,995 | 1,600 |
| [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC-BY-SA-4.0 | 46 | 6,000 | 987 |
| [hendrycks/ethics](https://huggingface.co/datasets/hendrycks/ethics) | MIT | 50 | 6,000 | 0 |
| [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | undefined per the TweetEval card (research release; Twitter ToS applies) | 44 | 5,401 | 3,204 |
| [registry.opendata.aws/noaa-gefs-reforecast](https://registry.opendata.aws/noaa-gefs-reforecast/) | public domain (NOAA GEFS v12 reforecast) | 27 | 5,328 | 7,992 |
| [philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters) | public domain (Federal Reserve Bank of Philadelphia) | 26 | 4,000 | 5,323 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: HaHackathon, SemEval-2021 Task 7 (Meaney et al. 2021) | 50 | 4,000 | 0 |
| [sick](https://huggingface.co/datasets/sick) | CC-BY-NC-SA-3.0 | 50 | 4,000 | 0 |
| [dl.fbaipublicfiles.com/mtop/mtop.zip](https://dl.fbaipublicfiles.com/mtop/mtop.zip) | CC-BY-SA-4.0 (MTOP release, LICENSE.txt) | 45 | 3,999 | 0 |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0-1.0 | 19 | 3,999 | 0 |
| [mteb/amazon_massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario) | CC-BY-4.0 | 46 | 3,750 | 1,000 |
| [github.com/nyu-mll/quality](https://github.com/nyu-mll/quality) | CC-BY-4.0 (QuALITY, Pang et al. 2022); article texts: Project Gutenberg licence (US public domain), OANC licence (Slate), CC-BY-4.0 (misc) | 51 | 3,549 | 1,042 |
| [masakhane/afrisenti](https://huggingface.co/datasets/masakhane/afrisenti) | CC-BY-4.0 | 46 | 3,000 | 500 |
| [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) | CC-BY-3.0 (SemEval tweets) + Twitter ToS | 46 | 2,499 | 0 |
| [github.com/logpai/loghub](https://github.com/logpai/loghub) | Loghub research-use (research/academic only; attribution to github.com/logpai/loghub + ISSRE'23 citation required; not OSI) | 14 | 2,217 | 443 |
| [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) | CC-BY-4.0 | 17 | 2,000 | 425 |
| [BEE-spoke-data/consumer-finance-complaints](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints) | public domain (US Government, CFPB Consumer Complaint Database; Hub mirror CC0-1.0) | 45 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: SARC (Khodak et al. 2018) | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: HYPO-L (Zhang & Wan 2022) | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: TalkDown (Wang & Potts 2019) | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Stanford Politeness Corpus, CC-BY-4.0 (Danescu-Niculescu-Mizil et al. 2013) | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Contextual Abuse Dataset (Vidgen et al. 2021) | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Diplomacy deception, It Takes Two to Lie (Peskov et al. 2020) | 50 | 2,000 | 0 |
| [alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | CC-BY-4.0 | 50 | 2,000 | 0 |
| [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | Apache-2.0 | 50 | 2,000 | 0 |
| [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa) | CC-BY-4.0 | 50 | 2,000 | 0 |
| [allenai/swag](https://huggingface.co/datasets/allenai/swag) | MIT | 50 | 2,000 | 0 |
| [benayas/snips](https://huggingface.co/datasets/benayas/snips) | CC-BY-4.0 | 16 | 2,000 | 0 |
| [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | CC-BY-3.0 (SemEval-2017 Task 4, per the TweetEval card) | 44 | 2,000 | 0 |
| [ccdv/arxiv-classification](https://huggingface.co/datasets/ccdv/arxiv-classification) | CC0-1.0 | 6 | 2,000 | 0 |
| [copenlu/fever_gold_evidence](https://huggingface.co/datasets/copenlu/fever_gold_evidence) | CC BY-SA 3.0 (FEVER claims; Wikipedia evidence) — share-alike | 34 | 2,000 | 0 |
| [demelin/moral_stories](https://huggingface.co/datasets/demelin/moral_stories) | MIT | 50 | 2,000 | 0 |
| [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | Apache-2.0 (per the fancyzhx/amazon_polarity card) | 44 | 2,000 | 0 |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | 50 | 2,000 | 0 |
| [ml4pubmed/pubmed-classification-20k](https://huggingface.co/datasets/ml4pubmed/pubmed-classification-20k) | public domain (NLM) | 7 | 2,000 | 0 |
| [nlp.stanford.edu/sentiment](https://nlp.stanford.edu/sentiment/) | no licence stated (Stanford Sentiment Treebank v1.0, nlp.stanford.edu/sentiment) | 44 | 2,000 | 0 |
| [pfb30/multi_woz_v22](https://huggingface.co/datasets/pfb30/multi_woz_v22) | Apache-2.0 | 50 | 2,000 | 0 |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | CC-BY-SA-4.0 | 24 | 2,000 | 0 |
| [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | AFL-3.0 | 50 | 2,000 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Question intimacy (Pei & Jurgens 2020) | 50 | 1,999 | 0 |
| [DeepPavlov/hwu64](https://huggingface.co/datasets/DeepPavlov/hwu64) | CC-BY-SA-3.0 (Liu et al. 2019, NLU-Evaluation-Data) | 50 | 1,999 | 0 |
| [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | CC-BY-SA-3.0 (Wikipedia / DBpedia abstracts) | 45 | 1,999 | 0 |
| [HelloWorld2307/dynasent](https://huggingface.co/datasets/HelloWorld2307/dynasent) | Apache-2.0 | 22 | 1,998 | 0 |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Empathic reactions to news (Buechel et al. 2018) | 50 | 1,673 | 0 |
| [atticusprojectai.org/cuad](https://www.atticusprojectai.org/cuad) | CC-BY-4.0 (CUAD v1, The Atticus Project; contracts from SEC EDGAR filings) | 51 | 1,586 | 334 |
| [Raywithyou/TabFact](https://huggingface.co/datasets/Raywithyou/TabFact) | CC-BY-4.0 (TabFact) | 31 | 1,318 | 1,112 |
| [coastalcph/lex_glue](https://huggingface.co/datasets/coastalcph/lex_glue) | CC-BY-NC-SA-4.0 (LexGLUE ECtHR, Chalkidis et al. 2019/2022; HUDOC case facts) -- non-commercial project, attribution owed | 51 | 1,178 | 160 |
| [ibm-research/claim_stance](https://huggingface.co/datasets/ibm-research/claim_stance) | CC-BY-3.0 | 44 | 1,038 | 0 |
| [tensorfeed/ai-ecosystem-daily](https://huggingface.co/datasets/tensorfeed/ai-ecosystem-daily) | Hub card: other | 1 | 920 | 0 |
| [mjbommar/opengloss-v1.3-definitions](https://huggingface.co/datasets/mjbommar/opengloss-v1.3-definitions) | Hub card: cc-by-4.0 | 1 | 909 | 727 |
| [dell-research-harvard/newswire](https://huggingface.co/datasets/dell-research-harvard/newswire) | Hub card: cc-by-4.0 | 1 | 896 | 92 |
| [strickvl/isafpressreleases](https://huggingface.co/datasets/strickvl/isafpressreleases) | Hub card: cc-by-sa-4.0 | 1 | 846 | 276 |
| [docs.manifold.markets/api](https://docs.manifold.markets/api) | Manifold Markets API data; non-commercial use only for AI training (Manifold terms) -- not for commercial model training | 28 | 789 | 308 |
| [mjbommar/opengloss-v1.3-dictionary](https://huggingface.co/datasets/mjbommar/opengloss-v1.3-dictionary) | Hub card: cc-by-4.0 | 1 | 644 | 460 |
| [Larxel/PHAENIX-1](https://huggingface.co/datasets/Larxel/PHAENIX-1) | Hub card: cc-by-nc-4.0 | 1 | 644 | 0 |
| [finnmok/congressional_bills](https://huggingface.co/datasets/finnmok/congressional_bills) | public domain (US Government) | 10 | 572 | 0 |
| [SnehaDeshmukh/IndianBailJudgments-1200](https://huggingface.co/datasets/SnehaDeshmukh/IndianBailJudgments-1200) | Hub card: cc-by-4.0 | 1 | 552 | 276 |
| [AlgorithmicResearchGroup/s2orc-cs-enriched](https://huggingface.co/datasets/AlgorithmicResearchGroup/s2orc-cs-enriched) | Hub card: none | 1 | 552 | 92 |
| [arxiv.org/help/oa](https://arxiv.org/help/oa) | CC-BY-4.0 / CC-BY-SA-4.0 / CC0-1.0 (per document, in meta) | 7 | 542 | 625 |
| [cy0307/awesome-loop-engineering](https://huggingface.co/datasets/cy0307/awesome-loop-engineering) | Hub card: cc0-1.0 | 1 | 506 | 175 |
| [momererkoc/social_bias_frames](https://huggingface.co/datasets/momererkoc/social_bias_frames) | CC-BY-4.0 | 23 | 464 | 222 |
| [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | Hub card: cc-by-nc-4.0 | 1 | 460 | 349 |
| [adugeen/personal-facts-msc](https://huggingface.co/datasets/adugeen/personal-facts-msc) | Hub card: cc-by-sa-4.0 | 1 | 460 | 92 |
| [gplsi/fake_job_postings_balanced_en](https://huggingface.co/datasets/gplsi/fake_job_postings_balanced_en) | Hub card: cc-by-4.0 | 1 | 460 | 0 |
| [github.com/spdx/license-list-data](https://github.com/spdx/license-list-data) | CC0-1.0 (SPDX licence list) | 15 | 441 | 0 |
| [NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026](https://huggingface.co/datasets/NeurIPS-1899-ED-2026/EpiBench-NeurIPS2026) | Hub card: cc-by-nc-sa-4.0 | 1 | 406 | 352 |
| [mteb/IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment) | Hub card: cc0-1.0 | 1 | 368 | 276 |
| [scbirlab/thomas-2018-spark-wt](https://huggingface.co/datasets/scbirlab/thomas-2018-spark-wt) | Hub card: cc-by-nc-4.0 | 1 | 368 | 184 |
| [einrafh/hnm-fashion-recommendations-data](https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data) | Hub card: none | 1 | 368 | 92 |
| [OpenSafetyLab/Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) | Hub card: apache-2.0 | 1 | 368 | 0 |
| [tmquan/anle-toaan-gov-vn](https://huggingface.co/datasets/tmquan/anle-toaan-gov-vn) | Hub card: cc-by-4.0 | 1 | 368 | 0 |
| [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | Hub card: apache-2.0 | 1 | 365 | 91 |
| [toxigen/toxigen-data](https://huggingface.co/datasets/toxigen/toxigen-data) | Hub card: none | 1 | 276 | 276 |
| [fahadhafeezofficial/cissp-llmbench](https://huggingface.co/datasets/fahadhafeezofficial/cissp-llmbench) | Hub card: cc-by-4.0 | 1 | 276 | 0 |
| [saattrupdan/womens-clothing-ecommerce-reviews](https://huggingface.co/datasets/saattrupdan/womens-clothing-ecommerce-reviews) | Hub card: none | 1 | 276 | 0 |
| [voilaj/swiss-caselaw](https://huggingface.co/datasets/voilaj/swiss-caselaw) | Hub card: cc0-1.0 | 1 | 252 | 0 |
| [annahbanannah/synthetic-math-toolcall-deception](https://huggingface.co/datasets/annahbanannah/synthetic-math-toolcall-deception) | Hub card: mit | 1 | 222 | 0 |
| [gtfintechlab/ipo-text](https://huggingface.co/datasets/gtfintechlab/ipo-text) | Hub card: cc-by-4.0 | 1 | 205 | 132 |
| [mjbommar/SHELF](https://huggingface.co/datasets/mjbommar/SHELF) | Hub card: cc-by-4.0 | 1 | 184 | 368 |
| [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) | Hub card: other | 1 | 184 | 92 |
| [data-is-better-together/fineweb-c](https://huggingface.co/datasets/data-is-better-together/fineweb-c) | Hub card: none | 1 | 184 | 92 |
| [jazzypajamas/mytown-local-gov-meetings](https://huggingface.co/datasets/jazzypajamas/mytown-local-gov-meetings) | Hub card: cc-by-4.0 | 1 | 184 | 92 |
| [laion/Scientific-Summaries](https://huggingface.co/datasets/laion/Scientific-Summaries) | Hub card: cc-by-4.0 | 1 | 184 | 92 |
| [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) | Hub card: apache-2.0 | 1 | 184 | 92 |
| [sileod/mindgames](https://huggingface.co/datasets/sileod/mindgames) | Hub card: apache-2.0 | 1 | 184 | 92 |
| [google-research-datasets/circa](https://huggingface.co/datasets/google-research-datasets/circa) | Hub card: cc-by-4.0 | 1 | 184 | 91 |
| [RevolutionCrossroads/nara_revolutionary_war_pension_files](https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files) | Hub card: cc0-1.0 | 1 | 184 | 0 |
| [almanach/Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched) | Hub card: none | 1 | 184 | 0 |
| [declare-lab/HarmfulQA](https://huggingface.co/datasets/declare-lab/HarmfulQA) | Hub card: apache-2.0 | 1 | 184 | 0 |
| [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Hub card: cc-by-sa-4.0 | 1 | 184 | 0 |
| [google/code_x_glue_cc_defect_detection](https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection) | Hub card: c-uda | 1 | 184 | 0 |
| [mnemoraorg/usgs-global-earthquake-catalog](https://huggingface.co/datasets/mnemoraorg/usgs-global-earthquake-catalog) | Hub card: ecl-2.0 | 1 | 184 | 0 |
| [nhop/OpenReview](https://huggingface.co/datasets/nhop/OpenReview) | Hub card: cc-by-nc-4.0 | 1 | 184 | 0 |
| [nvidia/Aegis-AI-Content-Safety-Dataset-2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | Hub card: cc-by-4.0 | 1 | 184 | 0 |
| [thegauravgiri/nepali-news-dataset](https://huggingface.co/datasets/thegauravgiri/nepali-news-dataset) | Hub card: mit | 1 | 184 | 0 |
| [yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K](https://huggingface.co/datasets/yatin-superintelligence/White-Hat-Security-Agent-Prompts-600K) | Hub card: cc-by-4.0 | 1 | 184 | 0 |
| [Exorde/exorde-social-media-one-month-2024](https://huggingface.co/datasets/Exorde/exorde-social-media-one-month-2024) | Hub card: mit | 1 | 183 | 92 |
| [data-is-better-together/10k_prompts_ranked](https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked) | Hub card: other | 1 | 183 | 92 |
| [witfoo/precinct6-cybersecurity-100m](https://huggingface.co/datasets/witfoo/precinct6-cybersecurity-100m) | Hub card: apache-2.0 | 1 | 159 | 65 |
| [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | Hub card: cc-by-4.0 | 1 | 153 | 0 |
| [edwarddgao/open-apply-jobs](https://huggingface.co/datasets/edwarddgao/open-apply-jobs) | Hub card: mit | 1 | 92 | 184 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | Hub card: apache-2.0 | 1 | 92 | 184 |
| [maastrichtlawtech/bsard](https://huggingface.co/datasets/maastrichtlawtech/bsard) | Hub card: cc-by-nc-sa-4.0 | 1 | 92 | 184 |
| [ysn-rfd/text-dataset-tiny-code-script-py-format](https://huggingface.co/datasets/ysn-rfd/text-dataset-tiny-code-script-py-format) | Hub card: apache-2.0 | 1 | 92 | 184 |
| [AI4Math/IneqMath](https://huggingface.co/datasets/AI4Math/IneqMath) | Hub card: cc-by-sa-4.0 | 1 | 92 | 92 |
| [J0nasW/science-datalake](https://huggingface.co/datasets/J0nasW/science-datalake) | Hub card: cc0-1.0,cc-by-4.0,cc-by-sa-4.0,cc-by-nc-sa-4.0,cc-by-nc-4.0 | 1 | 92 | 92 |
| [M-A-D/Mixed-Arabic-Datasets-Repo](https://huggingface.co/datasets/M-A-D/Mixed-Arabic-Datasets-Repo) | Hub card: none | 1 | 92 | 92 |
| [allenai/art](https://huggingface.co/datasets/allenai/art) | Hub card: unknown | 1 | 92 | 92 |
| [janavivekariya/aya_collection](https://huggingface.co/datasets/janavivekariya/aya_collection) | Hub card: apache-2.0 | 1 | 92 | 92 |
| [kareenamehta/ccnews](https://huggingface.co/datasets/kareenamehta/ccnews) | Hub card: none | 1 | 92 | 92 |
| [stanford-oval/ccnews](https://huggingface.co/datasets/stanford-oval/ccnews) | Hub card: none | 1 | 92 | 92 |
| [tmquan/phapdien-moj-gov-vn](https://huggingface.co/datasets/tmquan/phapdien-moj-gov-vn) | Hub card: cc-by-4.0 | 1 | 92 | 92 |
| [youssef101/artelingo-dummy](https://huggingface.co/datasets/youssef101/artelingo-dummy) | Hub card: mit | 1 | 92 | 92 |
| [DDSC/angry-tweets](https://huggingface.co/datasets/DDSC/angry-tweets) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [DDSC/lcc](https://huggingface.co/datasets/DDSC/lcc) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [MTEB-BR/factckbr](https://huggingface.co/datasets/MTEB-BR/factckbr) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [Noddybear/lies-v2](https://huggingface.co/datasets/Noddybear/lies-v2) | Hub card: mit | 1 | 92 | 0 |
| [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Hub card: cc-by-nc-4.0 | 1 | 92 | 0 |
| [RussianNLP/coat](https://huggingface.co/datasets/RussianNLP/coat) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [ScaleAI/fortress_public](https://huggingface.co/datasets/ScaleAI/fortress_public) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [SetFit/bbc-news](https://huggingface.co/datasets/SetFit/bbc-news) | Hub card: none | 1 | 92 | 0 |
| [Skyrmion/DAAD-X](https://huggingface.co/datasets/Skyrmion/DAAD-X) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [THU-KEG/RM-Bench](https://huggingface.co/datasets/THU-KEG/RM-Bench) | Hub card: odc-by | 1 | 92 | 0 |
| [Yusuf5/OpenCaselist](https://huggingface.co/datasets/Yusuf5/OpenCaselist) | Hub card: mit | 1 | 92 | 0 |
| [ai4bharat/indic_glue](https://huggingface.co/datasets/ai4bharat/indic_glue) | Hub card: other | 1 | 92 | 0 |
| [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | Hub card: other | 1 | 92 | 0 |
| [alisawuffles/WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [bastao/VeraCruz_PT-BR](https://huggingface.co/datasets/bastao/VeraCruz_PT-BR) | Hub card: none | 1 | 92 | 0 |
| [cw1521/ember2018-malware](https://huggingface.co/datasets/cw1521/ember2018-malware) | Hub card: none | 1 | 92 | 0 |
| [duarteocarmo/fineweb2-bagaco](https://huggingface.co/datasets/duarteocarmo/fineweb2-bagaco) | Hub card: odc-by | 1 | 92 | 0 |
| [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) | Hub card: cc0-1.0 | 1 | 92 | 0 |
| [google/boolq](https://huggingface.co/datasets/google/boolq) | Hub card: cc-by-sa-3.0 | 1 | 92 | 0 |
| [hezarai/sentiment-dksf](https://huggingface.co/datasets/hezarai/sentiment-dksf) | Hub card: none | 1 | 92 | 0 |
| [ibm-research/argument_quality_ranking_30k](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k) | Hub card: cc-by-3.0 | 1 | 92 | 0 |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [kaysss/leetcode-problem-solutions](https://huggingface.co/datasets/kaysss/leetcode-problem-solutions) | Hub card: mit | 1 | 92 | 0 |
| [kendx/NLP-ADBench](https://huggingface.co/datasets/kendx/NLP-ADBench) | Hub card: mit | 1 | 92 | 0 |
| [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) | Hub card: cc0-1.0 | 1 | 92 | 0 |
| [ltg/slide](https://huggingface.co/datasets/ltg/slide) | Hub card: mit | 1 | 92 | 0 |
| [masakhane/masakhanews](https://huggingface.co/datasets/masakhane/masakhanews) | Hub card: afl-3.0 | 1 | 92 | 0 |
| [medalpaca/medical_meadow_health_advice](https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice) | Hub card: none | 1 | 92 | 0 |
| [milistu/AMAZON-Products-2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023) | Hub card: mit | 1 | 92 | 0 |
| [mjbommar/opengloss-v1.3-query-examples-flat](https://huggingface.co/datasets/mjbommar/opengloss-v1.3-query-examples-flat) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [mteb/amazon_counterfactual](https://huggingface.co/datasets/mteb/amazon_counterfactual) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [mteb/toxic_conversations_50k](https://huggingface.co/datasets/mteb/toxic_conversations_50k) | Hub card: cc-by-4.0 | 1 | 92 | 0 |
| [napsternxg/wands](https://huggingface.co/datasets/napsternxg/wands) | MIT | 1 | 92 | 0 |
| [nyu-dice-lab/wavepulse-radio-summarized-transcripts](https://huggingface.co/datasets/nyu-dice-lab/wavepulse-radio-summarized-transcripts) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [open-index/hacker-news-rss](https://huggingface.co/datasets/open-index/hacker-news-rss) | Hub card: odc-by | 1 | 92 | 0 |
| [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification) | Hub card: none | 1 | 92 | 0 |
| [tasksource/arct2](https://huggingface.co/datasets/tasksource/arct2) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [tasksource/defeasible-nli](https://huggingface.co/datasets/tasksource/defeasible-nli) | Hub card: apache-2.0 | 1 | 92 | 0 |
| [tblard/allocine](https://huggingface.co/datasets/tblard/allocine) | Hub card: mit | 1 | 92 | 0 |
| [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | Hub card: unknown | 1 | 92 | 0 |
| [youngfish42/PaperVault](https://huggingface.co/datasets/youngfish42/PaperVault) | Hub card: gpl-3.0 | 1 | 92 | 0 |
| [sled-umich/SDN](https://huggingface.co/datasets/sled-umich/SDN) | Hub card: cc-by-nc-nd-4.0 | 1 | 91 | 0 |
| [BGPT-OFFICIAL/refute](https://huggingface.co/datasets/BGPT-OFFICIAL/refute) | Hub card: apache-2.0 | 1 | 80 | 0 |
| [Rapidata/text-2-video-human-preferences-seedance-1-pro](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-seedance-1-pro) | Hub card: apache-2.0 | 1 | 73 | 0 |
| [recursal/longbench-v2](https://huggingface.co/datasets/recursal/longbench-v2) | Hub card: apache-2.0 | 1 | 33 | 0 |
| [ComposoAI/PrimeBench](https://huggingface.co/datasets/ComposoAI/PrimeBench) | Hub card: apache-2.0 | 1 | 29 | 0 |
| [LidaSafety/fragbench](https://huggingface.co/datasets/LidaSafety/fragbench) | Hub card: cc-by-4.0 | 1 | 13 | 14 |
| [joey234/nan-nli](https://huggingface.co/datasets/joey234/nan-nli) | Hub card: cc-by-sa-4.0 | 1 | 12 | 25 |
| [open-index/open-arxiv](https://huggingface.co/datasets/open-index/open-arxiv) | Hub card: cc0-1.0 | 1 | 5 | 92 |
| [memo-ozdincer/jepa-qwen3-32b-pure-baselines-2026-05-25](https://huggingface.co/datasets/memo-ozdincer/jepa-qwen3-32b-pure-baselines-2026-05-25) | Hub card: cc-by-4.0 | 1 | 5 | 0 |
| [nickh007/cve-proof-corpus](https://huggingface.co/datasets/nickh007/cve-proof-corpus) | Hub card: apache-2.0 | 1 | 3 | 2 |
| [claritystorm/cfpb-consumer-complaints](https://huggingface.co/datasets/claritystorm/cfpb-consumer-complaints) | Hub card: other | 1 | 3 | 0 |
| [github.com/easonnie/ChaosNLI](https://github.com/easonnie/ChaosNLI) | CC-BY-SA-4.0 (ChaosNLI; Nie et al. 2020) | 25 | 0 | 4,641 (eval only) |
| [li2017dailydialog/daily_dialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog) | CC-BY-NC-SA-4.0 | 50 | 0 | 4,000 (eval only) |
| [CogComp/trec](https://huggingface.co/datasets/CogComp/trec) | no explicit licence; distributed freely by CogComp for research (Li & Roth 2002); testreal only | 45 | 0 | 3,999 (eval only) |
| [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | CC-BY-NC-4.0 | 46 | 0 | 2,400 (eval only) |
| [google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x) | PAWS-X terms: free for any purpose, acknowledge Google | 46 | 0 | 2,100 (eval only) |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Preotiuc-Pietro et al. 2019 complaints | 50 | 0 | 2,000 (eval only) |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: EmoBank, CC-BY-SA-4.0 (Buechel & Hahn 2017) | 50 | 0 | 2,000 (eval only) |
| [Blablablab/SOCKET](https://huggingface.co/datasets/Blablablab/SOCKET) | CC-BY-4.0 (SocKET aggregate, Choi et al. 2023); upstream: Jin et al. 2022 bragging | 50 | 0 | 2,000 (eval only) |
| [ColumbiaNLP/FLUTE](https://huggingface.co/datasets/ColumbiaNLP/FLUTE) | AFL-3.0 | 50 | 0 | 2,000 (eval only) |
| [SetFit/20_newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups) | public domain | 18 | 0 | 2,000 (eval only) |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | 50 | 0 | 2,000 (eval only) |
| [allenai/cosmos_qa](https://huggingface.co/datasets/allenai/cosmos_qa) | CC-BY-4.0 | 50 | 0 | 2,000 (eval only) |
| [allenai/scitail](https://huggingface.co/datasets/allenai/scitail) | Apache-2.0 | 50 | 0 | 2,000 (eval only) |
| [community-datasets/yahoo_answers_topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | Yahoo! Answers Comprehensive Q&A v1.0 (Webscope L6), research use (Hub card: unknown); testreal only | 45 | 0 | 2,000 (eval only) |
| [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | AG's corpus of news articles: free for non-commercial research use (Hub card: unknown); devreal only | 45 | 0 | 2,000 (eval only) |
| [metaeval/scruples](https://huggingface.co/datasets/metaeval/scruples) | Apache-2.0 | 50 | 0 | 2,000 (eval only) |
| [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | no licence stated (Maas et al. 2011, ai.stanford.edu/~amaas/data/sentiment) | 44 | 0 | 2,000 (eval only) |
| [strombergnlp/rumoureval_2019](https://huggingface.co/datasets/strombergnlp/rumoureval_2019) | CC-BY-4.0 | 50 | 0 | 2,000 (eval only) |
| [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | CC-BY-NC-SA-3.0 | 44 | 0 | 2,000 (eval only) |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT | 44 | 0 | 2,000 (eval only) |
| [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) | CC-BY-SA-3.0/OANC | 24 | 0 | 1,992 (eval only) |
| [pkavumba/balanced-copa](https://huggingface.co/datasets/pkavumba/balanced-copa) | CC-BY-4.0 | 50 | 0 | 1,499 (eval only) |
| [google-research-datasets/poem_sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | CC-BY-4.0 | 44 | 0 | 892 (eval only) |
| [CohereLabs/aya_collection](https://huggingface.co/datasets/CohereLabs/aya_collection) | Hub card: apache-2.0 | 1 | 0 | 184 (eval only) |
| [coldmind/reddit_dataset_94](https://huggingface.co/datasets/coldmind/reddit_dataset_94) | Hub card: mit | 1 | 0 | 184 (eval only) |
| [phreshphish/phreshphish](https://huggingface.co/datasets/phreshphish/phreshphish) | Hub card: cc-by-4.0 | 1 | 0 | 184 (eval only) |
| [yufan/arxiv-metadata-2020-2026](https://huggingface.co/datasets/yufan/arxiv-metadata-2020-2026) | Hub card: odc-by | 1 | 0 | 184 (eval only) |
| [12ml/e-CARE](https://huggingface.co/datasets/12ml/e-CARE) | Hub card: none | 1 | 0 | 92 (eval only) |
| [2A2I/Arabic_Aya](https://huggingface.co/datasets/2A2I/Arabic_Aya) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [Bose345/sp500_earnings_transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts) | Hub card: mit | 1 | 0 | 92 (eval only) |
| [Dr3dre/Genius-song-lyrics-cleaned](https://huggingface.co/datasets/Dr3dre/Genius-song-lyrics-cleaned) | Hub card: cc-by-4.0 | 1 | 0 | 92 (eval only) |
| [ParsiAI/FarsInstruct](https://huggingface.co/datasets/ParsiAI/FarsInstruct) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [Rapidata/text-2-video-human-preferences-wan2.1](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-wan2.1) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [RevolutionCrossroads/loc_chronicling_america_1770-1810](https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810) | Hub card: cc0-1.0 | 1 | 0 | 92 (eval only) |
| [aisbergpublicorganization/telegram-news-ua-dataset](https://huggingface.co/datasets/aisbergpublicorganization/telegram-news-ua-dataset) | Hub card: cc-by-4.0 | 1 | 0 | 92 (eval only) |
| [andrebadini/repojus](https://huggingface.co/datasets/andrebadini/repojus) | Hub card: cc-by-4.0 | 1 | 0 | 92 (eval only) |
| [codefuse-ai/F2LLM-v2](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [duarteocarmo/bagaco3](https://huggingface.co/datasets/duarteocarmo/bagaco3) | Hub card: other | 1 | 0 | 92 (eval only) |
| [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [laurievb/open-lid-dataset](https://huggingface.co/datasets/laurievb/open-lid-dataset) | Hub card: other | 1 | 0 | 92 (eval only) |
| [nyu-dice-lab/wavepulse-radio-raw-transcripts](https://huggingface.co/datasets/nyu-dice-lab/wavepulse-radio-raw-transcripts) | Hub card: apache-2.0 | 1 | 0 | 92 (eval only) |
| [pythainlp/wisesight_sentiment](https://huggingface.co/datasets/pythainlp/wisesight_sentiment) | Hub card: cc0-1.0 | 1 | 0 | 92 (eval only) |
| [sealuzh/app_reviews](https://huggingface.co/datasets/sealuzh/app_reviews) | Hub card: unknown | 1 | 0 | 92 (eval only) |
| [sonos-nlu-benchmark/snips_built_in_intents](https://huggingface.co/datasets/sonos-nlu-benchmark/snips_built_in_intents) | Hub card: cc0-1.0 | 1 | 0 | 92 (eval only) |
| [ucirvine/sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam) | Hub card: unknown | 1 | 0 | 92 (eval only) |
| [AgentAlphaAGI/Paper-Review-Dataset](https://huggingface.co/datasets/AgentAlphaAGI/Paper-Review-Dataset) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [Dk587/arctic](https://huggingface.co/datasets/Dk587/arctic) | Hub card: other | 1 | 0 | 0 (eval only) |
| [LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios) | Hub card: mit | 1 | 0 | 0 (eval only) |
| [PhilipMay/stsb_multi_mt](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt) | Hub card: other | 1 | 0 | 0 (eval only) |
| [RevolutionCrossroads/loc_chronicling_america_1770-1810_issues](https://huggingface.co/datasets/RevolutionCrossroads/loc_chronicling_america_1770-1810_issues) | Hub card: cc0-1.0 | 1 | 0 | 0 (eval only) |
| [RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs](https://huggingface.co/datasets/RevolutionCrossroads/nara_revolutionary_war_pension_files_PDFs) | Hub card: cc0-1.0 | 1 | 0 | 0 (eval only) |
| [VibrantVista/TTCW-Based-Review](https://huggingface.co/datasets/VibrantVista/TTCW-Based-Review) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [XIfr/Openalex-2005-2025](https://huggingface.co/datasets/XIfr/Openalex-2005-2025) | CC0-1.0 | 8 | 0 | 0 (eval only) |
| [cambridgeltl/vsr_random](https://huggingface.co/datasets/cambridgeltl/vsr_random) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [cambridgeltl/vsr_zeroshot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [cardiffnlp/super_tweeteval](https://huggingface.co/datasets/cardiffnlp/super_tweeteval) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [clue/clue](https://huggingface.co/datasets/clue/clue) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs) | Hub card: cc-by-nc-sa-4.0 | 1 | 0 | 0 (eval only) |
| [curaihealth/medical_questions_pairs](https://huggingface.co/datasets/curaihealth/medical_questions_pairs) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [deepghs/site_tags](https://huggingface.co/datasets/deepghs/site_tags) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [dlab-spp/safety-classifications](https://huggingface.co/datasets/dlab-spp/safety-classifications) | Hub card: odc-by | 1 | 0 | 0 (eval only) |
| [emrecan/stsb-mt-turkish](https://huggingface.co/datasets/emrecan/stsb-mt-turkish) | Hub card: none | 1 | 0 | 0 (eval only) |
| [google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws) | Hub card: other | 1 | 0 | 0 (eval only) |
| [jingjietan/kaggle-mbti](https://huggingface.co/datasets/jingjietan/kaggle-mbti) | Hub card: apache-2.0 | 1 | 0 | 0 (eval only) |
| [jmhessel/newyorker_caption_contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) | Hub card: apache-2.0 | 1 | 0 | 0 (eval only) |
| [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | Hub card: cc-by-nc-4.0 | 1 | 0 | 0 (eval only) |
| [marry-1111/x_dataset_0507238](https://huggingface.co/datasets/marry-1111/x_dataset_0507238) | Hub card: mit | 1 | 0 | 0 (eval only) |
| [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation) | Hub card: mit | 1 | 0 | 0 (eval only) |
| [moganai/turkishfineweb2-cleaned](https://huggingface.co/datasets/moganai/turkishfineweb2-cleaned) | Hub card: odc-by | 1 | 0 | 0 (eval only) |
| [mrinaldi/UsenetArchiveIT](https://huggingface.co/datasets/mrinaldi/UsenetArchiveIT) | Hub card: none | 1 | 0 | 0 (eval only) |
| [mteb/ArxivClassification](https://huggingface.co/datasets/mteb/ArxivClassification) | CC0-1.0 | 6 | 0 | 0 (eval only) |
| [nilc-nlp/assin2](https://huggingface.co/datasets/nilc-nlp/assin2) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [rdpahalavan/CIC-IDS2017](https://huggingface.co/datasets/rdpahalavan/CIC-IDS2017) | Hub card: apache-2.0 | 1 | 0 | 0 (eval only) |
| [rdpahalavan/UNSW-NB15](https://huggingface.co/datasets/rdpahalavan/UNSW-NB15) | Hub card: apache-2.0 | 1 | 0 | 0 (eval only) |
| [sbintuitions/JMTEB](https://huggingface.co/datasets/sbintuitions/JMTEB) | Hub card: cc-by-sa-4.0 | 1 | 0 | 0 (eval only) |
| [sbx/superlim-2](https://huggingface.co/datasets/sbx/superlim-2) | Hub card: none | 1 | 0 | 0 (eval only) |
| [scikit-fingerprints/LRGB_Peptides-func](https://huggingface.co/datasets/scikit-fingerprints/LRGB_Peptides-func) | Hub card: cc-by-nc-4.0 | 1 | 0 | 0 (eval only) |
| [scikit-fingerprints/MoleculeNet_BACE](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BACE) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [scikit-fingerprints/MoleculeNet_BBBP](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BBBP) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [scikit-fingerprints/MoleculeNet_PCBA](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_PCBA) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [scikit-fingerprints/MoleculeNet_Tox21](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Tox21) | Hub card: unknown | 1 | 0 | 0 (eval only) |
| [takschdube/moltbook-dataset](https://huggingface.co/datasets/takschdube/moltbook-dataset) | Hub card: cc-by-4.0 | 1 | 0 | 0 (eval only) |
| [tensorshield/reddit_dataset_30](https://huggingface.co/datasets/tensorshield/reddit_dataset_30) | Hub card: mit | 1 | 0 | 0 (eval only) |
| [wenknow/reddit_dataset_214](https://huggingface.co/datasets/wenknow/reddit_dataset_214) | Hub card: mit | 1 | 0 | 0 (eval only) |
| [wenknow/reddit_dataset_232](https://huggingface.co/datasets/wenknow/reddit_dataset_232) | Hub card: mit | 1 | 0 | 0 (eval only) |
