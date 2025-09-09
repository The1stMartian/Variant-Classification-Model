![Banner](./media/legoBanner.jpg)

# Predicting the effect of genetic variants
<i>A machine learning model for identifying pathogenic variants </i>

## Highlights:
- A histogram-based gradient boosting classification tree that predicts variant effects (likely "benign" or "pathogenic")
- Accuracy ~93%
- Type I error rate: 2.5% 
- Type II error rate: 4.5%
- Effective on 90% of variants

## Input/Output:
- Input - one genetic variant
- Output -  "benign" or "pathogenic"
- Example: chr1, position 123456, A>G
- <i>This is sufficient to predict benign or pathogenic status with high confidence.</i>

## Rationale
<b>Briefly:</b> Not all genetic variants are understood. This leaves a gap in our knowledgebase that could be filled via predictive modelling. This ML model is intended to fill that gap by making highly accurate predictions based upon millions of publicly available datapoints.

Many of us are interested in understanding our own genetics, often paying to have our own genomes commercially sequenced. However understanding the results of whole-genome sequencing can be a challenge due to the size of the genome and subsequent potential for unique variants to arise. Companies like 23AndMe provide general information about known variants detected in a patient sample, which is certainly helpful. However, rare or unique variants may have little or no data to support an interpretation, leaving patients to speculate about potential effects.<br>

<b>Basic Idea:</b> To offer clarity on the potential effects of unreported or under-reported genomic changes, I developed a machine learning model capable of predicting their effect. Here I used scikit-learn's Histogram-based Gradient Boosting Classification Tree model to interpret the clinical effect of single-nucleotide polymorphisms, insertions, deletions, synonymous, and non-synonuymous changes among others. Using data pooled from several sources including <b>ClinVar, Gnomad, MPC, SpliceAI, CADD,</b> and <b>REVEL,</b> I created a rich set of training dataset with high-confidence outcomes reported by the NIH's [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (for Clinical Variant) database. By virtue of the diverse data included in the training dataset, broad expectations about mutation tolerance of whole genes as well as more granular information about mutations in specific locations, even including the effects of synonymous/silent mutations are built into the model. Providing only the chromosome number, position and change (e.g. Chr1, 5326192, A>G) of a given variant, the model classifies the outcome into a simple binary prediction: "benign" or "pathogenic" with high (>95%) confidence. This allows users to gain insight into a given variant using only minimal information.<br>

## Prediction overview
Users can enter any mutation into the model. The first step is to look up the variant in the existing database (compiled by me as part of the training data creation) to see if clinical information has already been established for that change. If so, the effect is reported. If not, the Python script looks up additional features associated with the variant using a fasta formatted genome, and access to the raw information library I created while building the training dataset. With those data fields (fully described below), the query is complete and the model can use it to predict the clinical effects.

## Training library creation
I instantiated my training datbase using the full ClinVar [dataset]([https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi), available on NCBI's website. I split multiallelic entries into separate lines using bcftools then ensured proper variant coordinate calling using left-alignment. This shifts the start position of a variant to their left-most location, potentially normalizing/fixing issues wiht indel coordinate calling and allowing for consistent variant identification across databases.
I then downloaded the [Gnomad](https://gnomad.broadinstitute.org/) genome and Gnomad exome databases which are quite large, and contain disparate information. Gnomad exome has far more variants (~180 million), but they are limited to exons. Gnomad genome, has fewer samples  (~760 million), but includes variants from the full genome, allowing for detection of intronic and other non-coding variants. As such, the two databases have useful, though potentially overlapping variant lists. To mine thse datasets I used the bcftools annotate function which matches variants using the chromosome name, position, reference allele, and non-canonical allele. From the exome and genomic datasets, I collected the variant allele count and frequency, as well as the total allele count (all alleles). I also collected the "nhomalt" metric which is the number of homozygous-alternative individuals, an extremely informative metric. To prevent duplicating or misannotating records, I left-aligned and normalized the combined dataset.<br><br>
In addition to information about specific variants, I expected that gene-level data would also be helpful for predicting clinical outcomes by providing information about the general tolerance of individual genes to genetic alterations. Accordingly, I downloaded [gene-level data](aws s3 cp s3://gnomad-public-us-east-1/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv gsm.tsv) from Gnomad including constraint scores such as <b>LOEUF, pLI, missense intolerance, and synonymous mutation tolerance</b>. I added them to the training data by converting the gtf-formatted data to a .bed file, and and annotated my compiled data file using bcftools' annotate function.<br>
Next I added data from dbNSFP - a [database](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-020-00803-9) with pre-compiled information from a variety of sources including <b>CADD, REVEL, SpliceAI, MPC</b>. Lacking an academic email address, I was forced to download an older copy of the database which lacks <b>SpliceAI</b> data. This necessitated additional, separate downloads and matching of the SpliceAI's SNV data from [Ensemble]( https://ftp.ensembl.org/pub/data_files/homo_sapiens/GRCh38/variation_plugins/spliceai_scores.masked.snv.ensembl_mane.grch38.110.vcf.gz) and [indel data](https://molgenis26.gcc.rug.nl/downloads/vip/resources/GRCh38/spliceai_scores.masked.indel.hg38.vcf.gz) from a community server. I also downloaded <b>regional intolerance data</b> (CCR) including both [autosomes](https://ccrs.s3.us-east-2.amazonaws.com/ccrs/ccrs.autosomes.v2.20180420.bed.gz) and the [X](https://ccrs.s3.us-east-2.amazonaws.com/ccrs/ccrs.xchrom.v2.20180420.bed.gz) chromosome (no Y-chromosome data for some reason.). These data provided quantitative measurements about variants within coding intervals that are unusually depleted of any non-synonymous changes in large human cohorts. Last I incorporated [transcript-level variant effect predictions](https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/Homo_sapiens.GRCh38.115.chr.gff3.gz)] from <b>Ensemble</b>. At this point, the data were ready for cleanup.<br>

## Data cleanup and imputation
After exporting my final annotated .vcf file to .tsv (.gz!), there were a number of issues with data formatting. Specifically, dispatate data fields form SpliceAI were combined into a single pipe-delimted string. Similarly, mutation type and conservation were all packed into a single entry like: "missense|OR4F5|ENST00000641515|protein_coding|+|36E>36G|69134A>G". Fortunately, unpacking these columns in Python was trivial and the resulting data file was ready for import into a Pandas data frame. 
My Jupyter notebook (cleanTrainingData.ipynb) demonstrates my process of data field engineering. Steps included one-hot encoding of mutation type: from a column of factors like "missense"/"non-coding"/"synonymous" dummy columns were produced, one per factor, with a 0 or 1 value indicating the mutation was or was not a missense mutation, was or was not a non-coding mutation, etc. Target classes were filtered to clear "benign" or "pathogenic" classifications. i.e. Entries with "conflicting effect" were removed. At that point, the initial data fields, Chrom, pos, ref, and alt were removed as they are not directly informative and were only used to collect the other information fields.<br>


## Training data
<i>The final training data contains 3,674,815 variants and 34 features.</i>

|Field|Database of origin|Short description|
|:-------|:------|:------|
|nhomalt|gnomAD (exomes/genomes)|Number of homozygous-ALT individuals at the site.|
|revel|dbNSFP (REVEL)|REVEL pathogenicity score (0–1) for missense SNVs.|
|cadd_phred|dbNSFP (CADD)|CADD PHRED-scaled deleteriousness (higher = more deleterious).|
|cadd_raw|dbNSFP (CADD)|CADD raw score (unnormalized).|
|mpc|dbNSFP (MPC)|Missense badness/PolyPhen/Constraint score (missense SNVs).|
|phyloP100|dbNSFP (phyloP)|100-way phyloP conservation (higher = more conserved).|
|phastCons100|dbNSFP (phastCons)|100-way phastCons conservation probability (0–1).|
|gerp_rs|dbNSFP (GERP++)|GERP++ rejected substitutions (higher = more constrained).|
|loeuf|gnomAD constraint (by gene)|LoF o/e upper bound; lower = more LoF-intolerant gene.|
|pli|gnomAD constraint (by gene)|Probability of LoF intolerance (0–1; higher = more intolerant).|
|mis_z|gnomAD constraint (by gene)|Missense depletion Z-score (higher = more constrained).|
|syn_z|gnomAD constraint (by gene)|Synonymous Z-score (near 0 typically; control metric).|
|clnsig|ClinVar|Raw clinical significance term(s) (e.g., P/LP/LB/B/VUS).|
|ccr_pct|CCR (Constrained Coding Regions)|Percentile of regional coding constraint at position (0–100).|
|ref_len|Derived (REF/ALT)|Length of REF allele.|
|alt_len|Derived (REF/ALT)|Length of ALT allele.|
|is_indel|Derived (REF/ALT)|1 if not a single-base change; else 0.|
|is_snv|Derived (REF/ALT)|1 if single-nucleotide variant; else 0.|
|indel_len|Derived (REF/ALT)|abs(len(ALT) − len(REF)).|
|is_transition_num|Derived (REF/ALT)|SNVs: 1=transition, 0=transversion; −1 if not applicable.|
|spliceai_tx_count|Derived (SpliceAI INFO)|Number of transcript records parsed from SpliceAI string.|
|spliceai_ds_ag_max|SpliceAI precomputed VCF|Max ΔScore (0–1) for acceptor-gain across transcripts.|
|spliceai_ds_al_max|SpliceAI precomputed VCF|Max ΔScore for acceptor-loss.|
|spliceai_ds_dg_max|SpliceAI precomputed VCF|Max ΔScore for donor-gain.|
|spliceai_ds_dl_max|SpliceAI precomputed VCF|Max ΔScore for donor-loss.|
|ccr_top1|Derived (CCR)|1 if ccr_pct ≥ 99 (top 1% constrained), else 0.|
|is_frameshift|Derived (BCSQ/CSQ)|1 if any transcript consequence is frameshift; else 0.|
|label|Derived (ClinVar)|Training label: 1 = Pathogenic/Likely Pathogenic; 0 = Benign/Likely Benign.|
|is_missense|Derived (BCSQ/CSQ)|1 if any transcript consequence is missense; else 0.|
|mpc_filled|Derived (MPC + missense flag)|MPC where applicable; −1 sentinel when not missense.|
|mpc_is_missing|Derived|1 if MPC is missing among missense rows; else 0.|
|mt_missense|Derived (one-hot)|Mutation-type one-hot: missense.|
|mt_noncoding|Derived (one-hot)|Mutation-type one-hot: noncoding.|
|mt_nonsense|Derived (one-hot)|Mutation-type one-hot: nonsense/stop-gained.|
|mt_silent|Derived (one-hot)|Mutation-type one-hot: synonymous/silent.|


## Model Scoring
Given the richness of the training data, it is perhaps not surprising that the initial training produced a highly effective model with an overall accurately score of 93%.<br>

![Confusion Matrix 1](./media/cm1.jpg)<br>

 To further refine its accuracy, I examined the probabily scores (of a "beign" or "pathogenic" classification). Scores close to the 0.5 mark are ambiguous, potentially producing errors. Accordingly, I tried multiple confidence cutoffs and did a cost/benefit analysis:<br>

|Cutoff|Remaining Sample %|Type I Rate|Type II Rate|
|:-------|:------|:------|:------|
|>0.6 or <0.4|95%|5.2|5.8|
|>0.75 or < 0.25|90%|2.5|4.5|
|>0.9 or < 0.1|78%|1.0|2.9|

As shown by the resulting confusion matrices (below), the various cutoffs caused a meaningful decline in type I and type II errors while reducing the number of samples that could be analyzed. As a good balance, the >0.75 <0.25 confidence filter provided the best blend of maximizing the number of analyzable samples, while providing an accuracy score of >95%. Accordingly, I moved forward with these paramters. <br>
![Confusion Matrix 2](./media/cm2.jpg)<br>

## Real-World Use Case:
The NCBI ClinVar data contains a large number of samples (specifically 1,950,044) with conflicting reports about pathogenicity. These were removed from my dataset prior to training, but contain all the data fields necessary for analysis. As these mutations could conceivably be clinically relevant, I propse that inferring their effects could be valuable. Therefore, I entered them into my model and observed <b>995,382 variants expected to be pathogenic</b> without applying a probability cutoff. After increasing the stringency to >0.75 and < 0.25 which should yield a <b>95% confidence level</b>, the number of predicted pathogenic variants dropped to the still significant number of <b>493,310 predicted pathogenic variants</b>. 

## Conclusions
This exercise in model building is intended to showcase my ability to build an effective dataset for training a machine learning model, and to produce a model with a real-world use-case. I also hoped the model would produce results that are interesting and useful, not just for biologists, but ideally for the general public as well. By setting up a model capable of yielding useful and informative results using only the basic information associated with a genomic variant (chromosome name, position, reference base, and alternative base) this model could be used by anyone interested in understanding changes in their own genome that aren't currently explained via robust clinical data.<br><br>
This project isn't just a bioinformatics project, it also fits more broadly under the umbrella of data science. Accordingly, it is intended to showcase my datascience capabilities as much as my bioinformatics skills. Of course, I proposed a use-case within the biology sphere as it provided the opportunity to demonstrate technical abilities germane to my profession - bioinformatics - and to demonstrate specific skills such as integrating datasets from multiple public repositories. This is a common need in bioinformatics as "multiomics" has become a major interest in both biotech and academic biology.

## Future Work
I built this ML model to be usable for individuals who have only basic variant information: chromosome name, position, reference and alternative base. However, the model requires a full set of information from Gnomad, Clinvar, dbN