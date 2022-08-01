# Readme
This is the readme for the anonymized and obfuscated data from the SOAR (AI ATAC 3) user study.

The experiment, data, and our results are found in our paper. Please cite our paper when using this data or code.

(add citation once online)




## file: `anonymized_obfuscated_data.csv`

### Text description
The data in this table is not the "raw" data. It has been user- and SOAR-tool anonymized; SOAR tool names have been removed, as have participant user ids. Probabilistic techniques to obfuscate the data by adding noise while retaining its characteristics have been performed. To ensure utility of the anonymized data, we provide code to reproduce the Subcategory Average Correlation heatmap, found in the publication.

While the SOAR tool names have been removed,  a binary indicator for if the test data per row is from a SOAR tool or from a baseline environment (full SOC tool suite but no SOAR tool).

The likert columns correspond to questions posed to the users (participants) during and after working investigations with a SOAR tool or in a baseline environment (full SOC tool stack but with no SOAR tool).

Sentiment columns correspond to answers given by analysts to free response questions. Transcriptions of their responses are quantified with sentiment analysis.

The ticket columns report quality and completion of tickets from the test investigations based on percent of correctness according to a gold standard (rubric).

The window swapping and time columns report the number of times the participant changed windows in an investigation (quantifying context switching), and the duration of the investigation (quantifying efficiency), respectively.

User identification information has been removed.

There are four columns of user/participant demographics documenting their familiarity with SOAR tools, job role (SOC analyst or other), number of years working in their security-related job role, and the number of different security operation centers (SOCs) they have supported. These have all been converted to binary variables.


### Data Format
The table is simply a CSV and has the following columns/formats:

#### tool column
- `tool_label` column: binary string ('0' = Baseline environment w/ no SOAR tool, '1' = with a SOAR tool)

#### measurement data columns:
- `likert-*` columns are floats, (originally in {1, ..., 5}) have been normalized by subtracting the user's mean (landing in [-4,4]), then linearly scaled to [0,1] preserving orientation (larger is better)
- `sentiment-*` columns are floats (originally in [-1,1]) have been normalized by subtracting the user's mean (landing in [-2,2]), then linearly scaled to [0,1] preserving orientation (larger is better)
- `ticket-*` columns are floats (originally in [0,1]) have been normalized by subtracting the scenario-type mean (landing in [-1,1]), then linearly scaled to [0,1] preserving orientation (larger is better)
- `swaps-*` and `time-*` columns marker are floats, (originally in $[0, \infty)$) have been normalized by subtracting the scenario-type mean, then then linearly scaled to [0,1] preserving orientation (smaller is better)

#### user (participant) demographic columns
- `familiarity` column: binary string ("0" = NO familiarity with SOAR tools, "1" = some or expert familiarity with SOAR tools)
- `role` column: binary ("0" = NOC operator or other, "1" = SOC operator)
- `socs` column: binary ("0": <=1 SOC and "1": >1 SOC )
- `years` column: binary ("0": <=2 years and "1": >2 years)


## `code/` folder
`imputation_correlation.py` and `predict_missing_values.py` are called by `subcategory_averages_correlation.py`
descriptions of these algorithms reside in the paper.

- `subcategory_averages_correlation.py`
    - reads in the anonymized_obfuscated_data.csv,
    - creates the subcategory matrix (columns average over subcategories)
    - makes a plot of subcategory ave data's correlation (w/ nan's before imputation)
    - imputes missing data by (1) sampling from the DT-neighbor-informed KDEs and (2) sampling uniformly in [0,1]
    - plots subcategory ave data's correlation after imputing
    - makes the the convergence plot for the average correlation matrix (step change per imputation)
    - makes the correlation confidence intervals for a random selection of variables.
    - calls imputation_correlation.py and predict_missing_values.py functions
