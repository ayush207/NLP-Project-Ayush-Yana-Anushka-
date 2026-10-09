# SkillSignal: Resume–Job Matching with a Writing-Style Audit

SkillSignal is a small NLP project exploring how well a resume matches a job description, and whether the wording of a resume can affect that match even when the candidate's skills stay the same.

The idea is to compare a simple word-based method with a sentence-embedding method, then inspect the results rather than treating a similarity score as a hiring decision.

## What it does

- Compares resume text with job descriptions using TF-IDF and cosine similarity.
- Compares the same text using the `all-MiniLM-L6-v2` sentence-transformer model.
- Uses a small skills dictionary to identify skills mentioned in a job description and resume.
- Reports ranking metrics for resume groups with existing fit labels.
- Runs a controlled experiment comparing concise and more verbose descriptions of fictional candidate profiles.

## Dataset

The notebook loads [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) through the Hugging Face `datasets` library.

The dataset has three columns:

- `resume_text`
- `job_description_text`
- `label`

The labels are `No Fit`, `Potential Fit`, and `Good Fit`. The downloaded splits contain 6,241 training rows and 1,759 test rows.

For the initial ranking experiment, the notebook looks for job descriptions containing “business analyst”. This produces 240 rows across 10 unique job descriptions. The controlled writing-style experiment uses 15 fictional profiles with predefined skill sets.

The fictional profiles are useful for testing the effect of rewriting in a controlled setting, but they are not evidence of how all real candidates or hiring systems behave.

## Methods

### Text matching

**TF-IDF** represents text using the importance of words in the selected corpus. Cosine similarity is then used to compare each resume with the job description.

**Sentence embeddings** use `sentence-transformers` with the `all-MiniLM-L6-v2` model to represent the text as vectors. Cosine similarity compares those vectors.

### Skill extraction

A manually defined dictionary covers skills such as SQL, Excel, requirements gathering, stakeholder management, Agile, documentation, process improvement, communication, project management, and data analysis.

The current extractor uses phrase matching. It is simple and easy to inspect, but it can miss skills expressed with different wording.

### Ranking evaluation

The notebook calculates Precision@5, Recall@5, and NDCG@5 for resume groups with fit labels. The labels are also mapped to binary and graded relevance values.

These metrics depend on the dataset's labels and on which job groups are included. The evaluation output should be checked carefully before reporting final results: the notebook output says seven job postings were eligible, while the printed metric summaries say they were averaged across three jobs. That inconsistency needs to be resolved.

### Writing-style audit

The controlled experiment creates 15 fictional candidate profiles and writes concise and verbose versions for each profile. It then compares their scores and ranks under both matching methods.

In the current notebook output:

- Mean absolute TF-IDF score change: **0.1866**
- Mean absolute embedding score change: **0.0804**
- TF-IDF ranking changed for **93.3%** of profiles
- Embedding ranking changed for **80.0%** of profiles

These are preliminary results from a small, constructed experiment. They should not be interpreted as a general measure of hiring bias. The profiles, rewrites, and ranking setup need further checking before drawing strong conclusions.

The current skill-extraction check also reports that at least one skill was missed in all 15 verbose profiles. Since the extractor relies on exact phrase matching, this likely reflects a limitation of the keyword dictionary as well as the wording changes.

## Running the notebook

The notebook was developed in Google Colab. To run it:

1. Open the notebook in Google Colab.
2. Install the required packages:

   ```python
   !pip install -q datasets sentence-transformers scikit-learn pandas matplotlib
   ```

3. Run the cells from top to bottom.
4. Allow the dataset and sentence-transformer model to download when prompted by the code.

The main libraries used are:

- `datasets`
- `pandas`
- `scikit-learn`
- `sentence-transformers`
- `numpy`
- `matplotlib`

The plotting section currently stops with a `NameError` because `plt` is not defined in the executed session. Add `import matplotlib.pyplot as plt` before the plotting cell, then rerun it.

## Current limitations and next steps

This is a work in progress. Before treating the results as final, the next steps are:

1. Fix the plotting error and rerun the notebook from the beginning.
2. Investigate why the ranking evaluation reports seven eligible job groups but averages results across three.
3. Verify the relevance labels and ranking metrics against a few examples manually.
4. Improve skill extraction so it can recognise equivalent phrases, not just exact keyword matches.
5. Review the generated concise and verbose profiles to confirm they preserve the same underlying qualifications.
6. Add a clear explanation of matched and missing skills for an example resume–job pair.
7. Report the sample sizes and limitations alongside every result.

A similarity score is not a probability of being hired, and it does not measure a candidate's actual job performance. SkillSignal is intended as an experiment in explainable text matching, not as an automated hiring decision-maker.

## Project status

**Prototype in progress.** The basic matching methods and an initial controlled writing-style audit are implemented. Evaluation checks, plotting, and the explainability section still need refinement.
