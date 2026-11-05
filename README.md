# **Predicting the Price of Health: Understanding What Drives Future Spending**

#### *Primary Question: How accurately can we predict an adult’s next-year healthcare spending from current-year information, and which types of information (prior spending and use, diagnosed conditions, health status, or healthcare engagement) contribute most to the predictions?*
---

### **OVERVIEW:**
*Background and our purpose*

The Medical Expidenture Panel Survey is a nationally represented survey of the U.S. civilian noninstitutionalized population that follows households across multiple interview rounds to collect detailed info on health status, medical conditions, insurance, healthcare use, and spending. Since it follows individuals over the years, the MEPs is an excellents source to measure how one year's healthcare costs compare to the next. We're utilizing the [MEPS HC-257: MEPS Panel 28 Longitudinal Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-257&prfricon=yes&utm_source=chatgpt.com) from the **2023-2024 period**. We are also utilizing the [2023 Medical Conditions File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_results.jsp?cboDataYear=All&cboDataTypeY=1%2CHousehold+Full+Year+File&buttonYearandDataType=Search&cboPufNumber=All&SearchTitle=Medical+Conditions) to factor in reported conditions for our models.

---

### **MODELING:**
*Using the variables. Veronica is just thinking through the process here lol; will clean up later*

OUTPUT VARIABLE[S]

We are predicting total healthcare spending in 2024— the TOTEXPY2 column from the original dataset (or 'tot_exp_y2' in our cleaned meps_healthcare_expenditures_v1.csv dataset). We're also considering the following transformed- versions of the 2024 expidentures variable. But regardless of which we use, 2024 total expidentures is what we are predicting.

> tot_exp_y2: raw dollar amount
> log_tot_exp_y2: raw dollar amount is heavily skewed— hence we also log it
> high_cost_y2: 1 if the person is in the top 20% of 2024 spenders otherwise 0

INPUT VARIABLE[S]

> [!] Rule: keep any 2024 variables out of the predictors! You'll have to cross check with the data dictionary to make sure you're following this. Also "_y2" sometimes indicates 2024.

We are considering including the 2023 expenses as a predictor (tot_exp_y1)— obviously. But an interesting thing to deep dive into would be *How well can we predict costs without prior spending data?*. Let's challenge ourselves to build a model without the 2023 expenses perhaps?

You'll also notice a bunch of ccsr_LETTERSANDNUMBERS columns too— those are indicators for if that person has a certain disease. You can look up "ccsr CIR000" and "Diseases of the circulatory system, generic or masked." for instance. Let's use these disease presence columns as predictors as well. 

The rest of the columns should be self explanatory maybe— use the "Variables Selected from main_df" codechunk in costs_conditions_dataset_setup.ipynb** to find out the original names (from the MEPS data dictionary) if any of them confuse you.

As for where we should take our project, here's some next steps suggested by Claude. I was thinking of setting up individual ipynb files for each of you guys where you can each have an individual focus. 

Suggested project structure

EDA: look at the outcome distribution, how concentrated spending is, and the correlations.

Baseline: “next year = last year” (tot_exp_y1 alone).

Models: linear regression, then Lasso (to select variables), then a tree model (random forest or gradient boosting).

Block comparison: add the predictor blocks one at a time and see which ones raise test R².

Classification: a logistic regression and a tree model for high_cost_y2.

Interpretation: the coefficients Lasso keeps, the tree model’s feature importances, and what all of it says about what drives future costs.

_Will resume discussion with team here._

---

### **NAVIGATION:**
*Making sense of the files in the repository*

**costs_conditions_dataset_setup.ipynb** is a cleaned, merged, and standardized version of the original datasets— if columns are renamed, you can locate the original names in the "Variables Selected from main_df" code chunk and cross check with the [MEPS data dictionary](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H257&utm_source=chatgpt.com).

**meps_healthcare_expenditures_v1.csv** is the primary, version 1 of the dataset we will be modeling with– the output from the previous, setup file

*UPDATE AS MORE FILES GET ADDED*