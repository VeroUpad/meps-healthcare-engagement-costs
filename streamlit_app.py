# -*- coding: utf-8 -*-
"""
Predicting the Price of Health Dashboard

HOW TO RUN
    conda activate [env]           #replace [env] with any environment with pandas, numpy and plotly
    pip install streamlit            #if you do not have streamlit, install it once
    streamlit run streamlit_app.py   #run this from the repo folder

WHERE TO CHANGE THINGS  (search for the numbered banners, ex: "[1] VARIABLE")
    [0] CONFIG: file paths, colors, chart template, default values
    [1] VARIABLE: labels and predictor blocks for columns
    [2] DATA LOADING: reading the CSV and building grouped columns
    [3] STATISTICS: group summaries, confidence intervals, correlations
    [4] CHARTS: one function per Plotly figure
    [5] PAGE: header + sidebar filters
    [6] PAGE: KPI metric row
    [7] PAGE: spending-at-a-glance charts
    [8] PAGE: tabs
    [9] PAGE: footer expanders
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(
    page_title="MEPS - Healthcare Spending Drivers",
    page_icon=":hospital:",
    layout="wide",
)


pio.templates.default = "none"

#[0]CONFIG
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "meps_health_expenditures_v1.csv"  #point at v2 when it exists
LOGO_PATH = APP_DIR / "images" / "logo.png"  #an emoji is shown if missing
MODEL_OUTPUT_DIR = APP_DIR / "model_outputs"  #MODEL HOOK

#Key columns
OUTCOME = "tot_exp_y2"  #total 2024 healthcare spending ($)
OUTCOME_LOG = "log_tot_exp_y2"  #log(1 + tot_exp_y2)
HIGH_COST = "high_cost_y2"  #1 = top 20% of 2024 spenders
PRIOR_SPEND = "tot_exp_y1"  #total 2023 spending ($)
ID_COL = "dupersid"
NOT_PREDICTORS = {OUTCOME, OUTCOME_LOG, HIGH_COST, ID_COL}

#Statistics settings
N_BOOT = 400  #bootstrap resamples for mean/median CIs (more = smoother but slower)
RANDOM_SEED = 42  
DEFAULT_MIN_N = 30  #groups with fewer adults are hidden from charts (sidebar slider)

DEFAULT_EXPLORER = ("conditions", "n_chronic")  #(block, variable) in the explorer tab
DEFAULT_HEATMAP = ("age", "insurance")  #(rows, columns) in the two-variable tab
DEFAULT_TOP_N = 20  #bars in the "strongest associations" chart
DEFAULT_HIGHLIGHT = "prior_use"  #block highlighted in that chart ("" = none)

#sidebar filters: (column, label). Add any categorical column from the registry.
SIDEBAR_FILTERS = [
    ("sex", "Sex"),
    ("race_eth", "Race / Ethnicity"),
    ("insurance", "Insurance Coverage (2023)"),
    ("poverty_cat", "Family Income (2023)"),
    ("region", "Census Region"),
    ("tot_exp_y1", "Total Spending in 2023"),
]

#fields in the "find similar adults" form (tab[8]): (column, label).
PROFILE_FIELDS = [
    ("age", "Age"),
    ("sex", "Sex"),
    ("insurance", "Insurance Coverage (2023)"),
    ("n_chronic", "Self-Reported Chronic Conditions"),
    ("tot_exp_y1", "Total Spending in 2023"),
    ("self_rated_health", "Self-Rated Health (2023)"),
]

#the spending measures a user can pick in the sidebar. "unit" changes formatting.
STATS = {
    "mean": {"label": "Mean 2024 Spending", "short": "Mean $", "unit": "$"},
    "median": {"label": "Median 2024 Spending", "short": "Median $", "unit": "$"},
    "pct_high": {"label": "% High-Cost (top 20% in 2024)", "short": "% High-cost", "unit": "%"},
    "pct_zero": {"label": "% spent $0 in 2024", "short": "% spent $0", "unit": "%"},
}

#spending bands (lower bound included, upper bound excluded)-
SPEND_EDGES = [0, 1, 500, 2_000, 5_000, 10_000, 25_000, 50_000, np.inf]
SPEND_LABELS = ["$0", "$1–499", "$500–1,999", "$2,000–4,999", "$5,000–9,999",
                "$10,000–24,999", "$25,000–49,999", "$50,000+"]
SPEND_LABELS_SHORT = ["$0", "$1–499", "$500–2K", "$2K–5K", "$5K–10K", "$10K–25K",
                      "$25K–50K", "$50K+"]

#Colors taken from the Plotly's "seaborn" template and checked with a colorblind-safety validator
#A second set is used when a viewer switches Streamlit to dark mode
#(Settings -> Theme).
FONT_FAMILY = '"Source Sans", "Source Sans Pro", sans-serif'  #Streamlit's own font
COLORS = {
    "light": {
        "paper": "#FFFFFF",  # chart background outside the plot area
        "plot": "#EAEAF2",  #plot area
        "grid": "#FFFFFF",  #seaborn's white gridlines
        "font": "#242424",
        "surface": "#EAEAF2",  #plot area used for thin gaps between touching marks
        "primary": "#264653",  #single-series bars and lines
        "secondary": "#7A9E9F",  #second shade of the same family (2023 values, "all adults")
        "higher": "#E34948",  #group clearly ABOVE the overall value (95% CI excludes it)
        "lower": "#2A78D6",  #group clearly BELOW the overall value
        "neutral": "#A3A3A3",  #not clearly different from overall
        "muted": "#B7BDC7",  #blocks that are not highlighted
        "ink": "#262730",  #reference lines, error bars, annotations
        "band_ramp": ["#72A8B6", "#56909F", "#3D7886", "#2B616D", "#224750"],  # $0 -> top 20%
        "heat_scale": [[0.0, "#D5E5E8"], [1.0, "#224750"]],  
        "table_header": "#D5E5E8",  #light teal, so the dark header text is easy to read
        "table_header_text": "#262730",
        "table_cell": "#F0F2F6",
        "table_text": "#262730",
        "tint_higher": "#F8D7D6",
        "tint_lower": "#D6E5F7",
        "tint_small": "#DCDCE0",
    },
    "dark": {
        "paper": "rgba(0,0,0,0)",  #shows Streamlit's dark page (#171614)
        "plot": "#211F1C",
        "grid": "#3A3732",
        "font": "#EDEBE8",
        "surface": "#211F1C",
        "primary": "#6FB0C0",
        "secondary": "#3D7886",
        "higher": "#E66767",
        "lower": "#3987E5",
        "neutral": "#B0B0B0",
        "muted": "#55524C",
        "ink": "#E6E6E6",
        "band_ramp": ["#285E6A", "#357685", "#4B92A2", "#6FB0C0", "#97CFDD"],
        "heat_scale": [[0.0, "#16303A"], [1.0, "#97CFDD"]],
        "table_header": "#264653",
        "table_header_text": "#FFFFFF",
        "table_cell": "#2D2B27",
        "table_text": "#FFFFFF",
        "tint_higher": "#5A2A2C",
        "tint_lower": "#22385A",
        "tint_small": "#3A3732",
    },
}

MISSING = "Missing"  #label for missing / not-asked values (shown as its own group)
HIGHER, SIMILAR, LOWER, TOO_FEW = (
    "Higher than overall", "Not clearly different", "Lower than overall", "Too few adults")

#[1] Variables: one entry per predictor column
#label: short name (for dashboard)           
#block: predictor block (BLOCKS)
#meps: original MEPS variable 
#desc: one-line definition (from the codebook)
#kind: how values become groups
#binary: 0/1 for No/Yes
#coded: small integer codes that represent labels (ex: 1 = Excellent, 5 = Poor)
#category: text values are labels in the order listed
#binned: numbers cut into ranges [edge_i, edge_i+1) with bin_labels
#log: True = use log(1 + x) when correlating (skewed 2023 dollars & counts)
#To RELABEL or RE-BIN a variable, edit its entry. Columns that appear in the CSV

BLOCKS = {
    "demographics": {"label": "Demographics", "short": "Demographics"},
    "prior_use": {"label": "Prior spending & use (2023)", "short": "Prior use"},
    "conditions": {"label": "Diagnosed conditions", "short": "Conditions"},
    "health_status": {"label": "Health status", "short": "Health status"},
    "engagement": {"label": "Healthcare engagement", "short": "Engagement"},
    "behavior": {"label": "Behavior & attitudes", "short": "Behavior"},
    "other": {"label": "Other (auto-detected columns)", "short": "Other"},
}

def binary(label, block, meps, desc):
    return {"label": label, "block": block, "meps": meps, "desc": desc,
            "kind": "binary", "codes": {0: "No", 1: "Yes"}}

def coded(label, block, meps, desc, codes):
    return {"label": label, "block": block, "meps": meps, "desc": desc,
            "kind": "coded", "codes": codes}

def category(label, block, meps, desc, levels):
    return {"label": label, "block": block, "meps": meps, "desc": desc,
            "kind": "category", "levels": levels}

def binned(label, block, meps, desc, edges, bin_labels, log=False):
    return {"label": label, "block": block, "meps": meps, "desc": desc,
            "kind": "binned", "edges": edges, "bin_labels": bin_labels, "log": log}

INF = np.inf
HEALTH_5 = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
AGREE_5 = {"DISAGREE STRONGLY": "Disagree strongly", "DISAGREE SOMEWHAT": "Disagree somewhat",
           "UNCERTAIN": "Uncertain", "AGREE SOMEWHAT": "Agree somewhat",
           "AGREE STRONGLY": "Agree strongly"}

VARIABLES = {
    #A. Demographics
    "age": binned("Age", "demographics", "AGEY1X",
                  "Age as of 12/31/2023 (MEPS top-codes age at 85).",
                  [18, 30, 40, 50, 60, 70, 80, INF],
                  ["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–85"]),
    "sex": category("Sex", "demographics", "SEX", "Sex.",
                    {"FEMALE": "Female", "MALE": "Male"}),
    "race_eth": category("Race / ethnicity", "demographics", "RACETHX",
                         "Race/ethnicity (edited/imputed).",
                         {"HISPANIC": "Hispanic",
                          "NON-HISPANIC WHITE ONLY": "Non-Hispanic White",
                          "NON-HISPANIC BLACK ONLY": "Non-Hispanic Black",
                          "NON-HISPANIC ASIAN ONLY": "Non-Hispanic Asian",
                          "NON-HISPANIC OTHER RACE OR MULTIPLE RACE": "Non-Hispanic Other/Multiple Races"}),
    "marital": category("Marital status", "demographics", "MARRYY1X",
                        "Marital status as of 12/31/2023.",
                        {"MARRIED": "Married", "NEVER MARRIED": "Never Married",
                         "DIVORCED": "Divorced", "SEPARATED": "Separated", "WIDOWED": "Widowed"}),
    "education": category("Highest degree", "demographics", "HIDEG",
                          "Highest degree when the person first entered MEPS.",
                          {"NO DEGREE": "No degree", "GED": "GED",
                           "HIGH SCHOOL DIPLOMA": "High School Diploma",
                           "BACHELOR'S DEGREE": "Bachelor's Degree",
                           "MASTER'S DEGREE": "Master's Degree",
                           "DOCTORATE DEGREE": "Doctorate", "OTHER DEGREE": "Other Degree"}),
    "region": category("Census region", "demographics", "REGIONY1",
                       "Census region as of 12/31/2023.",
                       {"NORTHEAST": "Northeast", "MIDWEST": "Midwest", "SOUTH": "South",
                        "WEST": "West"}),
    "employment": category("Employment (Round 2)", "demographics", "EMPST2H",
                           "Employment status at the Round 2 interview (2023). MEPS merges "
                           "'not employed' with 'employed earlier in the round only'.",
                           {"EMPLOYED AT RD 2 INT DATE": "Employed",
                            "JOB TO RETURN TO AT RD 2 INT DATE": "Job to return to",
                            "NOT EMPL AT INT DATE/NOT EMPL DUR RD 2": "Not Employed"}),
    "poverty_cat": category("Family income (% of poverty line)", "demographics", "POVCATY1",
                            "2023 family income as a percent of the federal poverty line.",
                            {"POOR/NEGATIVE": "Poor (<100%)", "NEAR POOR": "Near Poor (100–124%)",
                             "LOW INCOME": "Low Income (125–199%)",
                             "MIDDLE INCOME": "Middle Income (200–399%)",
                             "HIGH INCOME": "High Income (400%+)"}),
    "poverty_pct": binned("Family income, % of poverty line (continuous)", "demographics",
                          "POVLEVY1", "2023 family income as a percent of the poverty line.",
                          [0, 100, 200, 300, 400, 600, INF],
                          ["<100%", "100–199%", "200–299%", "300–399%", "400–599%", "600%+"]),
    "insurance": category("Insurance coverage (2023)", "demographics", "INSCOVY1",
                          "Any private coverage at any time in 2023 / public only / "
                          "uninsured all year.",
                          {"ANY PRIVATE": "Any Private", "PUBLIC ONLY": "Public Only",
                           "UNINSURED": "Uninsured"}),
    "medicaid": binary("Had Medicaid/CHIP in 2023", "demographics", "MCDEVY1",
                       "Covered by Medicaid or CHIP at any time in 2023."),
    "medicare": binary("Had Medicare in 2023", "demographics", "MCREVY1",
                       "Covered by Medicare at any time in 2023."),
    "uninsured_all_year": binary("Uninsured all of 2023", "demographics", "UNINSY1",
                                 "No health insurance at any time in 2023."),

    #B. Prior spending & use (2023)
    "tot_exp_y1": binned("Total spending, 2023", "prior_use", "TOTEXPY1",
                         "Total 2023 health care expenditures, all payers.",
                         SPEND_EDGES, SPEND_LABELS, log=True),
    "ob_exp_y1": binned("Office-based spending, 2023", "prior_use", "OBVEXPY1",
                        "2023 spending on office-based provider visits.",
                        [0, 1, 500, 2_000, 5_000, INF],
                        ["$0", "$1–499", "$500–1,999", "$2,000–4,999", "$5,000+"], log=True),
    "er_exp_y1": binned("ER spending, 2023", "prior_use", "ERTEXPY1",
                        "2023 emergency room facility + doctor spending.",
                        [0, 1, 1_000, 5_000, INF], ["$0", "$1–999", "$1,000–4,999", "$5,000+"],
                        log=True),
    "ip_exp_y1": binned("Inpatient hospital spending, 2023", "prior_use", "IPTEXPY1",
                        "2023 hospital inpatient facility + doctor spending.",
                        [0, 1, 10_000, 30_000, INF],
                        ["$0", "$1–9,999", "$10,000–29,999", "$30,000+"], log=True),
    "rx_exp_y1": binned("Prescription drug spending, 2023", "prior_use", "RXEXPY1",
                        "2023 spending on prescription medicines.",
                        [0, 1, 100, 1_000, 5_000, INF],
                        ["$0", "$1–99", "$100–999", "$1,000–4,999", "$5,000+"], log=True),
    "hh_exp_y1": binned("Home health spending, 2023", "prior_use", "HHAEXPY1",
                        "2023 home health agency spending.",
                        [0, 1, 5_000, INF], ["$0", "$1–4,999", "$5,000+"], log=True),
    "oop_exp_y1": binned("Out-of-pocket spending, 2023", "prior_use", "TOTSLFY1",
                         "2023 amount paid by self/family.",
                         [0, 1, 250, 1_000, 3_000, INF],
                         ["$0", "$1–249", "$250–999", "$1,000–2,999", "$3,000+"], log=True),
    "ob_visits_y1": binned("Office-based visits, 2023", "prior_use", "OBTOTVY1",
                           "Number of office-based provider visits in 2023.",
                           [0, 1, 3, 6, 12, 25, INF], ["0", "1–2", "3–5", "6–11", "12–24", "25+"],
                           log=True),
    "op_visits_y1": binned("Hospital outpatient visits, 2023", "prior_use", "OPTOTVY1",
                           "Number of hospital outpatient department visits in 2023.",
                           [0, 1, 2, 5, INF], ["0", "1", "2–4", "5+"], log=True),
    "er_visits_y1": binned("ER visits, 2023", "prior_use", "ERTOTY1",
                           "Number of emergency room visits in 2023.",
                           [0, 1, 2, INF], ["0", "1", "2+"], log=True),
    "ip_discharges_y1": binned("Hospital stays, 2023", "prior_use", "IPDISY1",
                               "Number of hospital discharges in 2023.",
                               [0, 1, 2, INF], ["0", "1", "2+"], log=True),
    "ip_nights_y1": binned("Nights in hospital, 2023", "prior_use", "IPNGTDY1",
                           "Nights in the hospital for 2023 discharges.",
                           [0, 1, 4, 10, INF], ["0", "1–3", "4–9", "10+"], log=True),
    "rx_fills_y1": binned("Prescription fills, 2023", "prior_use", "RXTOTY1",
                          "Prescribed medicines including refills, 2023.",
                          [0, 1, 5, 12, 24, 48, INF], ["0", "1–4", "5–11", "12–23", "24–47", "48+"],
                          log=True),
    "hh_days_y1": binned("Home health days, 2023", "prior_use", "HHTOTDY1",
                         "Days with home health provider care in 2023.",
                         [0, 1, 30, INF], ["0", "1–29", "30+"], log=True),

    #C. Diagnosed conditions
    "dx_hibp": binary("Ever diagnosed: high blood pressure", "conditions", "HIBPDXY1",
                      "Ever told they have high blood pressure (not pregnancy-only), reported in 2023."),
    "dx_chd": binary("Ever diagnosed: coronary heart disease", "conditions", "CHDDXY1",
                     "Ever told they have coronary heart disease, reported in 2023."),
    "dx_angina": binary("Ever diagnosed: angina", "conditions", "ANGIDXY1",
                        "Ever told they have angina, reported in 2023."),
    "dx_mi": binary("Ever diagnosed: heart attack", "conditions", "MIDXY1",
                    "Ever told they had a heart attack (MI), reported in 2023."),
    "dx_oth_heart": binary("Ever diagnosed: other heart disease", "conditions", "OHRTDXY1",
                           "Ever told they have another kind of heart disease, reported in 2023."),
    "dx_stroke": binary("Ever diagnosed: stroke", "conditions", "STRKDXY1",
                        "Ever told they had a stroke or TIA (mini-stroke), reported in 2023."),
    "dx_emphysema": binary("Ever diagnosed: emphysema", "conditions", "EMPHDXY1",
                           "Ever told they have emphysema, reported in 2023."),
    "dx_bronchitis": binary("Chronic bronchitis (last 12 months)", "conditions", "CHBRON1",
                            "Had chronic bronchitis in the last 12 months (Round 1, 2023)."),
    "dx_chol": binary("Ever diagnosed: high cholesterol", "conditions", "CHOLDXY1",
                      "Ever told they have high cholesterol, reported in 2023."),
    "dx_cancer": binary("Ever diagnosed: cancer", "conditions", "CANCERY1",
                        "Ever told they have cancer or a malignancy of any kind, reported in 2023."),
    "dx_diabetes": binary("Ever diagnosed: diabetes", "conditions", "DIABDXY1_M18",
                          "Ever told they have diabetes (not gestational), reported in 2023."),
    "dx_arthritis": binary("Ever diagnosed: arthritis", "conditions", "ARTHDXY1",
                           "Ever told they have arthritis, reported in 2023."),
    "dx_asthma": binary("Ever diagnosed: asthma", "conditions", "ASTHDXY1",
                        "Ever told they have asthma, reported in 2023."),
    "n_chronic": binned("Number of self-reported chronic conditions", "conditions", "(derived)",
                        "How many of the 13 self-reported diagnoses above are 'yes' ",
                        [0, 1, 2, 3, 4, 5, INF], ["0", "1", "2", "3", "4", "5+"]),
    "n_conditions": binned("Number of conditions in the 2023 conditions file", "conditions",
                           "(derived, HC-249)",
                           "Distinct conditions recorded for the person in the 2023 Medical "
                           "Conditions file. Mostly recorded when a condition comes up at a "
                           "medical visit, so it partly reflects 2023 care use.",
                           [0, 1, 3, 6, 10, INF], ["0", "1–2", "3–5", "6–9", "10+"]),

    #D. Health status
    "self_rated_health": coded("Self-rated health", "health_status", "RTHLTH2",
                               "Perceived health status (Round 2, 2023), 1 = excellent ... 5 = poor.",
                               HEALTH_5),
    "self_rated_mental": coded("Self-rated mental health", "health_status", "MNHLTH2",
                               "Perceived mental health status (Round 2, 2023), 1 = excellent ... "
                               "5 = poor.", HEALTH_5),
    "any_limitation": binary("Any limitation", "health_status", "ANYLMIY1",
                             "Any ADL/IADL help, walking, activity, vision or hearing limitation "
                             "(Rounds 1-2, 2023)."),
    "adl_help": binary("Needs help with daily activities (ADL)", "health_status", "ADLHLP1",
                       "Receives help or supervision with activities of daily living (Round 1)."),
    "iadl_help": binary("Needs help with errands/chores (IADL)", "health_status", "IADLHP1",
                        "Receives help with using the phone, paying bills, medications, meals, "
                        "laundry or shopping (Round 1)."),
    "walk_limit": binary("Physical limitation", "health_status", "WLKLIM1",
                         "Difficulty walking, climbing stairs, grasping, reaching, lifting, "
                         "bending or standing (Round 1)."),
    "cog_limit": binary("Cognitive limitation", "health_status", "COGLIM1",
                        "Confusion/memory loss, trouble making decisions, or needs supervision "
                        "for safety (Round 1)."),
    "social_limit": binary("Social limitation", "health_status", "SOCLIM1",
                           "Limited in social or recreational activities (Round 1)."),
    "physical_score": binned("Physical health score (VR-12 PCS)", "health_status", "VPCS2",
                             "VR-12 physical component summary from the self-administered "
                             "questionnaire (SAQ); SF-12v2 metric, 50 = U.S. average, higher = "
                             "better.",
                             [0, 30, 40, 50, 60, INF], ["<30", "30–39", "40–49", "50–59", "60+"]),
    "mental_score": binned("Mental health score (VR-12 MCS)", "health_status", "VMCS2",
                           "VR-12 mental component summary (SAQ); SF-12v2 metric, 50 = U.S. "
                           "average, higher = better.",
                           [0, 30, 40, 50, 60, INF], ["<30", "30–39", "40–49", "50–59", "60+"]),
    "k6_distress": binned("Psychological distress (K6)", "health_status", "K6SUM2",
                          "Kessler K6 score for the past 30 days (SAQ), 0-24; 13+ = serious "
                          "psychological distress.",
                          [0, 5, 13, INF], ["0–4 (low)", "5–12 (moderate)", "13–24 (serious)"]),
    "phq2_depress": binned("Depression screen (PHQ-2)", "health_status", "PHQ22",
                           "PHQ-2 score for the past 2 weeks (SAQ), 0-6; 3+ = positive screen.",
                           [0, 1, 3, INF], ["0", "1–2", "3–6 (positive screen)"]),

    #E. Healthcare engagement
    "has_usual_source": binary("Has a usual source of care", "engagement", "HAVEUS2",
                               "Has a particular doctor's office, clinic or other place they "
                               "usually go when sick (Round 2)."),
    "routine_checkup": binary("Made a routine check-up appointment", "engagement", "ADRTCR2",
                              "Made an appointment for a check-up or routine care in the last "
                              "12 months (SAQ)."),
    "office_visits_saq": coded("Doctor's office visits, last 12 months (SAQ)", "engagement",
                               "ADAPPT2",
                               "Visits to a doctor's office or clinic in the last 12 months, "
                               "not counting ER visits (SAQ).",
                               {0: "None", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5–9", 6: "10+"}),
    "needed_specialist": binary("Made a specialist appointment", "engagement", "ADSPCL2",
                                "Made any appointment to see a specialist in the last 12 months "
                                "(SAQ). MEPS labels this 'needed to see specialist'."),
    "care_rating": binned("Rating of all health care (0–10)", "engagement", "ADHECR2",
                          "Rating of all health care in the last 12 months, 0 = worst ... "
                          "10 = best (SAQ). Asked only of adults with 1+ office visits, so "
                          "'Missing' includes people with no visits.",
                          [0, 7, 9, INF], ["0–6", "7–8", "9–10"]),
    "delayed_care_cost": binary("Delayed medical care due to cost", "engagement", "DLAYCA2",
                                "Delayed medical care because of worry about cost (past 12 months)."),
    "unafford_care": binary("Could not afford needed care", "engagement", "AFRDCA2",
                            "Needed medical care but could not afford it (past 12 months)."),
    "delayed_rx_cost": binary("Delayed prescriptions due to cost", "engagement", "DLAYPM2",
                              "Delayed getting prescription medicines because of cost "
                              "(past 12 months)."),
    "unafford_rx": binary("Could not afford prescriptions", "engagement", "AFRDPM2",
                          "Needed prescription medicines but could not afford them "
                          "(past 12 months)."),
    "covid_vax_ever": binary("Ever had a COVID-19 vaccine", "engagement", "COVAXEVR1",
                             "Ever received a COVID-19 vaccine (Round 1, 2023)."),

    #F. Behavior & attitudes
    "smoker": binary("Currently smokes", "behavior", "ADSMOK2", "Currently smokes (SAQ)."),
    "no_need_insurance": category("\"Healthy enough to not need insurance\"", "behavior",
                                  "ADINSA2",
                                  "Agreement with: \"I'm healthy enough that I really don't need "
                                  "health insurance.\" (SAQ)", AGREE_5),
    "risk_taker": category("\"Can overcome illness without medical help\"", "behavior",
                           "ADOVER2",
                           "Agreement with: \"I can overcome illness without help from a "
                           "medically trained person.\" (SAQ) Stored in our CSV as `risk_taker`.",
                           AGREE_5),
}

#Conditions-file columns (built from the 2023 Medical Conditions file)
#Body-system counts: n_cond_<system> = number of conditions recorded in that system.
BODY_SYSTEMS = {
    "blood": "Blood & Immune", "circulatory": "Circulatory", "congenital": "Congenital",
    "dental": "Dental", "digestive": "Digestive", "ear": "Ear",
    "endocrine": "Endocrine & Metabolic", "external_cause": "External Causes", "eye": "Eye",
    "genitourinary": "Genitourinary", "health_factors": "Health-Status Factors",
    "infectious": "Infectious", "injury": "Injury & Infections",
    "mental_behavioral": "Mental & Behavioral", "musculoskeletal": "Musculoskeletal",
    "neoplasm": "Neoplasms", "nervous": "Nervous System",
    "perinatal": "Perinatal", "pregnancy": "Pregnancy-Related", "respiratory": "Respiratory",
    "skin": "Skin", "symptoms": "Symptoms & Abnormal Findings",
}
for _key, _name in BODY_SYSTEMS.items():
    VARIABLES[f"n_cond_{_key}"] = binned(
        f"Conditions recorded: {_name.lower()}", "conditions", "(derived, HC-249)",
        f"Number of 2023 conditions-file records in the {_name.lower()} body system.",
        [0, 1, 2, INF], ["0", "1", "2+"])

#CCSR category flags (1 = person has a 2023 condition in this category)
CCSR_DESCRIPTIONS = {
    "CIR000": "Rare circulatory condition (masked by MEPS)",
    "CIR007": "Essential hypertension",
    "CIR011": "Coronary atherosclerosis and other heart disease",
    "CIR017": "Cardiac dysrhythmias",
    "DEN001": "Any dental condition including traumatic injury",
    "DEN002": "Nontraumatic dental conditions",
    "DIG004": "Esophageal disorders",
    "DIG025": "Other specified and unspecified gastrointestinal disorders",
    "EAR001": "Otitis media",
    "END001": "Thyroid disorders",
    "END002": "Diabetes mellitus without complication",
    "END005": "Diabetes mellitus, Type 2",
    "END007": "Nutritional deficiencies",
    "END010": "Disorders of lipid metabolism",
    "EYE002": "Cataract and other lens disorders",
    "FAC000": "Rare health-status/contact factor (masked by MEPS)",
    "FAC012": "Other specified encounters and counseling",
    "FAC014": "Medical examination/evaluation",
    "FAC025": "Other specified status",
    "GEN004": "Urinary tract infections",
    "INF003": "Bacterial infections",
    "INF008": "Viral infection",
    "INF012": "COVID-19",
    "INJ000": "Rare injury or poisoning (masked by MEPS)",
    "INJ031": "Allergic reactions",
    "INJ064": "Other unspecified injuries, subsequent encounter",
    "MBD002": "Depressive disorders",
    "MBD005": "Anxiety and fear-related disorders",
    "MBD007": "Trauma- and stressor-related disorders",
    "MBD014": "Neurodevelopmental disorders",
    "MUS000": "Rare musculoskeletal condition (masked by MEPS)",
    "MUS006": "Osteoarthritis",
    "MUS010": "Musculoskeletal pain, not low back pain",
    "MUS011": "Spondylopathies/spondyloarthropathy (including infective)",
    "MUS025": "Other specified connective tissue disease",
    "MUS038": "Low back pain",
    "NVS010": "Headache; including migraine",
    "NVS016": "Sleep wake disorders",
    "RSP001": "Sinusitis",
    "RSP006": "Other specified upper respiratory infections",
    "RSP007": "Other specified and unspecified upper respiratory disease",
    "RSP009": "Asthma",
    "SKN007": "Other specified and unspecified skin disorders",
    "SYM006": "Abdominal pain and other digestive/abdomen signs and symptoms",
    "SYM010": "Nervous system signs and symptoms",
    "SYM013": "Respiratory signs and symptoms",
    "SYM016": "Other general signs and symptoms",
    "SYM017": "Abnormal findings without diagnosis",
}
for _code, _desc in CCSR_DESCRIPTIONS.items():
    VARIABLES[f"ccsr_{_code}"] = binary(
        f"{_desc} ({_code})", "conditions", f"CCSR {_code} (HC-249)",
        f"Has a 2023 conditions-file record in CCSR category {_code}: {_desc}.")

#[2] Data loading
#Turn one column into an ordered categorical of display groups
def make_groups(values: pd.Series, spec: dict) -> pd.Series: 
    kind = spec["kind"]
    if kind == "binned":
        numbers = pd.to_numeric(values, errors="coerce")
        labels = pd.cut(numbers, bins=spec["edges"], labels=spec["bin_labels"], right=False)
        mapped, order = labels.astype(object), list(spec["bin_labels"])
    elif kind in ("binary", "coded"):
        mapped = pd.to_numeric(values, errors="coerce").map(spec["codes"])
        order = list(spec["codes"].values())
    elif kind == "category":
        mapped = values.map(spec["levels"])
        order = list(spec["levels"].values())
    else:
        raise ValueError(f"Unknown kind {kind!r} for {spec['label']}")
    grouped = pd.Categorical(mapped, categories=order + [MISSING])
    return pd.Series(grouped, index=values.index).fillna(MISSING)

def infer_spec(col: str, values: pd.Series) -> dict: #Registry entry for a column that is not in VARIABLES 
    desc = "Auto-detected column (add it to VARIABLES in streamlit_app.py to label it)."
    if pd.api.types.is_numeric_dtype(values):
        distinct = np.sort(values.dropna().unique())
        if len(distinct) and set(distinct) <= {0, 1}:
            return binary(col, "other", "", desc)
        if len(distinct) <= 8:
            return coded(col, "other", "", desc, {v: f"{v:g}" for v in distinct})
        edges = np.unique(np.nanquantile(values, [0, 0.2, 0.4, 0.6, 0.8]))
        edges = list(edges) + [INF]
        labels = [f"{lo:,.4g}+" if hi == INF else f"{lo:,.4g} to <{hi:,.4g}"
                  for lo, hi in zip(edges[:-1], edges[1:])]
        return binned(col, "other", "", desc + " Grouped into quintiles.", edges, labels)
    top_levels = values.dropna().astype(str).value_counts().index[:12]
    return category(col, "other", "", desc, {v: v.title() for v in top_levels})


def detail_band_labels(cutoff: float) -> list[str]: #Five 2024 spending bands; the last one is exactly the high-cost (top 20%) group
    return ["$0", "$1–999", "$1,000–4,999", f"$5,000–{cutoff - 1:,.0f}",
            f"Top 20% (${cutoff:,.0f}+)"]

@st.cache_data(show_spinner="Loading MEPS data...")
def load_data(path_str: str, file_modified: float) -> dict:
  #Read the CSV and pre-compute everything that does not depend on the filters.
    df = pd.read_csv(path_str)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])  # saved pandas index
    if OUTCOME_LOG not in df:
        df[OUTCOME_LOG] = np.log1p(df[OUTCOME])
    if HIGH_COST not in df:
        df[HIGH_COST] = (df[OUTCOME] >= df[OUTCOME].quantile(0.8)).astype(int)

    #Registry= hand-written entries that exist in this CSV + auto-detected extras.
    registry = {col: spec for col, spec in VARIABLES.items() if col in df.columns}
    for col in df.columns:
        if col not in registry and col not in NOT_PREDICTORS and "_y2" not in col:
            registry[col] = infer_spec(col, df[col])
    absent = [col for col in VARIABLES if col not in df.columns]

    groups = pd.DataFrame({col: make_groups(df[col], spec) for col, spec in registry.items()},
                          index=df.index)

    cutoff = float(df.loc[df[HIGH_COST] == 1, OUTCOME].min())  #smallest top-20% bill
    bands = pd.DataFrame({
        "band_2024": pd.cut(df[OUTCOME], SPEND_EDGES, right=False, labels=SPEND_LABELS_SHORT),
        "band_2023": pd.cut(df[PRIOR_SPEND], SPEND_EDGES, right=False, labels=SPEND_LABELS_SHORT),
        "band_detail": pd.cut(df[OUTCOME], [0, 1, 1_000, 5_000, cutoff, INF], right=False,
                              labels=detail_band_labels(cutoff)),
    }, index=df.index)

    #Inputs for the correlation ranking (tab [8]): one numeric column per variable
    #(log1p for skewed 2023 dollars/counts) and one 0/1 column per category level.
    assoc_cols, assoc_meta = {}, []
    for col, spec in registry.items():
        if spec["kind"] == "category":
            known = groups[col] != MISSING
            for level in groups[col].cat.categories:
                if level == MISSING:
                    continue
                item = f"{col}={level}"
                assoc_cols[item] = (groups[col] == level).astype(float).where(known)
                assoc_meta.append({"item": item, "variable": col, "block": spec["block"],
                                   "label": f"{spec['label']}: {level}"})
        else:
            x = pd.to_numeric(df[col], errors="coerce").astype(float)
            if spec.get("log"):
                x = np.log1p(x.clip(lower=0))
            assoc_cols[col] = x
            assoc_meta.append({"item": col, "variable": col, "block": spec["block"],
                               "label": spec["label"]})
    assoc_x = pd.DataFrame(assoc_cols, index=df.index)
    assoc_meta = pd.DataFrame(assoc_meta).set_index("item")
    assoc_meta["is_binary"] = [bool(set(assoc_x[c].dropna().unique()) <= {0.0, 1.0})
                               for c in assoc_x.columns]
    assert list(assoc_meta.index) == list(assoc_x.columns)  # rows of meta = columns of x

    dictionary = pd.DataFrame([{
        "Column": col,
        "Label": spec["label"],
        "Block": BLOCKS[spec["block"]]["label"],
        "MEPS Source": spec.get("meps", ""),
        "Groups used in charts": ", ".join(c for c in groups[col].cat.categories if c != MISSING),
        "% missing": round(float((groups[col] == MISSING).mean() * 100), 1),
        "Description": spec["desc"],
    } for col, spec in registry.items()])

    return {
        "df": df, "groups": groups, "bands": bands, "registry": registry, "absent": absent,
        "cutoff": cutoff, "assoc_x": assoc_x, "assoc_meta": assoc_meta, "dictionary": dictionary,
    }

#[3] statistics measured

def stat_value(y: np.ndarray, high: np.ndarray, stat: str) -> float: #The selected spending measure for one set of adults.
    if len(y) == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(y))
    if stat == "median":
        return float(np.median(y))
    if stat == "pct_high":
        return float(np.mean(high) * 100)
    if stat == "pct_zero":
        return float(np.mean(y == 0) * 100)
    raise ValueError(stat)

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half

def stat_ci(y: np.ndarray, high: np.ndarray, stat: str, rng: np.random.Generator):
    n = len(y)
    if n < 2:
        return np.nan, np.nan
    if stat in ("pct_high", "pct_zero"):
        hits = int(np.sum(high)) if stat == "pct_high" else int(np.sum(y == 0))
        lo, hi = wilson_interval(hits, n)
        return lo * 100, hi * 100
    samples = y[rng.integers(0, n, size=(N_BOOT, n))]
    estimates = samples.mean(axis=1) if stat == "mean" else np.median(samples, axis=1)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)

def group_summary(sel: pd.DataFrame, grp: pd.Series, stat: str, min_n: int,
                  show_missing: bool) -> tuple[pd.DataFrame, float]:
#One row per group of `grp`: size, spending summaries, the chosen statistic + 95% CI.
#flag compares each group's CI with the value for ALL adults in the selection:
#HIGHER / LOWER when the whole CI is above / below it, otherwise SIMILAR.
#MODEL HOOK (explorer) compares actual with predicted spending by group,
#add `pred_mean=sel["pred_tot_exp_y2"][m].mean()` to each row below.
    y = sel[OUTCOME].to_numpy(dtype=float)
    y23 = sel[PRIOR_SPEND].to_numpy(dtype=float)
    high = sel[HIGH_COST].to_numpy(dtype=float)
    overall = stat_value(y, high, stat)
    codes = grp.cat.codes.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i, name in enumerate(grp.cat.categories):
        if name == MISSING and not show_missing:
            continue
        m = codes == i
        n = int(m.sum())
        if n == 0:
            continue
        yg, hg = y[m], high[m]
        small = n < min_n
        lo, hi = (np.nan, np.nan) if small else stat_ci(yg, hg, stat, rng)
        flag = TOO_FEW if small else HIGHER if lo > overall else LOWER if hi < overall else SIMILAR
        rows.append({
            "group": name, "n": n, "pct_sel": n / len(y) * 100,
            "mean": yg.mean(), "median": float(np.median(yg)),
            "pct_high": hg.mean() * 100, "pct_zero": float(np.mean(yg == 0) * 100),
            "mean23": float(np.nanmean(y23[m])),
            "value": stat_value(yg, hg, stat), "lo": lo, "hi": hi, "flag": flag, "small": small,
        })
    return pd.DataFrame(rows), overall

def two_way_summary(sel: pd.DataFrame, g_rows: pd.Series, g_cols: pd.Series, stat: str,
                    show_missing: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    #The chosen statistic and the number of adults for every (row group, column group).
    d = pd.DataFrame({"r": g_rows, "c": g_cols, "y": sel[OUTCOME].astype(float),
                      "h": sel[HIGH_COST].astype(float)})
    if not show_missing:
        d = d[(d["r"] != MISSING) & (d["c"] != MISSING)]
    d["zero"] = (d["y"] == 0).astype(float)
    grouped = d.groupby(["r", "c"], observed=True)
    value = {"mean": grouped["y"].mean(), "median": grouped["y"].median(),
             "pct_high": grouped["h"].mean() * 100, "pct_zero": grouped["zero"].mean() * 100}[stat]
    counts = grouped.size()
    rows = [c for c in g_rows.cat.categories if c in set(d["r"])]
    cols = [c for c in g_cols.cat.categories if c in set(d["c"])]
    val = value.unstack("c").reindex(index=rows, columns=cols)
    cnt = counts.unstack("c").reindex(index=rows, columns=cols).fillna(0).astype(int)
    return val, cnt

def condition_summary(sel: pd.DataFrame, items: list[tuple], stat: str,
                      min_n: int) -> pd.DataFrame:
    y = sel[OUTCOME].to_numpy(dtype=float)
    high = sel[HIGH_COST].to_numpy(dtype=float)
    rows = []
    for col, label, x in items:
        has, has_not = x == 1, x == 0
        n_with, n_without = int(has.sum()), int(has_not.sum())
        if n_with == 0 or n_without == 0:
            continue
        v_with = stat_value(y[has], high[has], stat)
        v_without = stat_value(y[has_not], high[has_not], stat)
        rows.append({
            "column": col, "label": label, "n_with": n_with,
            "prevalence": n_with / (n_with + n_without) * 100,
            "value": v_with, "value_without": v_without,
            "ratio": v_with / v_without if v_without > 0 else np.nan,
            "small": n_with < min_n,
        })
    return pd.DataFrame(rows)

def association_table(assoc_x: pd.DataFrame, assoc_meta: pd.DataFrame, log_y: pd.Series,
                      min_n: int) -> pd.DataFrame:
#Pearson r between every predictor (or category level) and log(1 + 2024 spending).
#2023 dollars/counts are log1p-transformed, categories become 0/1 indicators
#each r uses the adults with a non-missing value 
#A 0/1 item needs at least `min_n` adults on each side.
#MODEL HOOK (associations) shows model-based importance 
    x = assoc_x.to_numpy(dtype=float)
    y = log_y.to_numpy(dtype=float)
    valid = ~np.isnan(x)
    v = valid.astype(float)
    xz = np.where(valid, x, 0.0)
    n = v.sum(axis=0)
    sum_y, sum_yy = v.T @ y, v.T @ (y * y)
    sum_x, sum_xx, sum_xy = xz.sum(axis=0), (xz * xz).sum(axis=0), xz.T @ y
    with np.errstate(invalid="ignore", divide="ignore"):
        var_x = sum_xx - sum_x**2 / n
        var_y = sum_yy - sum_y**2 / n
        r = (sum_xy - sum_x * sum_y / n) / np.sqrt(var_x * var_y)
    is_01 = assoc_meta["is_binary"].to_numpy()
    enough = (n >= min_n) & (~is_01 | ((sum_x >= min_n) & (n - sum_x >= min_n)))
    usable = enough & (var_x > 1e-9 * np.maximum(sum_xx, 1.0)) & (var_y > 0) & np.isfinite(r)
    out = assoc_meta.loc[usable].copy()
    out["r"], out["n"] = r[usable], n[usable].astype(int)
    out["abs_r"] = out["r"].abs()
    return out.sort_values("abs_r", ascending=False)

#[4]Charts

def current_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"

C = COLORS[current_theme()]  #active palette
st.session_state["drawn_theme"] = current_theme()  #palette this run's charts are drawn with

#st.context.theme can report "light" on a session's first run and right after a viewer changes
#the theme in Settings, until the browser sends its next rerun request. Without this check, dark
#mode could show light-palette charts and tables (white backgrounds). This empty fragment reruns
#every 2s (the browser triggers it, so it reads the true theme) and redraws the page on a mismatch.
@st.fragment(run_every="2s")
def redraw_on_theme_change() -> None:
    if current_theme() != st.session_state.get("drawn_theme"):
        st.rerun()

redraw_on_theme_change()

def fmt_dollars(x: float) -> str:
    return "–" if pd.isna(x) else f"${x:,.0f}"

def fmt_pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1f}%"

def fmt_stat(x: float, stat: str) -> str:
    return fmt_dollars(x) if STATS[stat]["unit"] == "$" else fmt_pct(x)

def hover_value(stat: str, field: str) -> str: #Plotly hover snippet for one number
    if STATS[stat]["unit"] == "$":
        return "$%{" + field + ":,.0f}"
    return "%{" + field + ":.1f}%"

def signed_dollars(x: float) -> str:
    return ("+" if x >= 0 else "-") + f"${abs(x):,.0f}"

def delta_args(diff: float, text: str, tolerance: float, inverse: bool = True) -> dict:
    if pd.isna(diff) or abs(diff) < tolerance:
        return {"delta": text, "delta_color": "off", "delta_arrow": "off"}
    return {"delta": text, "delta_color": "inverse" if inverse else "off"}

def axis_format(stat: str) -> dict:
    if STATS[stat]["unit"] == "$":
        return {"tickprefix": "$", "tickformat": "~s"}  
    return {"ticksuffix": "%", "tickformat": ".0f"}

def base_layout(title: str, subtitle: str = "", height: int | None = None, #Put chart-specific axis settings in `xaxis` / `yaxis`
                legend: bool = False, grid: bool = True, xaxis: dict | None = None,
                yaxis: dict | None = None, **extra) -> dict:
    
    def axis(custom: dict | None) -> dict:
        merged = {"showgrid": grid, "gridcolor": C["grid"], "linecolor": C["grid"],
                  "zerolinecolor": C["grid"], "ticks": "", "automargin": True,
                  "title": {"standoff": 12}}
        custom = dict(custom or {})
        if "title" in custom:  #keep the standoff when a chart adds an axis title
            merged["title"] = {**merged["title"], **custom.pop("title")}
        merged.update(custom)
        return merged

    title_spec = {"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top",
                  "yref": "container", "font": {"size": 15}}
    top = 40
    if subtitle:
        title_spec["subtitle"] = {"text": subtitle, "font": {"size": 12}}
        top += 20
    if legend:
        top += 35
    layout = {
        "font": {"family": FONT_FAMILY, "color": C["font"], "size": 12},
        "paper_bgcolor": C["paper"], "plot_bgcolor": C["plot"],
        "title": title_spec,
        "margin": {"l": 0, "r": 25, "b": 10, "t": top},  #r=25 keeps a last "100%" tick visible
        "showlegend": legend,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left",
                   "x": 0, "font": {"size": 11}, "traceorder": "normal"},
        "hoverlabel": {"font": {"size": 12}, "align": "left"},
        "xaxis": axis(xaxis), "yaxis": axis(yaxis),
    }
    if height:
        layout["height"] = height
    layout.update(extra)
    return layout

def new_fig(title: str, **layout_kwargs) -> go.Figure:
    return go.Figure(layout=base_layout(title, **layout_kwargs))

PLOTLY_CONFIG = {"displaylogo": False}  #toolbar options shown when hovering a chart

def show_chart(where, fig: go.Figure, key: str) -> None: #Draw a figure in a column/container. theme=None keeps the og style
    where.plotly_chart(fig, theme=None, key=key, config=PLOTLY_CONFIG)

def empty_fig(message: str, height: int = 300) -> go.Figure:
    fig = new_fig("", height=height, xaxis={"visible": False}, yaxis={"visible": False})
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14})
    return fig

#[7] "at a glance" charts 

def fig_spending_bands(band_2024: pd.Series, y: pd.Series) -> go.Figure:
    """Share of ADULTS vs. share of DOLLARS in each 2024 spending band."""
    counts = band_2024.value_counts().reindex(SPEND_LABELS_SHORT, fill_value=0)
    dollars = y.groupby(band_2024, observed=False).sum().reindex(SPEND_LABELS_SHORT, fill_value=0)
    pct_adults = counts / max(counts.sum(), 1) * 100
    pct_dollars = dollars / dollars.sum() * 100 if dollars.sum() > 0 else dollars * 0.0
    fig = new_fig("Adults vs. Dollars by 2024 Spending Band", height=380, legend=True,
                  barmode="group", bargap=0.3, bargroupgap=0.06,
                  xaxis={"type": "category", "title": {"text": "2024 Total Spending"}},
                  yaxis={"ticksuffix": "%", "rangemode": "tozero"})
    fig.add_bar(x=SPEND_LABELS_SHORT, y=pct_adults, name="Share of Adults",
                marker_color=C["secondary"], customdata=counts.to_numpy(),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% of adults "
                              "(%{customdata:,} people)<extra></extra>")
    fig.add_bar(x=SPEND_LABELS_SHORT, y=pct_dollars, name="Share of All 2024 Dollars",
                marker_color=C["primary"], customdata=dollars.to_numpy(),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% of all 2024 spending "
                              "($%{customdata:,.0f})<extra></extra>")
    return fig


def fig_concentration(y: pd.Series) -> go.Figure:
    """Cumulative share of spending held by the top X% of spenders."""
    values = np.sort(y.to_numpy(dtype=float))[::-1]
    n, total = len(values), values.sum()
    if n == 0 or total <= 0:
        return empty_fig("No 2024 spending in this selection", height=380)
    cum_share = np.cumsum(values) / total * 100
    people = np.arange(1, n + 1) / n * 100
    keep = np.unique(np.linspace(0, n - 1, min(n, 300)).round().astype(int))  # light figure
    fig = new_fig("How Concentrated is 2024 Spending?", height=380, legend=True,
                  xaxis={"ticksuffix": "%", "range": [0, 100], "dtick": 25,
                         "title": {"text": "Adults, Ranked From Highest to Lowest Spender"}},
                  yaxis={"ticksuffix": "%", "range": [0, 103],
                         "title": {"text": "Share of all 2024 Spending"}})
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="If Everyone Spent the Same",
                    line={"color": C["neutral"], "width": 1.5, "dash": "dot"}, hoverinfo="skip")
    fig.add_scatter(x=np.r_[0, people[keep]], y=np.r_[0, cum_share[keep]], mode="lines",
                    name="Actual 2024 Spending", line={"color": C["primary"], "width": 2.5},
                    hovertemplate="Top %{x:.1f}% of adults hold "
                                  "%{y:.1f}% of spending<extra></extra>")
    # Direct labels on three points only (top 1%, 10%, 50%), as in the notebook EDA.
    marks = [(p, cum_share[max(1, n * p // 100) - 1]) for p in (1, 10, 50)]
    fig.add_scatter(x=[p for p, _ in marks], y=[s for _, s in marks], mode="markers+text",
                    text=[f"Top {p}% → {s:.0f}%" for p, s in marks], textposition="middle right",
                    marker={"size": 9, "color": C["primary"],
                            "line": {"color": C["surface"], "width": 2}},
                    showlegend=False, hoverinfo="skip")
    return fig


def fig_prior_vs_next(sel: pd.DataFrame, band_2023: pd.Series, min_n: int) -> go.Figure:
    """Mean 2024 Spending (bars) vs. Mean 2023 Spending (line) for each 2023 Band.
    """
    d = pd.DataFrame({"band": band_2023, "y24": sel[OUTCOME], "y23": sel[PRIOR_SPEND]})
    g = d.groupby("band", observed=True).agg(n=("y24", "size"), mean24=("y24", "mean"),
                                             mean23=("y23", "mean"))
    g = g[g["n"] >= min_n]
    if g.empty:
        return empty_fig(f"No 2023 spending band has {min_n}+ adults", height=380)
    x = g.index.astype(str).tolist()
    fig = new_fig("Does Last Year Predict Next Year?", height=380, legend=True, bargap=0.35,
                  xaxis={"type": "category", "title": {"text": "Total Spending in 2023"}},
                  yaxis={"tickprefix": "$", "tickformat": "~s", "rangemode": "tozero"})
    fig.add_bar(x=x, y=g["mean24"], name="Mean 2024 Spending", marker_color=C["primary"],
                customdata=np.c_[g["n"], g["mean23"]],
                hovertemplate="<b>2023 Spending %{x}</b><br>Mean 2024: $%{y:,.0f}"
                              "<br>Mean 2023: $%{customdata[1]:,.0f}"
                              "<br>Adults: %{customdata[0]:,.0f}<extra></extra>")
    fig.add_scatter(x=x, y=g["mean23"], mode="lines+markers",
                    name="Mean 2023 Spending (\"next year = last year\")",
                    line={"color": C["ink"], "width": 2},
                    marker={"size": 8, "color": C["ink"], "line": {"color": C["surface"], "width": 2}},
                    hoverinfo="skip")
    return fig

#[8] variable explorer

def fig_group_stat(summary: pd.DataFrame, stat: str, overall: float, show_ci: bool,
                   min_n: int) -> go.Figure:
#Horizontal bars of the chosen statistic per group, with 95% CIs and an overall line.
#MODEL HOOK (explorer): to overlay model predictions, add one more trace here
#name="Model-predicted mean" after adding pred_mean in group_summary().
    
    s = summary[~summary["small"]]
    if s.empty:
        return empty_fig(f"No group has at least {min_n} adults")
    order = s["group"].astype(str).tolist()
    subtitle = f"Dashed Line = All Adults in Selection ({fmt_stat(overall, stat)})"
    if show_ci:
        subtitle += "; Whiskers = 95% Confidence Interval"
    fig = new_fig(f"{STATS[stat]['label']} by group", subtitle=subtitle, legend=True,
                  height=max(320, 44 * len(order) + 150), barmode="overlay", bargap=0.3,
                  xaxis={"rangemode": "tozero", **axis_format(stat)},
                  yaxis={"type": "category", "categoryorder": "array", "categoryarray": order,
                         "autorange": "reversed"})
    for flag, color in ((HIGHER, C["higher"]), (SIMILAR, C["neutral"]), (LOWER, C["lower"])):
        d = s[s["flag"] == flag]
        if d.empty:
            continue
        error = None
        if show_ci:
            error = {"type": "data", "symmetric": False,
                     "array": (d["hi"] - d["value"]).clip(lower=0),
                     "arrayminus": (d["value"] - d["lo"]).clip(lower=0),
                     "color": C["ink"], "thickness": 1.2, "width": 5}
        fig.add_bar(
            y=d["group"].astype(str), x=d["value"], orientation="h", name=flag,
            marker_color=color, error_x=error,
            customdata=np.c_[d["n"], d["lo"], d["hi"]],
            hovertemplate="<b>%{y}</b><br>" + STATS[stat]["label"] + ": " + hover_value(stat, "x")
                          + "<br>95% CI: " + hover_value(stat, "customdata[1]") + " to "
                          + hover_value(stat, "customdata[2]")
                          + "<br>Adults: %{customdata[0]:,.0f}<extra></extra>")
    fig.add_vline(x=overall, line={"color": C["ink"], "width": 1.5, "dash": "dash"})
    return fig


def fig_group_composition(grp: pd.Series, band_detail: pd.Series,
                          order: list[str]) -> go.Figure:
#100% stacked bars: how each group splits across five 2024 spending bands
    if not order:
        return empty_fig("No groups to show")
    d = pd.DataFrame({"g": grp.astype(str), "b": band_detail})
    d = d[d["g"].isin(order)]
    labels = list(band_detail.cat.categories)
    counts = (d.groupby(["g", "b"], observed=True).size().unstack("b")
              .reindex(index=order, columns=labels).fillna(0))
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    fig = new_fig("Where Each Group Falls in 2024 Spending",
                  subtitle="Share of Each Group in Five 2024 Spending Bands",
                  legend=True, height=max(320, 44 * len(order) + 150), barmode="stack",
                  bargap=0.3, xaxis={"range": [0, 100], "ticksuffix": "%"},
                  yaxis={"type": "category", "categoryorder": "array", "categoryarray": order,
                         "autorange": "reversed"})
    for i, band in enumerate(labels):
        fig.add_bar(y=order, x=pct[band], orientation="h", name=band,
                    marker={"color": C["band_ramp"][i], "line": {"color": C["surface"], "width": 1}},
                    customdata=counts[band].to_numpy(),
                    hovertemplate="<b>%{y}</b><br>" + band
                                  + ": %{x:.1f}% (%{customdata:,.0f} adults)<extra></extra>")
    return fig

def table_group_summary(summary: pd.DataFrame, stat: str, title: str) -> go.Figure:
    #Styled summary table, like the example's go.Table with color-coded cells
    s = summary.reset_index(drop=True)
    change = s["mean"] - s["mean23"]
    header = ["Group", "Adults", "% of Selection", "Mean 2024", "Median 2024",
              "% High-cost", "% Spent $0", "Mean 2023", "Change in Mean"]
    cells = [
        s["group"].astype(str).tolist(),
        [f"{n:,}" for n in s["n"]],
        [fmt_pct(v) for v in s["pct_sel"]],
        [fmt_dollars(v) for v in s["mean"]],
        [fmt_dollars(v) for v in s["median"]],
        [fmt_pct(v) for v in s["pct_high"]],
        [fmt_pct(v) for v in s["pct_zero"]],
        [fmt_dollars(v) for v in s["mean23"]],
        [("+" if v >= 0 else "-") + fmt_dollars(abs(v)) for v in change],
    ]
    #Cell colors: the selected statistic's column is tinted red/blue when the group is clearly above/below overall
    stat_col = {"mean": 3, "median": 4, "pct_high": 5, "pct_zero": 6}[stat]
    tint = {HIGHER: C["tint_higher"], LOWER: C["tint_lower"]}
    fills = [[C["tint_small"] if small else tint.get(flag, C["table_cell"]) if j == stat_col
              else C["table_cell"] for flag, small in zip(s["flag"], s["small"])]
             for j in range(len(header))]
    table = go.Table(
        columnorder=list(range(len(header))), columnwidth=[34, 12, 13, 14, 14, 12, 12, 14, 14],
        header={"values": [f"<b>{h}</b>" for h in header], "fill_color": C["table_header"],
                "font": {"color": C["table_header_text"], "size": 12},
                "line_color": "rgba(255,255,255,0.2)", "align": ["left", "center"], "height": 28},
        cells={"values": cells, "fill_color": fills,
               "font": {"color": C["table_text"], "size": 12},
               "line_color": "rgba(255,255,255,0.2)", "align": ["left", "center"], "height": 26})
    layout = {"title": {"text": title, "x": 0, "font": {"color": C["primary"], "size": 15}},
              "font": {"family": FONT_FAMILY}, "paper_bgcolor": C["paper"],
              "margin": {"l": 0, "r": 10, "b": 10, "t": 30}, "height": 26 * len(s) + 95}
    return go.Figure(data=[table], layout=layout)

#[8] strongest associations

def fig_associations(top: pd.DataFrame, highlight: str) -> go.Figure:
#Ranked bars of correlation with log 2024 spending
    if top.empty:
        return empty_fig("No variables to rank in this selection")
    labels = [f"{label}  ·  {BLOCKS[block]['short']}"
              for label, block in zip(top["label"], top["block"])]
    colors = [C["primary"] if highlight in ("", block) else C["muted"] for block in top["block"]]
    limit = max(0.05, float(top["r"].abs().max()) * 1.1)
    fig = new_fig(f"Top {len(top)} associations with 2024 spending",
                  subtitle=f"Highlighted: {BLOCKS[highlight]['label']}" if highlight else "",
                  height=26 * len(top) + 130, bargap=0.3,
                  xaxis={"range": [-limit, limit], "zeroline": False,
                         "title": {"text": "Correlation (r) with log(1 + 2024 spending)"}},
                  yaxis={"type": "category", "categoryorder": "array", "categoryarray": labels,
                         "autorange": "reversed"})
    fig.add_bar(x=top["r"], y=labels, orientation="h", marker_color=colors,
                customdata=np.c_[top["n"]],
                hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<br>Adults with data: "
                              "%{customdata[0]:,.0f}<extra></extra>")
    fig.add_vline(x=0, line={"color": C["ink"], "width": 1})
    return fig

#[8] two variables at once

def fig_heatmap(val: pd.DataFrame, cnt: pd.DataFrame, stat: str, min_n: int,
                row_label: str, col_label: str) -> go.Figure:
    """Heatmap of the chosen statistic for every combination of two variables."""
    if val.empty:
        return empty_fig("No data for this combination")
    enough = cnt.to_numpy() >= min_n
    z = np.where(enough, val.to_numpy(dtype=float), np.nan)
    text = [[fmt_stat(v, stat) if ok else f"n<{min_n}" for v, ok in zip(row_v, row_ok)]
            for row_v, row_ok in zip(val.to_numpy(dtype=float), enough)]
    colorbar = {"title": {"text": STATS[stat]["short"]}}
    colorbar.update({k: v for k, v in axis_format(stat).items() if k != "tickformat"})
    fig = new_fig(f"{STATS[stat]['label']} by {row_label.lower()} and {col_label.lower()}",
                  height=max(320, 52 * len(val) + 170), grid=False,
                  plot_bgcolor=C["paper"],  # blank (too-small) cells read as "no data"
                  xaxis={"type": "category", "title": {"text": col_label}},
                  yaxis={"type": "category", "autorange": "reversed",
                         "title": {"text": row_label}})
    fig.add_trace(go.Heatmap(
        z=z, x=[str(c) for c in val.columns], y=[str(r) for r in val.index], text=text,
        texttemplate="%{text}", textfont={"size": 12}, colorscale=C["heat_scale"],
        xgap=3, ygap=3, customdata=cnt.to_numpy(), colorbar=colorbar,
        hovertemplate=f"<b>{row_label}: %{{y}}<br>{col_label}: %{{x}}</b><br>"
                      f"{STATS[stat]['label']}: %{{text}}<br>Adults: %{{customdata:,}}"
                      "<extra></extra>"))
    return fig

#[8] conditions 

def fig_conditions(cond: pd.DataFrame, stat: str, overall: float, min_n: int,
                   top_n: int) -> go.Figure:
#the chosen statistic for adults WITHOUT (light dot) vs. WITH (dark dot) each condition.
    d = cond[~cond["small"]].sort_values("value", ascending=False).head(top_n)
    if d.empty:
        return empty_fig(f"No condition has {min_n}+ adults in this selection", height=400)
    labels = d["label"].tolist()
    fig = new_fig(f"{STATS[stat]['label']}: Adults Without vs. With Each Condition",
                  subtitle="Sorted by the value WITH the condition; dashed line = all adults "
                           f"in selection ({fmt_stat(overall, stat)})",
                  legend=True, height=30 * len(d) + 160,
                  xaxis={"rangemode": "tozero", **axis_format(stat)},
                  yaxis={"type": "category", "categoryorder": "array", "categoryarray": labels,
                         "autorange": "reversed"})
    xs, ys = [], []
    for label, without, with_ in zip(labels, d["value_without"], d["value"]):
        xs += [without, with_, None]
        ys += [label, label, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line={"color": C["neutral"], "width": 2},
                    hoverinfo="skip", showlegend=False)
    info = np.c_[d["n_with"], d["prevalence"], d["value_without"], d["value"], d["ratio"]]
    hover = ("<b>%{y}</b><br>With: " + hover_value(stat, "customdata[3]")
             + " (%{customdata[0]:,.0f} adults, %{customdata[1]:.1f}% of selection)"
             + "<br>Without: " + hover_value(stat, "customdata[2]")
             + "<br>With ÷ without: %{customdata[4]:.2f}×<extra></extra>")
    for name, column, color, size in (("Without Condition", "value_without", C["secondary"], 10),
                                      ("With Condition", "value", C["primary"], 11)):
        fig.add_scatter(x=d[column], y=labels, mode="markers", name=name, customdata=info,
                        marker={"size": size, "color": color,
                                "line": {"color": C["surface"], "width": 2}},
                        hovertemplate=hover)
    fig.add_vline(x=overall, line={"color": C["ink"], "width": 1.2, "dash": "dash"})
    return fig

#[8] Find similar adults

def fig_profile_bands(matched: pd.Series, everyone: pd.Series) -> go.Figure:
    labels = list(everyone.cat.categories)
    fig = new_fig("2024 spending: Similar Adults vs. Everyone", height=360, legend=True,
                  barmode="group", bargap=0.3, bargroupgap=0.06, xaxis={"type": "category"},
                  yaxis={"ticksuffix": "%", "rangemode": "tozero"})
    for name, series, color in (("Similar Adults", matched, C["primary"]),
                                ("All Adults", everyone, C["secondary"])):
        pct = series.value_counts(normalize=True).reindex(labels, fill_value=0) * 100
        fig.add_bar(x=labels, y=pct, name=name, marker_color=color,
                    hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.1f}%<extra></extra>")
    return fig

#[5] page: header + sidebar filters

if not DATA_PATH.exists():
    st.error(f"Could not find the data file:\n\n`{DATA_PATH}`\n\nPut "
             "`meps_health_expenditures_v1.csv` next to streamlit_app.py (or edit DATA_PATH).")
    st.stop()

data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
df, groups, bands = data["df"], data["groups"], data["bands"]
registry, cutoff = data["registry"], data["cutoff"]
N_ALL = len(df)
VARS_BY_BLOCK = {b: [v for v, spec in registry.items() if spec["block"] == b] for b in BLOCKS}
VARS_BY_BLOCK = {b: v for b, v in VARS_BY_BLOCK.items() if v}  # drop empty blocks

#Header (same two-column layout as the example: logo | title + info line) 
t1, t2 = st.columns((0.07, 1))
if LOGO_PATH.exists():
    t1.image(str(LOGO_PATH), width=120)
else:
    t1.markdown("<div style='font-size:3.4rem;line-height:1.15'>🏥</div>",
                unsafe_allow_html=True)
t2.title("Predicting the Price of Health: What Drives Next-Year Healthcare Spending?")
t2.markdown(
    f"**Data:** MEPS HC-257, Panel 28 (2023 to 2024) **|** **Adults:** {N_ALL:,} "
    "**|** **Outcome:** Total 2024 Healthcare Spending, All Payers **|** "
    "[MEPS codebook](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H257)")

#Sidebar: population filters + display options
AGE_MIN, AGE_MAX = int(df["age"].min()), int(df["age"].max())
FILTER_COLS = [(col, label) for col, label in SIDEBAR_FILTERS if col in groups]

def filter_options(col: str) -> list[str]:
    return [c for c in groups[col].cat.categories if c != MISSING]

def reset_filters() -> None:
    st.session_state["f_age"] = (AGE_MIN, AGE_MAX)
    for col, _ in FILTER_COLS:
        st.session_state[f"f_{col}"] = []

with st.sidebar:
    st.header("Filter the Population")
    st.caption("Every chart and table updates to the adults you select. "
               "Leave a box empty to include everyone.")
    age_range = st.slider("Age in 2023", AGE_MIN, AGE_MAX, (AGE_MIN, AGE_MAX), key="f_age")
    chosen = {col: st.multiselect(label, filter_options(col), key=f"f_{col}", placeholder="All")
              for col, label in FILTER_COLS}
    st.button("Reset filters", on_click=reset_filters, icon=":material/restart_alt:")

    st.divider()
    st.header("Display options")
    stat = st.radio("Spending measure", list(STATS), format_func=lambda k: STATS[k]["label"],
                    key="stat", help="Used by the variable explorer, the two-variable heatmap "
                                     "and the conditions chart.")
    min_n = st.slider("Hide groups with fewer adults than", 10, 200, DEFAULT_MIN_N, step=10,
                      key="min_n", help="Small groups give noisy numbers. Hidden groups still "
                                        "appear (grayed out) in the explorer's table.")
    show_missing = st.toggle("Show 'Missing' as its own group", value=True, key="show_missing",
                             help="About 31% of adults skipped the self-administered "
                                  "questionnaire (SAQ). They spend less, so their 'Missing' "
                                  "group carries real information.")
    show_ci = st.toggle("Show 95% confidence intervals", value=True, key="show_ci")

#apply filters
mask = df["age"].between(*age_range)
for col, picked in chosen.items():
    if picked:
        mask &= groups[col].isin(picked)
sel, gsel, bsel = df[mask], groups[mask], bands[mask]
n_sel = len(sel)
is_filtered = n_sel < N_ALL

with st.sidebar:
    st.divider()
    st.caption(f"**{n_sel:,}** of {N_ALL:,} adults selected ({n_sel / N_ALL:.0%}).")

if n_sel == 0:
    st.warning("No adults match these filters. Loosen a filter or press **Reset filters**.")
    st.stop()

#[6]page: KPI metric row
with st.spinner("Updating report..."):
    y24, y23 = sel[OUTCOME], sel[PRIOR_SPEND]
    mean24, mean23 = y24.mean(), y23.mean()
    med24, med23 = y24.median(), y23.median()
    high_pct, high_pct_all = sel[HIGH_COST].mean() * 100, df[HIGH_COST].mean() * 100
    zero24, zero23 = (y24 == 0).mean() * 100, (y23 == 0).mean() * 100

    #KPI cards: value + a change ("delta") line
    #Spending increases show red (delta_color="inverse"), as rising cost would.
    m1, m2, m3, m4, m5 = st.columns(5)
    card = {"border": True, "height": "stretch"}  #equal-height cards in the row
    m1.metric("Adults in Selection", f"{n_sel:,}", f"{n_sel / N_ALL:.0%} of all {N_ALL:,}",
              delta_color="off", delta_arrow="off", **card,
              help="Unweighted count of MEPS Panel 28 adults matching the sidebar filters.")
    m2.metric("Mean 2024 Spending", fmt_dollars(mean24),
              **delta_args(mean24 - mean23, f"{signed_dollars(mean24 - mean23)} vs. 2023", 0.5),
              **card, help="Average total 2024 healthcare spending (all payers), compared with "
                           "the same adults' 2023 average.")
    m3.metric("Median 2024 Spending", fmt_dollars(med24),
              **delta_args(med24 - med23, f"{signed_dollars(med24 - med23)} vs. 2023", 0.5),
              **card, help="Half of the selected adults spent less than this in 2024.")
    if is_filtered:
        high_delta = delta_args(high_pct - high_pct_all,
                                f"{high_pct - high_pct_all:+.1f} pts vs. all adults", 0.05)
    else:
        high_delta = {"delta": f"2024 spending ≥ ${cutoff:,.0f}", "delta_color": "off",
                      "delta_arrow": "off"}
    m4.metric("High-Cost (Top 20%)", fmt_pct(high_pct), **high_delta, **card,
              help=f"Share of adults whose 2024 spending was at least ${cutoff:,.0f}, the cutoff "
                   f"for the top 20% of all {N_ALL:,} adults.")
    m5.metric("Spent $0 in 2024", fmt_pct(zero24),
              **delta_args(zero24 - zero23, f"{zero24 - zero23:+.1f} pts vs. 2023", 0.05,
                           inverse=False), **card,
              help="Share of selected adults with no healthcare spending in 2024, compared "
                   "with the share at $0 in 2023.")
    #MODEL HOOK (KPIs): add a sixth metric here for the headline model result
    #metrics = pd.read_csv(MODEL_OUTPUT_DIR / "model_comparison.csv")
    #m6.metric("Best model test R² (log $)", f"{metrics['test_r2_log'].max():.2f}")
    #(change st.columns(5) to st.columns(6) above

    #[7] spending-at-a-glance charts 
    st.markdown("### 2024 Spending at a Glance")
    g1, g2, g3 = st.columns(3)
    show_chart(g1, fig_spending_bands(bsel["band_2024"], y24), key="g1_bands")
    show_chart(g2, fig_concentration(y24), key="g2_concentration")
    show_chart(g3, fig_prior_vs_next(sel, bsel["band_2023"], min_n), key="g3_prior")

    #[8]tabs
    tab_explore, tab_rank, tab_two, tab_cond, tab_profile, tab_models = st.tabs([
        "🔍 Variable Explorer", "📊 Strongest Associations", "🧩 Variable Interactions",
        "🩺 Conditions", "👤 Find Similar Adults", "🔮 Predictive Models (Coming Soon)",
    ])

    #[8]Variable explorer (the effect of ONE variable on 2024 spending)
    with tab_explore:
        st.markdown("#### How Does 2024 Spending Differ Across Groups of One 2023 Variable?")
        blocks_available = list(VARS_BY_BLOCK)
        c1, c2 = st.columns((1, 2))
        block = c1.selectbox(
            "Type of Information (Predictor Block)", blocks_available,
            index=blocks_available.index(DEFAULT_EXPLORER[0])
            if DEFAULT_EXPLORER[0] in blocks_available else 0,
            format_func=lambda b: BLOCKS[b]["label"], key="ex_block",
            help="The same six blocks used in the modeling.")
        options = VARS_BY_BLOCK[block]
        default_var = DEFAULT_EXPLORER[1] if DEFAULT_EXPLORER[1] in options else options[0]
        var = c2.selectbox("Variable", options, index=options.index(default_var),
                           format_func=lambda v: registry[v]["label"],
                           key=f"ex_var_{block}")  #one remembered choice per block
        spec = registry[var]
        grp = gsel[var]
        pct_missing = float((grp == MISSING).mean() * 100)
        st.caption(f"**{spec['label']}**: {spec['desc']}  ·  MEPS Source: `{spec['meps']}`  ·  "
                   f"Column: `{var}`  ·  {pct_missing:.1f}% Missing in Selection")

        summary, overall = group_summary(sel, grp, stat, min_n, show_missing)
        shown = summary[~summary["small"]] if not summary.empty else summary

        #a small metric strip that sums up the variable.
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Groups Shown", f"{len(shown)} of {len(summary)}", border=True, height="stretch",
                  help=f"Groups with fewer than {min_n} adults are hidden from the charts.")
        real = shown[shown["group"] != MISSING] if not shown.empty else shown
        if len(real) >= 2:
            top_g, bottom_g = real.loc[real["value"].idxmax()], real.loc[real["value"].idxmin()]
            if STATS[stat]["unit"] == "$":
                gap = (f"{top_g['value'] / bottom_g['value']:.1f}×" if bottom_g["value"] > 0
                       else "n/a (lowest group is $0)")
            else:
                gap = f"{top_g['value'] - bottom_g['value']:.1f} pts"
            k2.metric("Highest vs. Lowest Group", gap,
                      f"{top_g['group']} vs. {bottom_g['group']}", delta_color="off",
                      delta_arrow="off", border=True, height="stretch",
                      help=f"Compares {STATS[stat]['label'].lower()} between the highest and "
                           "lowest group shown (not counting 'Missing').")
        else:
            k2.metric("Highest vs. lowest group", "–", border=True, height="stretch")
        assoc_all = association_table(data["assoc_x"][mask], data["assoc_meta"],
                                      sel[OUTCOME_LOG], min_n)
        var_assoc = assoc_all[assoc_all["variable"] == var]
        if var_assoc.empty:
            k3.metric("Correlation with Log Spending", "–", border=True, height="stretch")
        else:
            best = var_assoc.iloc[0]  #already sorted by |r|
            level = best.name.split("=", 1)[1] if "=" in best.name else None
            k3.metric("Correlation with Log Spending", f"r = {best['r']:+.2f}",
                      f"strongest level: {level}" if level else None,
                      delta_color="off", delta_arrow="off", border=True, height="stretch",
                      help="Pearson r with log(1 + 2024 spending). For categories, each level "
                           "is compared with all other levels. These are descriptive, not causal.")
        k4.metric("Missing / Not Asked", fmt_pct(pct_missing), border=True, height="stretch",
                  help="Share of the selection with no answer for this variable.")

        left, right = st.columns((1.15, 1))
        show_chart(left, fig_group_stat(summary, stat, overall, show_ci, min_n),
                   key="ex_stat_chart")
        show_chart(right, fig_group_composition(grp, bsel["band_detail"],
                                                shown["group"].astype(str).tolist()
                                                if not shown.empty else []),
                   key="ex_comp_chart")
        st.caption("**Red:** The group's 95% CI is entirely above the value for all "
                   "selected adults; **Blue**: Entirely below; **Gray**: Not clearly different. "
                   "CIs use a bootstrap for $ measures and the Wilson interval for percentages. ")
        if not summary.empty:
            show_chart(st, table_group_summary(summary, stat, f"{spec['label']}: group summary"),
                       key="ex_table")
            st.download_button("Download the Table (CSV)",
                               data=summary.drop(columns=["value"]).to_csv(index=False),
                               file_name=f"group_summary_{var}.csv", mime="text/csv",
                               on_click="ignore", key="dl_group_summary")

    #[8] Strongest associations across ALL variables
    with tab_rank:
        st.markdown("#### Which 2023 Variables Move Most Closely with 2024 Spending?")
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider("Number of Variables", 5, 40, DEFAULT_TOP_N, key="rank_n")
        blocks_in = c2.multiselect("Blocks to Include", blocks_available, default=blocks_available,
                                   format_func=lambda b: BLOCKS[b]["label"], key="rank_blocks")
        highlight_options = [""] + blocks_available
        highlight = c3.selectbox(
            "Highlight a Block", highlight_options,
            index=highlight_options.index(DEFAULT_HIGHLIGHT)
            if DEFAULT_HIGHLIGHT in highlight_options else 0,
            format_func=lambda b: BLOCKS[b]["label"] if b else "(none)", key="rank_highlight")
        ranked = assoc_all[assoc_all["block"].isin(blocks_in)]
        left, right = st.columns((2.4, 1))
        show_chart(left, fig_associations(ranked.head(top_n), highlight), key="rank_chart")
        with right:
            st.markdown("**How To Read**")
            st.caption("r is the Pearson correlation with log(1 + 2024 spending), the scale our "
                       "models use. 2023 dollars and visit counts are log-transformed first, "
                       "and each category level is compared with all other levels. Positive means "
                       "higher spending. Binary items need at least the sidebar minimum of adults "
                       "on each side.")
            st.caption("These are one-to-one descriptive associations. Many overlap with "
                       "prior spending, so they are not causal effects or prove model importances. "
                       "Note: Uninsured all of 2023 and Insurance coverage: Uninsured "
                       "are the same adults.")
        st.markdown("**Strongest Single Variable in Each Block**")
        best_per_block = ranked.groupby("block", sort=False).head(1)
        st.dataframe(
            pd.DataFrame({"Block": [BLOCKS[b]["label"] for b in best_per_block["block"]],
                          "Strongest Variable": best_per_block["label"].to_numpy(),
                          "r": best_per_block["r"].to_numpy(),
                          "|r|": best_per_block["abs_r"].to_numpy()}),
            hide_index=True,
            column_config={
                "Block": st.column_config.TextColumn(width="medium"),
                "Strongest variable": st.column_config.TextColumn(width="large"),
                "r": st.column_config.NumberColumn(format="%+.2f", width="small"),
                "|r|": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                       format="%.2f", width="medium"),
            })
        with st.expander("Full Ranked Table"):
            st.dataframe(
                ranked.assign(block=[BLOCKS[b]["label"] for b in ranked["block"]])
                [["label", "block", "r", "n"]]
                .rename(columns={"label": "Variable / level", "block": "Block",
                                 "n": "Adults with data"}),
                hide_index=True,
                column_config={"r": st.column_config.NumberColumn(format="%+.3f")})

    #[8] Two variables at once (interaction heatmap)
    with tab_two:
        st.markdown("#### How Do Two Variables Interact?")
        ordered_vars = [v for b in blocks_available for v in VARS_BY_BLOCK[b]]

        def var_label(v: str) -> str:
            return f"{BLOCKS[registry[v]['block']]['short']} · {registry[v]['label']}"

        c1, c2 = st.columns(2)
        row_default = DEFAULT_HEATMAP[0] if DEFAULT_HEATMAP[0] in registry else ordered_vars[0]
        col_default = DEFAULT_HEATMAP[1] if DEFAULT_HEATMAP[1] in registry else ordered_vars[1]
        row_var = c1.selectbox("Rows", ordered_vars, index=ordered_vars.index(row_default),
                               format_func=var_label, key="hm_row")
        col_var = c2.selectbox("Columns", ordered_vars, index=ordered_vars.index(col_default),
                               format_func=var_label, key="hm_col")
        if row_var == col_var:
            st.info("Pick two different variables.")
        else:
            val, cnt = two_way_summary(sel, gsel[row_var], gsel[col_var], stat, show_missing)
            show_chart(st, fig_heatmap(val, cnt, stat, min_n, registry[row_var]["label"],
                                       registry[col_var]["label"]), key="hm_chart")
            st.caption(f"Cells with fewer than {min_n} adults are blank. "
                       "Hover a cell to see how many adults it contains.")

    #[8] Conditions
    with tab_cond:
        st.markdown("#### Which Conditions Correlate to Higher 2024 Spending?")
        sources = {
            "Self-Reported Diagnoses": [v for v in registry if v.startswith("dx_")],
            "Conditions Categories (CCSR)": [v for v in registry if v.startswith("ccsr_")],
            "Body Systems": [v for v in registry if v.startswith("n_cond_")],
        }
        c1, c2 = st.columns((2, 1))
        with c1:
            source = st.segmented_control("Condition List", list(sources),
                                          default=list(sources)[0], required=True,
                                          key="cond_source")
        items = []
        for v in sources[source or list(sources)[0]]:
            raw = pd.to_numeric(sel[v], errors="coerce").to_numpy(dtype=float)
            if v.startswith("n_cond_"):  # any condition recorded in that body system
                raw = np.where(np.isnan(raw), np.nan, (raw > 0).astype(float))
                label = BODY_SYSTEMS.get(v.removeprefix("n_cond_"), v)
            else:
                label = registry[v]["label"].removeprefix("Ever diagnosed: ")
                label = label[:1].upper() + label[1:]
            items.append((v, label, raw))
        cond = condition_summary(sel, items, stat, min_n)
        overall_value = stat_value(sel[OUTCOME].to_numpy(dtype=float),
                                   sel[HIGH_COST].to_numpy(dtype=float), stat)
        n_charted = int((~cond["small"]).sum()) if not cond.empty else 0
        if n_charted > 1:  # a slider needs min < max
            top_cond = c2.slider("Conditions Shown", 1, n_charted, min(n_charted, 20),
                                 key=f"cond_top_{source}",
                                 help="Highest values first. Conditions with fewer adults than "
                                      "the sidebar minimum are left out of the chart (but are "
                                      "listed in the table).")
        else:
            top_cond = 1
        show_chart(st, fig_conditions(cond, stat, overall_value, min_n, top_cond),
                   key="cond_chart")
        if not cond.empty:
            money = STATS[stat]["unit"] == "$"
            value_format = "$%,d" if money else "%.1f%%"
            table = cond.sort_values("value", ascending=False)
            if money:  #whole dollars, still numbers (so the columns sort correctly)
                table = table.assign(value=table["value"].round(),
                                     value_without=table["value_without"].round())
            st.dataframe(
                table[["label", "prevalence", "n_with", "value", "value_without", "ratio"]],
                hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn("Condition", width="large"),
                    "prevalence": st.column_config.ProgressColumn(
                        "Prevalence", min_value=0.0, max_value=100.0, format="%.1f%%"),
                    "n_with": st.column_config.NumberColumn("Adults with it", format="%,d"),
                    "value": st.column_config.NumberColumn(f"{STATS[stat]['short']} with",
                                                           format=value_format),
                    "value_without": st.column_config.NumberColumn(
                        f"{STATS[stat]['short']} without", format=value_format),
                    "ratio": st.column_config.NumberColumn("With ÷ without", format="%.2f×"),
                })
        st.caption("Self-reported diagnoses were asked of every adult. Conditions "
                   "categories (CCSR) and body systems are recorded mostly when a condition "
                   "comes up at a medical visit, so they partly measure whether someone used care "
                   "in 2023. Codes ending in 000 are rare conditions that MEPS masks for "
                   "confidentiality.")

    #[8]Find similar adults 
    with tab_profile:
        st.markdown("#### What did 2024 spending look like for adults like this?")
        st.caption(f"Pick a profile and press the button. This looks up real adults in the full "
                   f"sample of {N_ALL:,} (the sidebar filters are ignored here).")
        profile_fields = [(col, label) for col, label in PROFILE_FIELDS if col in groups]
        with st.form("profile_form", border=True):
            form_cols = st.columns(3)
            profile = {}
            for i, (col, label) in enumerate(profile_fields):
                profile[col] = form_cols[i % 3].selectbox(
                    label, ["Any"] + filter_options(col), key=f"pf_{col}")
            st.form_submit_button("Find Similar Adults", type="primary")
        match = pd.Series(True, index=df.index)
        for col, choice in profile.items():
            if choice != "Any":
                match &= groups[col] == choice
        matched = df[match]
        if all(choice == "Any" for choice in profile.values()):
            st.info("Choose at least one characteristic above, then press **Find Similar Adults**.")
        elif matched.empty:
            st.warning("No adults in the sample match this profile. Set a field back to "
                       "'Any' to widen it.")
        else:
            if len(matched) < DEFAULT_MIN_N:
                st.warning(f"Only {len(matched)} adults match.")
            p1, p2, p3, p4 = st.columns(4)
            card = {"border": True, "height": "stretch"}
            med_diff = matched[OUTCOME].median() - df[OUTCOME].median()
            mean_diff = matched[OUTCOME].mean() - df[OUTCOME].mean()
            high_diff = matched[HIGH_COST].mean() * 100 - high_pct_all
            p1.metric("Matching adults", f"{len(matched):,}", f"{len(matched) / N_ALL:.1%} of all",
                      delta_color="off", delta_arrow="off", **card)
            p2.metric("Median 2024 spending", fmt_dollars(matched[OUTCOME].median()),
                      **delta_args(med_diff, f"{signed_dollars(med_diff)} vs. everyone", 0.5),
                      **card)
            p3.metric("Mean 2024 Spending", fmt_dollars(matched[OUTCOME].mean()),
                      **delta_args(mean_diff, f"{signed_dollars(mean_diff)} vs. everyone", 0.5),
                      **card)
            p4.metric("High-cost (top 20%)", fmt_pct(matched[HIGH_COST].mean() * 100),
                      **delta_args(high_diff, f"{high_diff:+.1f} pts vs. everyone", 0.05), **card)
            show_chart(st, fig_profile_bands(bands.loc[match, "band_detail"],
                                             bands["band_detail"]), key="profile_chart")
            #MODEL HOOK (profile): once a model is saved predict for this profile instead of only looking up matches
            #model = joblib.load(MODEL_OUTPUT_DIR / "final_model.joblib")
            #pred_log = model.predict(<one-row DataFrame built from `profile`>)
            #st.metric("Model-predicted 2024 spending", fmt_dollars(np.expm1(pred_log[0])))
            #Fields the form does not ask about need typical values (median / mode), like the way spline builds a "typical adult".

    #[8]Predictive models (PLACEHOLDER)
    with tab_models:
        st.markdown("#### Predictive Model Results")
        st.info("**Coming soon**\n"
                "Results from modeling will be added here once they are final.",
                icon=":material/construction:")
        st.markdown(
            "**Planned Sections**\n"
            "- Model Comparisons\n" #Cross-validated and test R² / RMSE for each model
            "- Contributions of Block Categories\n"
            "- Predicted vs. Actual 2024 Spending, by Decile of Prediction\n"
            "- Who Gets Flagged as a Likely Top-20% Spender?") #classification results
        #MODEL HOOK (main): export small result tables
        #reg_table.to_csv("model_outputs/model_comparison.csv")       
        #model, cv_r2, test_r2 ...
        #final_outcomes.to_csv("model_outputs/block_contributions.csv")
        #Any CSV saved in model_outputs is listed below automatically
        #replace these plain tables with charts once the result format is settled
        result_files = sorted(MODEL_OUTPUT_DIR.glob("*.csv")) if MODEL_OUTPUT_DIR.exists() else []
        if result_files:
            for path in result_files:
                with st.expander(f"`{path.name}`", expanded=False):
                    st.dataframe(pd.read_csv(path), hide_index=True)
        else:
            st.caption(f"No result files yet.")#Save CSVs to `{MODEL_OUTPUT_DIR.name}/` next to this script and they will show up here.


#[9]page: footer expanders 
with st.expander("About this Dataset and How to Read the Dashboard"):
    st.markdown(f"""
- **Who is in the Data?:** {N_ALL:,} adults (age 18+ in 2023) in MEPS Panel 28 with a valid 2024 
  spending value. Counts are **unweighted** (MEPS survey weights are not applied), so the numbers
  describe this sample, not the whole U.S. population.
- **Predictors:** 2023 information grouped into six blocks.
  The self-administered questionnaire (SAQ) went out in late 2023 to early 2024, so a few
  items slightly overlap 2024.
- **Outcome:** Total 2024 Healthcare Spending for All Payers (`tot_exp_y2`, MEPS TOTEXPY2).
- **Who is a High-Cost Spender?:** Top 20% of 2024 spenders, i.e. ${cutoff:,.0f} or more.
- **Missing Data:** About 31% of adults did not complete the SAQ, and they
  spent less in 2024. That is why "Missing" is shown as its own group by default.
- **Conditions-File Features** (CCSR categories, body systems, number of conditions) are
  recorded mostly when a condition comes up during a medical event, so they partly measure
  2023 care use. Codes ending in 000 are rare categories that MEPS masks for confidentiality.
- **Confidence Intervals:** Percentile bootstrap ({N_BOOT} resamples) for means and medians,
  Wilson interval for percentages.
- **Correlations** are one-variable-at-a-time and descriptive. They are not causal effects and
  not model importances.
- **Labeling note:** the CSV column `risk_taker` holds MEPS item ADOVER2, *"I can overcome
  illness without help from a medically trained person,"* and is labeled that way here.
""")

with st.expander("Data Dictionary"):
    if data["absent"]:
        st.caption("Registry columns not found in this CSV: " + ", ".join(data["absent"]))
    query = st.text_input("Search", placeholder="e.g. diabetes, SAQ, visits", key="dict_search")
    dictionary = data["dictionary"]
    if query:
        hit = dictionary.apply(
            lambda row: row.astype(str).str.contains(query, case=False, regex=False).any(), axis=1)
        dictionary = dictionary[hit]
    st.dataframe(dictionary, hide_index=True,
                 column_config={"% missing": st.column_config.NumberColumn(format="%.1f%%")})

with st.expander("Download Data"):
    st.caption(f"The {n_sel:,} adults currently selected in the sidebar, all columns.")
    st.download_button("Download Filtered Data (CSV)", data=lambda: sel.to_csv(index=False),
                       file_name="meps_filtered.csv", mime="text/csv", on_click="ignore",
                       key="dl_filtered")
