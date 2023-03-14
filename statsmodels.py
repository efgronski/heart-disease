"""
Statsmodels Analysis

This file contains the functions used to analyze our heart disease
data using the statsmodels library. Coding analysis for these
functions can be found within the Jupyter notebooks that have the
same coding process within them, with output next to processes for
an easier reading.
"""

# Import libraries
import pandas as pd
import numpy as np
import statsmodels.api as smf
import matplotlib.pyplot as plt


def load_in_data(filename: str) -> pd.DataFrame:
    """
    Loads in heart disease data as a pandas dataframe
    """
    return pd.read_csv(filename)


def ols_summary(dataframe: pd.DataFrame, ols_input: str):
    """
    Takes in the data and the ols input to compute an ols model ane then
    create a summary table of information. Your ols input must be in
    syntax of 'output ~ input' with the option to add multiple input
    written as 'output ~ input + input'.
    """
    m = smf.ols(ols_input, data = dataframe).fit()
    return m.summary()


def split_location(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes heart data and splits it by location, and then returns a US
    dataframe. This will only return data for US as data for Europe will not
    run in OLS model due to its size.
    """
    us = data[(data['dataset'] == 'VA Long Beach') | (data['dataset'] == 'Cleveland')]
    return us


def split_gender(data: pd.DataFrame, gender: str = 'Female') -> pd.DataFrame:
    """
    Filters heart data by gender, and returns a dataframe with the specified gender.
    In heart data there is only two options, and in this function the default value
    is female. For heart data, the first letter of gender should be capitalized.
    """
    new_data = data[data['sex'] == gender]
    return new_data


def split_age(data: pd.DataFrame, under55: bool = True) -> pd.DataFrame:
    """
    Filters heart data by age, and returns a dataframe with the specifiec age
    range. For our analysis we only observe above and below 55 years of age.
    For values under 55, set under55 to True, and for 55 and above set value
    to False. The defualt value is set to True.
    """
    if under55 == True:
        return data[data['age'] < 55]
    else:
        return data[data['age'] >= 55]


def main():
    # load in the data
    heart = load_in_data('heart_disease_uci.csv')
    # create summary table for all data, excluding location
    print("Summary Table for all Data excluding Location:")
    ols_summary(heart,
                "num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    # create summary table for statistically significant values
    print("Summary Table for our most statistically significant values:")
    ols_summary(heart, "num ~ cp + oldpeak + ca")
    # create summary table for all data
    print("Summary Table for all data, including location:")
    ols_summary(heart,
                "num ~ age + sex + dataset + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    # creates summary table for statitsically significant values and location
    print("Summary Table for most statistically significant values, including location:")
    ols_summary(heart, "num ~ cp + oldpeak + ca + dataset")
    # creates summary table for us data only
    print("Summary Table for US data only:")
    ols_summary(split_location(heart),
                "num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    # creates summary table for women and for men
    print("Summary Table for Women's data:")
    ols_summary(split_gender(heart),
                "num ~ age + sex + dataset + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    print("Summary Table for Men's data:")
    ols_summary(split_gender(heart, 'Male'),
                "num ~ age + sex + dataset + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    # creates summary table for patients under 55 and those 55 and above in age
    print("Summary Table for patients under 55 in age:")
    ols_summary(split_age(heart),
                "num ~ age + sex + dataset + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    print("Summary Table for patients 55 and above in age:")
    ols_summary(split_age(heart, False),
                "num ~ age + sex + dataset + cp + trestbps + chol + fbs + restecg + thalch + exang + oldpeak + slope + ca + thal")
    

if __name__ == '__main__':
    main()