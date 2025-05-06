import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

# Set up Streamlit app layout
st.set_page_config(page_title="Mental Health & Social Media Dashboard", layout="wide")
st.title("📱 Mental Health and Social Media Usage Dashboard")

# Load data
@st.cache_data
def load_data():
    url_social = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_1SGG51k-OJbCzzFck7SFQOSt4EvafRqwedPaxyyIzIrie_RdcuZcfOU9SYu4AQImcMJFEVNqO-Ma/pub?output=csv"
    url_platform = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQzRy338yxDiqPrifzNmXLYdh6toGhvRoArsA3vMd4Cwt5LrfhvzmQULVL6KYYqSFFdCVWlasoEuA15/pub?output=csv"
    return pd.read_csv(url_social), pd.read_csv(url_platform)

df_social, df_platform = load_data()

# Clean and preprocess df_social
mh_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df_social['Mental_Health_Score'] = df_social['Mental_Health_Status'].map(mh_map)
df_social['Support_Systems_Access'] = df_social['Support_Systems_Access'].astype(str)
df_social = df_social.dropna(subset=['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours', 'Mental_Health_Score'])

# Sidebar filters
st.sidebar.header("Filters")
support_filter = st.sidebar.selectbox("Support System Access", ["All", "Yes", "No"])
if support_filter != "All":
    df_social = df_social[df_social['Support_Systems_Access'] == support_filter]

# Correlation heatmap
st.subheader("🔍 Correlation Heatmap: Lifestyle Factors vs Mental Health")
corr_vars = ['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours', 'Mental_Health_Score']
corr = df_social[corr_vars].corr()
fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

# Multiple Regression Analysis (statsmodels)
st.subheader("📈 Multiple Regression: Predicting Mental Health Score (Statsmodels)")
X_sm = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
X_sm = sm.add_constant(X_sm)
y_sm = df_social['Mental_Health_Score']
model_sm = sm.OLS(y_sm, X_sm).fit()
st.text(model_sm.summary())

# Scikit-learn Regression
st.subheader("🤖 Scikit-learn Linear Regression")
X = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
y = df_social['Mental_Health_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

# Classification Model
st.subheader("🧠 Classification Model: Logistic Regression")
df_social['MH_Class'] = (df_social['Mental_Health_Score'] >= 3).astype(int)  # 1 = Good/Excellent, 0 = Poor/Fair
X_class = df_social[['Sleep_Hours', 'Physical_Activity_Hours', 'Screen_Time_Hours']]
y_class = df_social['MH_Class']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)

st.write(f"Classification Accuracy: {acc:.4f}")
st.write(f"F1 Score: {f1:.4f}")

# Load and clean df_platform for platform analysis
st.subheader("📱 Platform-Based Mental Health Comparison")
df_platform = df_platform[df_platform['6. Do you use social media?'] == 'Yes'].copy()
df_platform['Mental_Health_Score'] = 5 - pd.to_numeric(df_platform['18. How often do you feel depressed or down?'], errors='coerce')
platform_col = '7. What social media platforms do you commonly use?'
platforms = df_platform[platform_col].dropna().str.split(", ").explode().str.strip().unique().tolist()

# Extract platform usage and assign labels
df_platform['Platform'] = None
for platform in platforms:
    df_platform.loc[df_platform[platform_col].str.contains(platform, na=False), 'Platform'] = platform

# Drop rows with missing values
anova_df = df_platform.dropna(subset=['Mental_Health_Score', 'Platform'])

# Tukey's HSD Test
st.text("Tukey HSD Summary:")
tukey = pairwise_tukeyhsd(endog=anova_df['Mental_Health_Score'], groups=anova_df['Platform'], alpha=0.05)
st.text(tukey.summary())

# Bar plot with CI
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=anova_df, x='Platform', y='Mental_Health_Score', ci=95, ax=ax2)
ax2.set_title('Mental Health Score by Platform with 95% Confidence Intervals')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig2)

# Gender T-Test
st.subheader("👥 Gender-Based Mental Health Comparison")
gender_scores = df_platform[['2. Gender', 'Mental_Health_Score']].dropna()
male_scores = gender_scores[gender_scores['2. Gender'] == 'Male']['Mental_Health_Score']
female_scores = gender_scores[gender_scores['2. Gender'] == 'Female']['Mental_Health_Score']
t_stat, p_value = ttest_ind(male_scores, female_scores, equal_var=False)
st.write(f"T-Test: Male vs Female Mental Health Scores")
st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Optional: Show raw data
toggle = st.checkbox("Show Raw Data")
if toggle:
    st.write("Platform Dataset:", df_platform.head())
    st.write("Lifestyle Dataset:", df_social.head())
