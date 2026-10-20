import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IT Employee Burnout & Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD CSS
# =========================================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =========================================================
# COMPANY NAME
# =========================================================

COMPANY_NAME = "InnoTech Solutions"


# =========================================================
# ADMIN LOGIN DETAILS
# =========================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="login-container">'
        '<div class="login-main-title">'
        'IT Employee Burnout &amp; Attrition Predictor'
        '</div>'
        '<div class="login-subtitle">'
        'InnoTech Solutions'
        '</div>'
        '<div class="login-subtitle">'
        'Secure Employee Prediction System'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="login-title">🔐 Admin Login</div>',
        unsafe_allow_html=True
    )

    username = st.text_input(
        "Username",
        placeholder="Enter admin username"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter admin password"
    )

    if st.button(
        "LOGIN",
        use_container_width=True
    ):

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            st.session_state.logged_in = True
            st.session_state.username = username

            st.rerun()

        else:
            st.error("Invalid username or password.")

    st.stop()


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("employee_data.csv")


# =========================================================
# MACHINE LEARNING DATA
# =========================================================

X = df[
    [
        "Working_Hours",
        "Overtime",
        "Years_At_Company",
        "Salary_Hike"
    ]
].copy()

y = df["Quit_Job"]


# =========================================================
# CONVERT OVERTIME YES / NO
# =========================================================

X["Overtime"] = X["Overtime"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL ACCURACY
# =========================================================

test_prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_prediction
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="hero-section">'

    '<div class="hero-title">'
    'IT Employee Burnout &amp; Attrition Predictor'
    '</div>'

    '<div class="hero-description">'
    '<strong>InnoTech Solutions</strong>'
    '</div>'

    '<div class="hero-description">'
    'Explore employee work patterns and understand '
    'potential burnout and attrition indicators.'
    '</div>'

    '<div class="gold-line"></div>'

    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOGIN INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Logged In User",
        st.session_state.username
    )

with col2:

    st.metric(
        "Role",
        "Admin"
    )

with col3:

    st.metric(
        "Access",
        "All Employees"
    )


# =========================================================
# LOGOUT
# =========================================================

if st.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.username = ""

    st.rerun()


st.divider()


# =========================================================
# ADMIN DASHBOARD
# =========================================================

st.markdown(
    '<div class="section-title">👑 Admin Dashboard</div>',
    unsafe_allow_html=True
)

st.info(
    "Admin can view the employee dataset and enter "
    "new employee details for prediction."
)


# =========================================================
# EMPLOYEE DATASET
# =========================================================

st.markdown(
    '<div class="section-title">📋 Employee Dataset</div>',
    unsafe_allow_html=True
)

st.info(
    "These 20 employee records are used to train "
    "the machine learning model."
)

st.dataframe(
    df[
        [
            "Employee_ID",
            "Employee_Name",
            "Team",
            "Working_Hours",
            "Overtime",
            "Years_At_Company",
            "Salary_Hike",
            "Quit_Job"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# NEW EMPLOYEE PREDICTION
# =========================================================

st.markdown(
    '<div class="section-title">👨‍💼 New Employee Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="normal-description">'
    'Enter the employee details below. The entered Employee Name '
    'and Employee ID will be displayed in the prediction result.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMPLOYEE NAME AND ID
# =========================================================

col1, col2 = st.columns(2)

with col1:

    employee_name = st.text_input(
        "Employee Name",
        placeholder="Enter employee name"
    )

with col2:

    employee_id = st.text_input(
        "Employee ID",
        placeholder="Enter employee ID"
    )


# =========================================================
# WORK DETAILS
# =========================================================

st.markdown(
    '<div class="section-title">💼 Work Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    working_hours = st.number_input(
        "Weekly Working Hours",
        min_value=30,
        max_value=80,
        value=40,
        step=1
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

with col2:

    overtime = st.selectbox(
        "Regular Overtime",
        ["No", "Yes"]
    )

    salary_hike = st.number_input(
        "Salary Hike (%)",
        min_value=0,
        max_value=30,
        value=10,
        step=1
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Employee Status",
    use_container_width=True
):

    # =====================================================
    # VALIDATE NAME
    # =====================================================

    if employee_name.strip() == "":
        st.error("Please enter the Employee Name.")
        st.stop()


    # =====================================================
    # VALIDATE ID
    # =====================================================

    if employee_id.strip() == "":
        st.error("Please enter the Employee ID.")
        st.stop()


    # =====================================================
    # CONVERT OVERTIME
    # =====================================================

    overtime_value = 1 if overtime == "Yes" else 0


    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = pd.DataFrame({
        "Working_Hours": [working_hours],
        "Overtime": [overtime_value],
        "Years_At_Company": [years_at_company],
        "Salary_Hike": [salary_hike]
    })


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        input_data
    )[0]


    # =====================================================
    # QUIT PROBABILITY
    # =====================================================

    probabilities = model.predict_proba(
        input_data
    )[0]

    if "Yes" in model.classes_:

        yes_index = list(
            model.classes_
        ).index("Yes")

        quit_probability = (
            probabilities[yes_index] * 100
        )

    else:

        quit_probability = 0


    # =====================================================
    # ATTRITION STATUS
    # =====================================================

    if quit_probability <= 30:

        attrition_status = "STAY"
        status_icon = "🟢"

    elif quit_probability <= 60:

        attrition_status = "MEDIUM"
        status_icon = "🟡"

    else:

        attrition_status = "QUIT"
        status_icon = "🔴"


    # =====================================================
    # BURNOUT SCORE
    # =====================================================

    burnout_score = 0


    # Working Hours

    if working_hours > 50:

        burnout_score += 40

    elif working_hours > 45:

        burnout_score += 25

    else:

        burnout_score += 10


    # Overtime

    if overtime == "Yes":

        burnout_score += 30

    else:

        burnout_score += 5


    # Salary Hike

    if salary_hike < 8:

        burnout_score += 20

    elif salary_hike < 12:

        burnout_score += 10


    burnout_score = min(
        burnout_score,
        100
    )


    # =====================================================
    # BURNOUT RISK
    # =====================================================

    if burnout_score >= 70:

        burnout_risk = "High"

    elif burnout_score >= 40:

        burnout_risk = "Medium"

    else:

        burnout_risk = "Low"


    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    st.markdown(
        '<div class="section-title">✨ Prediction Results</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # RESULT CARDS
    # =====================================================

    col1, col2 = st.columns(2)


    # =====================================================
    # BURNOUT RESULT
    # =====================================================

    with col1:

        st.metric(
            "🔥 Burnout Assessment",
            f"{burnout_score} /100",
            delta=f"{burnout_risk} Burnout Risk",
            delta_color="off"
        )


    # =====================================================
    # ATTRITION RESULT
    # =====================================================

    with col2:

        st.metric(
            f"{status_icon} Attrition Status",
            attrition_status,
            delta=f"Quit Probability: {quit_probability:.1f}%",
            delta_color="off"
        )


    # =====================================================
    # BURNOUT PROGRESS
    # =====================================================

    st.progress(
        burnout_score / 100
    )

    st.markdown(
        f"""
        <div class="burnout-score">
            <strong>Burnout Score:</strong>
            {burnout_score}/100
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # BURNOUT MESSAGE
    # =====================================================

    if burnout_risk == "Low":

        st.success(
            "✅ Low Burnout Risk — the entered work pattern "
            "produced a low burnout score."
        )

    elif burnout_risk == "Medium":

        st.warning(
            "⚠️ Medium Burnout Risk — the entered work pattern "
            "shows a moderate burnout score."
        )

    else:

        st.error(
            "🔴 High Burnout Risk — the entered work pattern "
            "shows a high burnout score."
        )


    # =====================================================
    # ATTRITION MESSAGE
    # =====================================================

    if attrition_status == "STAY":

        st.info(
            f"Model predicts the Stay class. "
            f"Demo model probability of Quit: "
            f"{quit_probability:.1f}%"
        )

    elif attrition_status == "MEDIUM":

        st.warning(
            f"Model indicates a moderate attrition level. "
            f"Demo model probability of Quit: "
            f"{quit_probability:.1f}%"
        )

    else:

        st.error(
            f"Model indicates a higher attrition level. "
            f"Demo model probability of Quit: "
            f"{quit_probability:.1f}%"
        )


    # =====================================================
    # EMPLOYEE NAME AND ID
    # =====================================================

    st.markdown(
        f"""
        <div class="summary-item">
            <strong>Employee Name:</strong> {employee_name}
        </div>

        <div class="summary-item">
            <strong>Employee ID:</strong> {employee_id}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    if attrition_status == "STAY":

        st.success(
            "🟢 Employee is showing a lower attrition risk."
        )

        st.markdown("""
        **💡 Recommendations**

        - Maintain a balanced workload.
        - Encourage skill development and career growth.
        - Maintain good communication with the team.
        - Continue monitoring workload and job satisfaction.
        """)

    elif attrition_status == "MEDIUM":

        st.warning(
            "🟡 Employee is showing a moderate attrition risk."
        )

        st.markdown("""
        **💡 Recommendations**

        - Review workload and working hours.
        - Discuss career growth opportunities.
        - Encourage regular communication with the manager.
        - Monitor overtime and job satisfaction.
        """)

    else:

        st.error(
            "🔴 Employee is showing a higher attrition risk."
        )

        st.markdown("""
        **💡 Recommendations**

        - Review excessive workload and overtime.
        - Discuss work-life balance.
        - Explore career development opportunities.
        - Provide appropriate support and regular feedback.
        """)


    # =====================================================
    # EMPLOYEE SUMMARY
    # =====================================================

    st.markdown(
        '<div class="summary-title">'
        '📋 Employee Summary'
        '</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Employee Name:</strong> {employee_name}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Employee ID:</strong> {employee_id}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Working Hours:</strong> {working_hours}'
            f'</div>',
            unsafe_allow_html=True
        )

    with summary_col2:

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Overtime:</strong> {overtime}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Years at Company:</strong> {years_at_company}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="summary-item">'
            f'<strong>Salary Hike:</strong> {salary_hike}%'
            f'</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.info(
        "This result is only a coding demonstration. "
        "The sample dataset cannot reliably predict real "
        "employees' decisions or burnout."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:

    st.metric(
        "Training Records",
        len(df)
    )

with col3:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'

    '<div class="footer-title">'
    'IT Employee Burnout &amp; Attrition Predictor'
    '</div>'

    '<div class="footer-text">'
    'InnoTech Solutions'
    '</div>'

    '<div class="footer-text">'
    'Educational Machine Learning Project'
    '</div>'

    '</div>',
    unsafe_allow_html=True
)