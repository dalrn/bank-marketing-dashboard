import streamlit as st
import pandas as pd
import warnings
import altair as alt
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Academya Dashboard',
                   page_icon=':bank:', layout='wide')

st.markdown(
    """
    <style>
        html, body, [class*="st-"] {
            font-family: "Futura", sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
df = pd.read_csv('bank-full.csv')
df = df.rename(columns={'y': 'Subscribed'})

# Capitalize the first letter of each column name
df.columns = [col.capitalize() for col in df.columns]

# Capitalize all values in the dataframe
df = df.applymap(lambda x: x.capitalize() if isinstance(x, str) else x)

st.markdown("""<h1 style='text-align: center; font-size: 50px; font-weight: bold; 
            color: #F56060; font-family: Futura, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin-top: -50px;'>
            Bank Marketing Performance Analysis</h1>
            <h3 style="text-align: center; font-weight: normal; color: gray;">Who Subscribed and Why?</h3>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True) 

# Define custom styles for better visuals
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebarContent"] {
        background-color: #1E1E1E; /* Dark sidebar background */
        padding: 25px;
        border-radius: 10px;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 40px; /* Increased padding */
    }

    .sidebar-button {
        display: block;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        color: #CCCCCC;
        background-color: transparent;
        border: none;
        text-align: center;
        padding: 12px 0; /* Increased height */
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 10px auto;
    }

    .sidebar-button:hover {
        background-color: #333333;
        color: #F56060;
    }

    .sidebar-button.active {
        background-color: #F56060;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Dataset Overview"  # Default active page

# Sidebar buttons for navigation
if st.sidebar.button("📊 Dataset Overview", key="overview"):
    st.session_state.page = "Dataset Overview"

if st.sidebar.button("📈 Dataset Visualization", key="visualization"):
    st.session_state.page = "Dataset Visualization"

if st.sidebar.button("🤖 Predictive Model", key="predictive"):
    st.session_state.page = "Predictive Model"

# JavaScript to set active button styling
st.sidebar.markdown(
    f"""
    <script>
        var buttons = window.parent.document.querySelectorAll('.sidebar-button');
        buttons.forEach(btn => btn.classList.remove('active'));
        buttons[{["Dataset Overview", "Dataset Visualization", "Predictive Model"].index(st.session_state.page)}].classList.add('active');
    </script>
    """,
    unsafe_allow_html=True
)

# Add dropdown for "About the Dataset"
with st.sidebar.expander("About the Dataset"):
    st.write("""
        This dataset contains information about a bank's marketing campaigns. 
        The goal is to predict whether a client will subscribe to a term deposit.
    """)

if st.session_state.page == "Dataset Overview":
    # Divide the page into two columns with a little space between them
    col1, _, col2 = st.columns([1, 0.05, 2])
    
    with col1.container(height=850):
        st.markdown("### Filters")

        # Toggle to show/hide missing values
        show_missing = st.checkbox("Show missing values", value=True)
        
        # Categorical Filters
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        selected_cat_var = st.multiselect("Select Categorical Variable to Filter By", categorical_cols)
        
        selected_values = {}

        # Generate checkboxes for each selected categorical variable
        for cat in selected_cat_var:
            unique_vals = df[cat].dropna().unique().tolist()
            
            if cat == 'Month':
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                unique_vals = sorted(unique_vals, key=lambda x: month_order.index(x))

            st.markdown(f"Select **{cat}**:")  # Title for each categorical variable

            col1_1, col1_2, col1_3 = st.columns(3)  # Create three columns layout
            selected_values[cat] = []

            # Loop through categories and assign checkboxes to columns
            for i, val in enumerate(unique_vals):
                with [col1_1, col1_2, col1_3][i % 3]:  # Rotate through columns
                    if st.checkbox(val, key=f"{cat}_{val}", value=False):
                        selected_values[cat].append(val)

            # Apply filtering
            for cat, selected_vals in selected_values.items():
                if selected_vals:  # Only apply filter if values are selected
                    df = df[df[cat].isin(selected_vals)]
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add space between filters and columns to display
        
        # Numerical Filters
        numerical_cols = df.select_dtypes(include='number').columns.tolist()
        selected_num_vars = st.multiselect("Select Numerical Variables to Filter By", numerical_cols)
        
        for selected_num_var in selected_num_vars:
            min_val, max_val = st.slider(f"Select Range for {selected_num_var}", float(df[selected_num_var].min()), float(df[selected_num_var].max()), (float(df[selected_num_var].min()), float(df[selected_num_var].max())))
            df = df[(df[selected_num_var] >= min_val) & (df[selected_num_var] <= max_val)]
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add space between numerical filters and columns to display

    with col2:
        if not show_missing:
            df = df.dropna()
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown(f"<div style='text-align: right; font-size: 14px'>Showing <strong style='color: #F56060;'>{len(df):,}</strong> past marketing records</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Split the second column into two sub-columns
        sub_col1, _, sub_col2 = st.columns([1, 0.05, 1])
        
        with sub_col1:
            st.markdown("### Subscription Percentage")
            st.markdown("<br>", unsafe_allow_html=True)

            # Calculate the percentage of Y = "Yes"
            yes_count = df['Subscribed'].value_counts().get('Yes', 0)
            no_count = df['Subscribed'].value_counts().get('No', 0)
            yes_percentage = round((yes_count/(yes_count+no_count))*100)

            if yes_percentage <= 30:
                yes_color = 'red'
            elif yes_percentage >= 70:
                yes_color = '#52F257'
            else:
                yes_color = '#EBF226'

            # Create a donut chart using Altair
            source = pd.DataFrame({'Subscribed': ['Yes', 'No'], 'Count': [yes_count, no_count], '%': [yes_percentage, 100-yes_percentage]})  
            source['Legend_Label'] = source.apply(lambda x: f"{x['Subscribed']} ({x['Count']})", axis=1)

            chart = alt.Chart(source).mark_arc(innerRadius=90, outerRadius=110).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Legend_Label", type="nominal",
                                scale=alt.Scale(domain=source["Legend_Label"].tolist(), range=[yes_color, '#D3D3D3']),
                                legend=alt.Legend(title="Subscribed (Count)")),  # Updated legend
                tooltip=["Subscribed", "Count", "%"]
            ).properties(
                width=300,
                height=300
            )

            text = alt.Chart(pd.DataFrame({'text': [f"{yes_percentage}%"]})).mark_text(
                size=30, fontWeight='bold', color=yes_color
            ).encode(
                text='text:N'
            ).properties(
                width=300,
                height=300
            )

            st.altair_chart(chart+text)
        
        with sub_col2:
            st.markdown(f"""  
                #### Out of those <span style='color:{yes_color};'>{yes_percentage}%</span> who subscribed...  
            """, unsafe_allow_html=True)     
            st.markdown("<br>", unsafe_allow_html=True)

            df_yes = df[df['Subscribed'] == 'Yes']      
            def display_percentage(df, column, description):
                most_common = df[column].value_counts().idxmax().lower()
                count = df[column].value_counts().max()
                percentage = round((count / len(df)) * 100)

                if column == "Poutcome":
                    if most_common == "unknown":
                        description = "had an <span style='color: #F56060;'>unknown</span> outcome last campaign"
                    elif most_common == "success":
                        description = "<span style='color: #F56060;'>subscribed</span> to the product last campaign"
                    elif most_common == "failure":
                        description = "<span style='color: #F56060;'>did not subscribe</span> to the product last campaign"
                else:
                    description = description.format(f"<span style='color: #F56060;'>{most_common}</span>")

                st.markdown(
                    f"""
                    <div style="font-size: 40px; font-weight: bold; color: #F56060; display: inline;">{percentage}%</div>
                    <div style="font-size: 20px; font-weight: bold; display: inline;"> {description}</div>
                    <div style="font-size: 15px;">({count} people)</div>
                    """,
                    unsafe_allow_html=True
                )

            display_percentage(df_yes, "Contact", "were contacted using {}")
            display_percentage(df_yes, "Marital", "were {}")
            display_percentage(df_yes, "Poutcome", "had an unknown outcome last campaign")
                
elif st.session_state.page == "Data Visualization":
    st.title("Data Visualization")
