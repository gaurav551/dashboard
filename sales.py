import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Superstore Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .st-bq {
        border-radius: 10px;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .st-emotion-cache-1bdcqw1 svg.stActionButton {
        display: none;
    }
    .stAppDeployButton {
     display: none;
    }
    .st-emotion-cache-1bdcqw1 button[kind="primaryFormSubmit"] {
        display: none;
    }
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Load the data with proper date parsing
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("D:/train.csv")
        # Convert date columns with correct format (day/month/year)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
        
        # Add derived columns
        df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
        df['Order Year'] = df['Order Date'].dt.year
        df['Order Day of Week'] = df['Order Date'].dt.day_name()
        
        # Check if Profit column exists, if not create a dummy one
        if 'Profit' not in df.columns:
            df['Profit'] = df['Sales'] * 0.1  # Assuming 10% profit if column doesn't exist
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data is loaded
if df.empty:
    st.error("No data loaded. Please check your data file path and format.")
    st.stop()

# Sidebar filters
st.sidebar.title("🔍 Filters")

# Date range filter - convert to datetime.date objects for Streamlit
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    # Convert back to datetime for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['Order Date'] >= start_date) & 
            (df['Order Date'] <= end_date)]
else:
    st.sidebar.warning("Please select a date range.")

# Region filter
regions = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)
df = df[df['Region'].isin(regions)]

# Category filter
categories = st.sidebar.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)
df = df[df['Category'].isin(categories)]

# Segment filter
segments = st.sidebar.multiselect(
    "Select Customer Segments",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)
df = df[df['Segment'].isin(segments)]

# Ship Mode filter
ship_modes = st.sidebar.multiselect(
    "Select Ship Modes",
    options=df['Ship Mode'].unique(),
    default=df['Ship Mode'].unique()
)
df = df[df['Ship Mode'].isin(ship_modes)]

# Main page
st.title("📈 Superstore Sales Analytics Dashboard")
st.markdown("##")

# KPI cards
total_sales = df['Sales'].sum()
total_orders = df['Order ID'].nunique()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0
total_profit = df['Profit'].sum()
avg_profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
top_product = df.groupby('Product Name')['Sales'].sum().idxmax() if not df.empty else "N/A"
top_customer = df.groupby('Customer Name')['Sales'].sum().idxmax() if not df.empty else "N/A"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sales", f"${total_sales:,.2f}", help="Sum of all sales in the selected period")
with col2:
    st.metric("Total Orders", f"{total_orders:,}", help="Count of unique order IDs")
with col3:
    st.metric("Avg Order Value", f"${avg_order_value:,.2f}", help="Total sales divided by number of orders")
with col4:
    st.metric("Total Profit", f"${total_profit:,.2f}", help="Total profit in the selected period")

col1, col2 = st.columns(2)
with col1:
    st.metric("Avg Profit Margin", f"{avg_profit_margin:.1f}%", help="Average profit margin across all sales")
with col2:
    st.metric("Top Product", top_product[:20] + "..." if len(top_product) > 20 else top_product)

st.markdown("---")

# First row of charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Sales Trend")
    
    monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()
    
    fig1 = px.line(
        monthly_sales,
        x='Order Month',
        y='Sales',
        markers=True,
        line_shape='spline',
        title="Monthly Sales Trend",
        template="plotly_white"
    )
    fig1.update_traces(line=dict(width=3), marker=dict(size=8))
    fig1.update_layout(
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sales by Day of Week")
    
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = df.groupby('Order Day of Week')['Sales'].sum().reindex(dow_order).reset_index()
    
    fig2 = px.bar(
        dow_sales,
        x='Order Day of Week',
        y='Sales',
        color='Sales',
        color_continuous_scale='Blues',
        text='Sales'
    )
    fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig2.update_layout(
        xaxis_title="",
        yaxis_title="Sales ($)",
        height=400,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# Second row of charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    
    fig3 = px.pie(
        category_sales,
        values='Sales',
        names='Category',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Sales by Sub-Category")
    
    subcat_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    
    fig4 = px.bar(
        subcat_sales,
        x='Sales',
        y='Sub-Category',
        orientation='h',
        color='Sub-Category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig4.update_layout(
        yaxis={'categoryorder':'total ascending'},
        showlegend=False,
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

# Third row - Geographic visualizations
st.subheader("Geographic Sales Distribution")

# Create state-level sales data
state_sales = df.groupby('State')['Sales'].sum().reset_index()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sales by State (Choropleth Map)**")
    
    fig5 = px.choropleth(
        state_sales,
        locations='State',
        locationmode="USA-states",
        color='Sales',
        scope="usa",
        color_continuous_scale="blues",
        hover_name='State',
        hover_data={'Sales': ':$,.0f'},
        labels={'Sales': 'Total Sales'}
    )
    fig5.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("**Top States by Sales**")
    
    fig6 = px.bar(
        state_sales.nlargest(10, 'Sales'),
        x='Sales',
        y='State',
        orientation='h',
        color='Sales',
        color_continuous_scale='Blues',
        text='Sales'
    )
    fig6.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig6.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=500,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig6, use_container_width=True)

# Fourth row - Additional visualizations
st.subheader("Additional Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 Products by Sales**")
    
    top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
    
    fig7 = px.bar(
        top_products,
        x='Sales',
        y='Product Name',
        orientation='h',
        color='Sales',
        color_continuous_scale='Blues'
    )
    fig7.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=400
    )
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.markdown("**Top 10 Customers by Sales**")
    
    top_customers = df.groupby('Customer Name')['Sales'].sum().nlargest(10).reset_index()
    
    fig8 = px.bar(
        top_customers,
        x='Sales',
        y='Customer Name',
        orientation='h',
        color='Sales',
        color_continuous_scale='Greens'
    )
    fig8.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Sales ($)",
        yaxis_title="",
        height=400
    )
    st.plotly_chart(fig8, use_container_width=True)

# Data table with download option
st.subheader("Detailed Sales Data")
st.dataframe(df.head(100), use_container_width=True)

# Add download button
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv'
)