import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import random

# Set page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Function to generate sample data
@st.cache_data
def generate_sales_data():
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Date range for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Products
    products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones', 'Speaker']
    categories = ['Electronics', 'Electronics', 'Electronics', 'Peripherals', 'Peripherals', 'Peripherals', 'Audio', 'Audio']
    product_category = dict(zip(products, categories))
    
    # Regions and salespeople
    regions = ['North', 'South', 'East', 'West', 'Central']
    salespeople = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Jamie', 'Avery', 'Quinn', 'Dakota']
    
    # Prepare data
    data = []
    for date in date_range:
        # More sales on weekends and at the end of quarters
        day_factor = 1.5 if date.weekday() >= 5 else 1.0
        month_end_factor = 1.3 if date.day > 25 else 1.0
        quarter_end_factor = 1.5 if date.month in [3, 6, 9, 12] and date.day > 25 else 1.0
        season_factor = 1.3 if date.month in [11, 12] else 1.0  # Holiday season boost
        
        # Generate different sales patterns for each product
        for product in products:
            # Product-specific factors
            if product == 'Laptop':
                product_factor = 1.5
                base_price = 1200
                price_var = 200
            elif product == 'Smartphone':
                product_factor = 2.0
                base_price = 800
                price_var = 150
            elif product == 'Tablet':
                product_factor = 1.2
                base_price = 500
                price_var = 100
            elif product == 'Monitor':
                product_factor = 0.8
                base_price = 300
                price_var = 50
            elif product == 'Keyboard':
                product_factor = 0.6
                base_price = 80
                price_var = 20
            elif product == 'Mouse':
                product_factor = 0.7
                base_price = 50
                price_var = 15
            elif product == 'Headphones':
                product_factor = 1.0
                base_price = 150
                price_var = 50
            else:  # Speaker
                product_factor = 0.9
                base_price = 120
                price_var = 40
            
            # Daily transactions for this product
            daily_transactions = int(np.random.poisson(
                3 * product_factor * day_factor * month_end_factor * quarter_end_factor * season_factor
            ))
            
            for _ in range(daily_transactions):
                salesperson = random.choice(salespeople)
                region = random.choice(regions)
                
                # Add some randomness to price
                price = base_price + np.random.randint(-price_var, price_var)
                
                # Random quantity between 1 and 5
                quantity = np.random.randint(1, 6)
                
                # Calculate discount (more discounts during end of quarter and season)
                discount_factor = quarter_end_factor * season_factor
                # Ensure probabilities sum to 1
                p_no_discount = 0.5
                p_small = 0.2
                p_medium = 0.15
                p_large = 0.1
                p_xlarge = min(0.05 * discount_factor, 0.05)  # Cap at 0.05
                # Normalize probabilities to ensure they sum to 1
                total_p = p_no_discount + p_small + p_medium + p_large + p_xlarge
                probs = [p_no_discount/total_p, p_small/total_p, p_medium/total_p, p_large/total_p, p_xlarge/total_p]
                
                discount_rate = np.random.choice([0, 0.05, 0.1, 0.15, 0.2], p=probs)
                discount = price * discount_rate
                
                # Final price after discount
                final_price = price - discount
                total_amount = final_price * quantity
                
                # Customer satisfaction (higher for discounted items and quality products)
                base_satisfaction = np.random.normal(4, 0.5)  # Base rating around 4 stars
                discount_satisfaction = discount_rate * 2  # Higher discount, higher satisfaction
                product_quality = {'Laptop': 0.5, 'Smartphone': 0.3, 'Tablet': 0.2, 
                                  'Monitor': 0.1, 'Keyboard': 0, 'Mouse': 0,
                                  'Headphones': 0.4, 'Speaker': 0.2}.get(product, 0)
                
                satisfaction = min(5, max(1, base_satisfaction + discount_satisfaction + product_quality))
                satisfaction = round(satisfaction, 1)
                
                # Add entry
                data.append({
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Day': date.day,
                    'Quarter': (date.month - 1) // 3 + 1,
                    'WeekDay': date.day_name(),
                    'Product': product,
                    'Category': product_category[product],
                    'Region': region,
                    'Salesperson': salesperson,
                    'Price': price,
                    'Quantity': quantity,
                    'Discount': discount,
                    'Discount_Rate': discount_rate,
                    'Final_Price': final_price,
                    'Total_Amount': total_amount,
                    'Customer_Satisfaction': satisfaction
                })
    
    df = pd.DataFrame(data)
    return df

# Generate or load data
sales_df = generate_sales_data()

# Sidebar - Filters
st.sidebar.title("Dashboard Filters")

# Date range filter
with st.sidebar.expander("Date Filter", expanded=True):
    min_date = sales_df['Date'].min().date()
    max_date = sales_df['Date'].max().date()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(max_date - timedelta(days=90), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = sales_df[(sales_df['Date'].dt.date >= start_date) & 
                              (sales_df['Date'].dt.date <= end_date)]
    else:
        filtered_df = sales_df.copy()

# Product filter
with st.sidebar.expander("Product Filter", expanded=True):
    all_products = sales_df['Product'].unique().tolist()
    selected_products = st.multiselect("Select Products", all_products, default=all_products)
    
    if selected_products:
        filtered_df = filtered_df[filtered_df['Product'].isin(selected_products)]

# Region filter
with st.sidebar.expander("Region Filter", expanded=True):
    all_regions = sales_df['Region'].unique().tolist()
    selected_regions = st.multiselect("Select Regions", all_regions, default=all_regions)
    
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]

# Salesperson filter
with st.sidebar.expander("Salesperson Filter", expanded=False):
    all_salespeople = sorted(sales_df['Salesperson'].unique().tolist())
    selected_salespeople = st.multiselect("Select Salespeople", all_salespeople, default=all_salespeople)
    
    if selected_salespeople:
        filtered_df = filtered_df[filtered_df['Salesperson'].isin(selected_salespeople)]

# Main dashboard
st.title("📊 Sales Analytics Dashboard")

# Alert for filtered data
if len(filtered_df) == 0:
    st.error("No data available based on the selected filters. Please adjust your filter settings.")
elif len(filtered_df) < len(sales_df):
    st.info(f"Showing {len(filtered_df)} out of {len(sales_df)} total sales records based on your filters.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_df['Total_Amount'].sum()
    st.metric(
        label="Total Sales",
        value=f"${total_sales:,.2f}",
        delta=f"{((total_sales / filtered_df['Total_Amount'].count()) - 100):.1f}% per transaction" 
        if filtered_df['Total_Amount'].count() > 0 else "0%"
    )

with col2:
    avg_order_value = filtered_df['Total_Amount'].mean() if len(filtered_df) > 0 else 0
    st.metric(
        label="Average Order Value",
        value=f"${avg_order_value:.2f}"
    )

with col3:
    total_units = filtered_df['Quantity'].sum()
    st.metric(
        label="Total Units Sold",
        value=f"{total_units:,}"
    )

with col4:
    avg_satisfaction = filtered_df['Customer_Satisfaction'].mean() if len(filtered_df) > 0 else 0
    st.metric(
        label="Avg. Customer Satisfaction",
        value=f"{avg_satisfaction:.2f} ⭐"
    )

st.markdown("---")

# Time series analysis
st.subheader("Sales Trends Over Time")

time_granularity = st.radio(
    "Select Time Granularity", 
    options=["Daily", "Weekly", "Monthly", "Quarterly"],
    horizontal=True
)

# Prepare time series data based on selected granularity
if time_granularity == "Daily":
    time_df = filtered_df.groupby(pd.Grouper(key='Date', freq='D'))['Total_Amount'].sum().reset_index()
    x_title = "Date"
elif time_granularity == "Weekly":
    time_df = filtered_df.groupby(pd.Grouper(key='Date', freq='W-MON'))['Total_Amount'].sum().reset_index()
    x_title = "Week Starting"
elif time_granularity == "Monthly":
    time_df = filtered_df.groupby(pd.Grouper(key='Date', freq='MS'))['Total_Amount'].sum().reset_index()
    x_title = "Month"
else:  # Quarterly
    filtered_df['Quarter_Start'] = pd.PeriodIndex(filtered_df['Date'], freq='Q').to_timestamp()
    time_df = filtered_df.groupby('Quarter_Start')['Total_Amount'].sum().reset_index()
    x_title = "Quarter Starting"

# Plot time series
fig_time = px.line(
    time_df, 
    x='Date', 
    y='Total_Amount',
    markers=True,
    labels={'Date': x_title, 'Total_Amount': 'Total Sales ($)'},
    title=f'{'time_granularity'} Sales Trend'
)

fig_time.update_traces(
    line=dict(width=3),
    marker=dict(size=8)
)

fig_time.update_layout(
    xaxis_title=x_title,
    yaxis_title='Total Sales ($)',
    height=400,
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.1)'),
)

st.plotly_chart(fig_time, use_container_width=True)

# Product and Region Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Product")
    
    product_df = filtered_df.groupby('Product')['Total_Amount'].sum().reset_index().sort_values('Total_Amount', ascending=False)
    
    fig_product = px.bar(
        product_df,
        x='Product',
        y='Total_Amount',
        color='Product',
        text_auto='.2s',
        labels={'Total_Amount': 'Total Sales ($)', 'Product': 'Product'},
        title='Sales by Product'
    )
    
    fig_product.update_traces(
        texttemplate='$%{y:,.2f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}'
    )
    
    fig_product.update_layout(
        xaxis_title='Product',
        yaxis_title='Total Sales ($)',
        height=400,
        template='plotly_white',
        showlegend=False,
        xaxis=dict(tickangle=-45),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    st.plotly_chart(fig_product, use_container_width=True)

with col2:
    st.subheader("Sales by Region")
    
    region_df = filtered_df.groupby('Region')['Total_Amount'].sum().reset_index()
    
    fig_region = px.pie(
        region_df,
        values='Total_Amount',
        names='Region',
        hole=0.4,
        labels={'Total_Amount': 'Total Sales ($)', 'Region': 'Region'},
        title='Sales Distribution by Region'
    )
    
    fig_region.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.2f}<br>Percentage: %{percent}'
    )
    
    fig_region.update_layout(
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )
    
    st.plotly_chart(fig_region, use_container_width=True)

# Sales Performance by Category and Salesperson
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    
    category_df = filtered_df.groupby('Category')['Total_Amount'].sum().reset_index()
    
    fig_category = px.bar(
        category_df,
        x='Category',
        y='Total_Amount',
        color='Category',
        text_auto='.2s',
        labels={'Total_Amount': 'Total Sales ($)', 'Category': 'Product Category'},
        title='Sales by Product Category'
    )
    
    fig_category.update_traces(
        texttemplate='$%{y:,.2f}',
        textposition='outside',
    )
    
    fig_category.update_layout(
        xaxis_title='Product Category',
        yaxis_title='Total Sales ($)',
        height=400,
        template='plotly_white',
        showlegend=False,
    )
    
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.subheader("Top Salespeople")
    
    salesperson_df = filtered_df.groupby('Salesperson')['Total_Amount'].sum().reset_index().sort_values('Total_Amount', ascending=False).head(5)
    
    fig_salesperson = px.bar(
        salesperson_df,
        y='Salesperson',
        x='Total_Amount',
        orientation='h',
        color='Total_Amount',
        color_continuous_scale='Viridis',
        labels={'Total_Amount': 'Total Sales ($)', 'Salesperson': 'Salesperson'},
        title='Top 5 Salespeople by Revenue'
    )
    
    fig_salesperson.update_traces(
        texttemplate='$%{x:,.2f}',
        textposition='outside',
    )
    
    fig_salesperson.update_layout(
        xaxis_title='Total Sales ($)',
        yaxis_title='Salesperson',
        height=400,
        template='plotly_white',
        coloraxis_showscale=False,
    )
    
    st.plotly_chart(fig_salesperson, use_container_width=True)

# Sales by Weekday and Discount Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Day of Week")
    
    # Ensure proper ordering of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_df = filtered_df.groupby('WeekDay')['Total_Amount'].sum().reset_index()
    weekday_df['WeekDay_Ordered'] = pd.Categorical(weekday_df['WeekDay'], categories=day_order, ordered=True)
    weekday_df = weekday_df.sort_values('WeekDay_Ordered')
    
    fig_weekday = px.line(
        weekday_df,
        x='WeekDay',
        y='Total_Amount',
        markers=True,
        labels={'Total_Amount': 'Total Sales ($)', 'WeekDay': 'Day of Week'},
        title='Sales Performance by Day of Week',
        category_orders={"WeekDay": day_order}
    )
    
    fig_weekday.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )
    
    fig_weekday.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Total Sales ($)',
        height=400,
        template='plotly_white',
    )
    
    st.plotly_chart(fig_weekday, use_container_width=True)

with col2:
    st.subheader("Discount Analysis")
    
    # Create discount brackets
    filtered_df['Discount_Bracket'] = pd.cut(
        filtered_df['Discount_Rate'], 
        bins=[-0.001, 0.001, 0.051, 0.101, 0.151, 1.0],
        labels=['No Discount', '0-5%', '5-10%', '10-15%', '15%+']
    )
    
    discount_df = filtered_df.groupby('Discount_Bracket')[['Total_Amount', 'Customer_Satisfaction']].mean().reset_index()
    
    fig_discount = px.bar(
        discount_df,
        x='Discount_Bracket',
        y='Total_Amount',
        color='Customer_Satisfaction',
        color_continuous_scale='RdYlGn',
        text_auto='.2f',
        labels={
            'Total_Amount': 'Avg Order Value ($)', 
            'Discount_Bracket': 'Discount Level',
            'Customer_Satisfaction': 'Satisfaction'
        },
        title='Average Order Value & Satisfaction by Discount Level'
    )
    
    fig_discount.update_traces(
        texttemplate='$%{y:.2f}',
        textposition='outside',
    )
    
    fig_discount.update_layout(
        xaxis_title='Discount Level',
        yaxis_title='Average Order Value ($)',
        height=400,
        template='plotly_white',
        coloraxis_colorbar=dict(title='Avg.<br>Satisfaction')
    )
    
    st.plotly_chart(fig_discount, use_container_width=True)

# Detailed Data Section
st.markdown("---")
st.subheader("Detailed Sales Data")

show_data = st.checkbox("Show Raw Data Table")
if show_data:
    # Allow pagination of data
    page_size = st.slider("Rows per page", min_value=5, max_value=50, value=10)
    
    # Calculate number of pages
    n_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    
    if n_pages > 0:
        page_num = st.number_input(
            f"Page (1-{n_pages})", 
            min_value=1, 
            max_value=max(1, n_pages), 
            step=1
        )
        
        # Display the relevant slice of data
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        display_cols = ['Date', 'Product', 'Category', 'Region', 'Salesperson', 
                       'Quantity', 'Price', 'Discount', 'Final_Price', 'Total_Amount', 'Customer_Satisfaction']
        
        st.dataframe(filtered_df.iloc[start_idx:end_idx][display_cols], use_container_width=True)
    else:
        st.write("No data to display")

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center;'><i>Data refreshes daily. Last updated: May 12, 2025</i></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>For any questions, contact the Analytics Team</div>", unsafe_allow_html=True)