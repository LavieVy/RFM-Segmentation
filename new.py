#----------------------------------------------------------------------------------------------------
# Thư viện
#----------------------------------------------------------------------------------------------------
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import plotly.express as px
import squarify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


#----------------------------------------------------------------------------------------------------
# Hàm
#----------------------------------------------------------------------------------------------------
def load_data(uploaded_file_products, uploaded_file_transactions):
    try:
        products = pd.read_csv(uploaded_file_products)
        transactions = pd.read_csv(uploaded_file_transactions)
        return products, transactions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    
    # Function to validate data
def validate_data(products, transactions):
    required_product_columns = {'productId', 'productName', 'price'}
    required_transaction_columns = {'Member_number', 'Date', 'productId', 'items'}
    
    if not required_product_columns.issubset(products.columns):
        return False, "Products file is missing required columns"
    if not required_transaction_columns.issubset(transactions.columns):
        return False, "Transactions file is missing required columns"
    
    return True, None
def calculate_rfm(transactions, products):
    transactions['Date'] = pd.to_datetime(transactions['Date'], format='%d-%m-%Y')
    df = transactions.merge(products, on='productId')
    df = df.drop_duplicates()
    df = df.dropna()
    df['gross_sales'] = df['items'] * df['price']
    max_date = df['Date'].max().date()
    Recency = lambda x: (max_date - x.max().date()).days
    Frequency = lambda x: len(x)
    Monetary = lambda x: round(sum(x), 2)

    df_RFM = df.groupby('Member_number').agg({
        'Date': Recency,
        'productId': Frequency,
        'gross_sales': Monetary
    }).reset_index()
    df_RFM.columns = ['Member_number', 'Recency', 'Frequency', 'Monetary']
    return df_RFM, df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    df_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return df_filtered, df_outliers

def assign_rfm_labels(df_RFM):
	r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
	f_labels = range(1, 5)
	m_labels = range(1, 5)
	# Tách outliers
	df_filtered_recency, df_outliers_recency = remove_outliers_iqr(df_RFM, 'Recency')
	df_filtered_frequency, df_outliers_frequency = remove_outliers_iqr(df_RFM, 'Frequency')
	df_filtered_monetary, df_outliers_monetary = remove_outliers_iqr(df_RFM, 'Monetary')
	
	# Gán nhãn cho dữ liệu
	r_groups = pd.qcut(df_filtered_recency['Recency'].rank(method='first'), q=4, labels=r_labels)
	f_groups = pd.qcut(df_filtered_frequency['Frequency'].rank(method='first'), q=4, labels=f_labels)
	m_groups = pd.qcut(df_filtered_monetary['Monetary'].rank(method='first'), q=4, labels=m_labels)
	
	df_filtered_recency = df_filtered_recency.assign(R = r_groups.values)
	df_outliers_recency['R'] = df_outliers_recency['Recency'].apply(lambda x: 1 if x > df_filtered_recency['Recency'].max() else 4)

	df_filtered_frequency = df_filtered_frequency.assign(F = f_groups.values)
	df_outliers_frequency['F'] = df_outliers_frequency['Frequency'].apply(lambda x: 4 if x > df_filtered_frequency['Frequency'].max() else 1)

	df_filtered_monetary = df_filtered_monetary.assign(M = m_groups.values)
	df_outliers_monetary['M'] = df_outliers_monetary['Monetary'].apply(lambda x: 4 if x > df_filtered_frequency['Frequency'].max() else 1)

	df_complete_recency = pd.concat([df_filtered_recency, df_outliers_recency])
	df_complete_frequency = pd.concat([df_filtered_frequency, df_outliers_frequency])
	df_complete_monetary = pd.concat([df_filtered_monetary, df_outliers_monetary])
	# Merge các DataFrame
	df_complete = pd.merge(df_complete_recency, df_complete_frequency, on=['Member_number','Recency', 'Frequency', 'Monetary'], how='inner')
	df_complete = pd.merge(df_complete, df_complete_monetary, on=['Member_number','Recency', 'Frequency', 'Monetary'], how='inner')
	return df_complete


def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))

def rfm_level(df):
    # Check for special 'Champions' and 'New Customers' and 'Can’t Lose Them' conditions first
    if df['RFM_Score'] == 12:
        return 'CHAMP' # Bought recently, buy often and spend the most 
    elif df['R'] == 4 and df['F'] == 1 and df['M'] == 1:
        return 'NEW' 
    elif df['R'] == 1 and df['F'] == 4 and df['M'] == 4:
        return 'CANT_LOSE' # Made big purchases and often, but long time ago
    elif df['RFM_Score'] == 3:
        return 'LOST'    # Lowest recency, frequency & monetary scores 
    
    elif df['R'] == 2 and df['F'] == 4 and df['M'] == 4:
        return 'RISK' # Spent big money, purchased often but quite long time ago 
    elif df['R'] == 4 and df['M'] < 3:
        return 'PROMISING' # Recent shoppers, but haven't spent much
    elif df['R'] == 4 and df['F'] >1 and df['M']>2:
        return 'POT_LOYAL' # Recent customers, spent good amount, bought more than once   
    # Then check for other conditions
    elif df['M'] == 4:
        return 'LOYAL'
    else:
        return 'REGULAR'
    
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input, na=False)]

    return df
#----------------------------------------------------------------------------------------------------
# Template
#----------------------------------------------------------------------------------------------------
# Template data for downloads
products_template = """
productId,productName,price
1,tropical fruit,7.803532
2, whole milk, 1.8
"""

transactions_template = """
Member_number,Date,productId,items
1808,21-07-2015,1,3
2552,5/1/2015,2,1
"""
#********************************************************************************************************************************
# GUI
#********************************************************************************************************************************

# Tiêu đề
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('IMG/download.jpg')

st.markdown("<h3 style='text-align: center; color: grey;'>Trung tâm tin học - ĐH KHTN</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Data Science</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Topic: Customer Segmentation </h1>", unsafe_allow_html=True)
st.title('PHÂN CỤM KHÁCH HÀNG ÁP DỤNG RFM')
st.markdown(''' Xây dựng hệ thống phân cụm khách hàng dựa trên dữ liệu mà cửa hàng cung cấp. 
            Hệ thống này sẽ giúp cửa hàng xác định các nhóm khách hàng khác nhau, 
            từ đó phát triển các chiến lược kinh doanh và dịch vụ chăm sóc khách hàng phù hợp với từng nhóm đối tượng
            ''')
st.image('IMG/1.jpg')

#----------------------------------------------------------------------------------------------------
# Side bar
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------    
menu = ['Tổng quan Project', 'Xây dựng mô hình', 'Phân cụm khách hàng', 'Tra tìm khách hàng']
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.subheader('Download Templates')
st.sidebar.download_button(
    label="Download Products Template",
    data=products_template,
    file_name='products_template.csv',
    mime='text/csv',)
st.sidebar.download_button(
    label="Download Transactions Template",
    data=transactions_template,
    file_name='transactions_template.csv',
    mime='text/csv',)

update_data = st.sidebar.selectbox('Chọn:', ('Không update data', 'Update data mới'))
    
if update_data == 'Update data mới':
    st.sidebar.header('Update Data')
    uploaded_file_products = st.sidebar.file_uploader("Choose a CSV file for Products", type="csv")
    uploaded_file_transactions = st.sidebar.file_uploader("Choose a CSV file for Transactions", type="csv")
    
    if uploaded_file_products is not None and uploaded_file_transactions is not None:
        products, transactions = load_data(uploaded_file_products, uploaded_file_transactions)
        
        if products is not None and transactions is not None:
            is_valid, error_message = validate_data(products, transactions)
            if is_valid:
                # Process data if valid (RFM calculation example)
                st.sidebar.success('Data loaded and validated successfully')
                # Insert RFM calculation and processing code here
                df_RFM, df =  calculate_rfm(transactions, products)
                df_complete = assign_rfm_labels(df_RFM)
                df_complete['RFM_Segment'] = df_complete.apply(join_rfm, axis=1)
                df_complete['RFM_Score'] = df_complete[['R','F','M']].sum(axis=1)
                df_RFM_rule = df_complete.copy()
                df_RFM_rule['RFM_Level'] = df_RFM_rule.apply(rfm_level, axis=1)
                
            else:
                st.sidebar.error(f"Data validation failed: {error_message}")
        else:
            st.sidebar.error("Failed to load files. Please check the file format and content.")
       
      
else:
    products, transactions = load_data('Products_with_Prices.csv', 'Transactions.csv')
    df_RFM_rule = pd.read_csv('RFM_rule_segments.csv')
    df =  pd.read_csv('full_trans.csv')


#----------------------------------------------------------------------------------------------------
if choice == 'Tổng quan Project':
#----------------------------------------------------------------------------------------------------
    st.subheader('Tổng quan Project')
    st.markdown('''
                ### Business Objective/Problem:

Cửa hàng X chuyên cung cấp các sản phẩm thiết yếu như rau, củ, quả, thịt, cá, trứng, sữa, và nước giải khát. Đối tượng khách hàng chủ yếu của cửa hàng là người tiêu dùng cá nhân. Chủ cửa hàng mong muốn tăng doanh số bán hàng, giới thiệu sản phẩm đến đúng đối tượng khách hàng, và nâng cao chất lượng dịch vụ để đạt được sự hài lòng tối đa từ khách hàng.
### Mục tiêu:

- Cải thiện hiệu quả quảng bá
- Tăng doanh thu bán hàng
- Cải thiện mức độ hài lòng của khách hàng
                ''')
    st.markdown('''
    ### Các kiến thức/ kỹ năng cần để giải quyết vấn đề này :
- Hiểu vấn đề
- Import các thư viện cần thiết và hiểu cách sử dụng
- Đọc dữ liệu (dữ liệu project này được cung cấp)
- Thực hiện EDA cơ bản
- Tiền xử lý dữ liệu: làm sạch, tạo tính năng mới, lựa chọn tính năng cần thiết…
- Trực quan hóa dữ liệu
- Lựa chọn thuật toán cho bài toán phân cụm
- Xây dựng model
- Đánh giá model
- Báo cáo kết quả''')
    st.write('''
    **Bước 1** : Business Understanding

    **Bước 2** : Data Understanding ==>  Xây dựng hệ thống phân cụm khách hàng dựa trên dữ liệu mà cửa hàng cung cấp. Hệ thống này sẽ giúp cửa hàng xác định các nhóm khách hàng khác nhau, từ đó phát triển các chiến lược kinh doanh và dịch vụ chăm sóc khách hàng phù hợp với từng nhóm đối tượng

    **Bước 3** : Data Preparation/ Prepare : làm sạch, trực quan hóa dữ liệu, cấu trúc dữ liệu RMF, phân tích dữ liệu RMF, xử lý ngoại lệ

    ''') 
    st.write(''' 
 1. Xây dựng giải pháp phân cụm khách hàng theo RFM với tập luật tự định nghĩa
 2. Xây dựng model phân cụm khách hàng theo RFM & thuật toán phân cụm:
- RFM + Kmeans (sklearn)
- RFM + Hierarchical Clustering
- RFM + Kmeans (pySpark)
''')
    
    st.write('''**Kết luận**''')
    st.write('''**Bước 6: Deployment & Feedback/ Act**''')
    st.write('''Đưa ra những cải tiến phù hợp để nâng cao sự hài lòng của khách hàng, thu hút sự chú ý của khách hàng mới''')
    
    st.subheader('Giáo viên hướng dẫn')
    st.write('''
    **Cô : Khuất Thùy Phương**
    ''')
    st.subheader('Học viên thực hiện')
    st.write('''
    **HV : Lê Thống Nhứt & Nguyễn Thị Tường Vy**
    ''')

#----------------------------------------------------------------------------------------------------
elif choice == 'Xây dựng mô hình':
#----------------------------------------------------------------------------------------------------
    st.subheader('Xây dựng mô hình')

#----------------------------------------------------------------------------------------------------
elif choice == 'Phân cụm khách hàng':
#----------------------------------------------------------------------------------------------------
      
    # Phần xem tổng quan dữ liệu
    st.header('Phân cụm khách hàng')
    st.subheader('Data Overview')
    st.subheader('Products')
    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=products.shape[0])
    col2.metric(label="Columns", value=products.shape[1])  
    st.write('Samples of Products data')
    st.dataframe(products.sample(5))

    st.subheader('Transactions')
    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=transactions.shape[0])
    col2.metric(label="Columns", value=transactions.shape[1]) 
    st.write('Samples of Transactions data')
    st.dataframe(transactions.sample(5))
        
    st.subheader('Kết quả phân vùng của khách hàng theo RFM')
    st.write('Download file csv đã phân nhóm khách hàng theo RFM')
    csv = df_RFM_rule.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='RFM_segments.csv',
        mime='text/csv',)
    st.dataframe(filter_dataframe(df_RFM_rule[['Recency', 'Frequency', 'Monetary', 'RFM_Segment', 'RFM_Level'] ]))

    rfm_agg = df_RFM_rule.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    
    st.subheader('Trực quan dữ liệu')
    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    colors_dict = {
        'CHAMP': 'gold',
        'LOYAL': 'purple',
        'POT_LOYAL': 'violet',
        'NEW': 'green',
        'PROMISING': 'yellow',
        'Need Attention': 'orange',
        
        'RISK': 'maroon',
        'CANT_LOSE': 'red',
    
        'LOST': 'black',
        'REGULAR': 'lightblue'
    }

    # Tạo danh sách màu theo thứ tự nhãn trong rfm_agg
    colors = [colors_dict[label] for label in rfm_agg['RFM_Level']]

    squarify.plot(sizes=rfm_agg['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors,
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                        for i in range(0, len(rfm_agg))], alpha=0.5 )


    plt.title("RFM rule Customers Segments \n",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig)
    
    fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
           hover_name="RFM_Level", size_max=100, title="RFM Rule segments 2D Scatter plot")
    for level, color in colors_dict.items():
        fig.update_traces(marker=dict(color=color), selector=dict(name=level))
    st.plotly_chart(fig)
    
    palette = [colors_dict[key] for key in df_RFM_rule['RFM_Level'].unique()]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Recency', y='Monetary', hue='RFM_Level', palette=palette, data=df_RFM_rule)
    plt.title('RFM Rule Segments 2D Scatter Plot')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    
    fig = px.scatter_3d(df_RFM_rule, x='Recency', y='Frequency', z='Monetary',
                    color='RFM_Level', opacity = 0.7, title='RFM Rule  segments 3D Scatter plot')
    for level, color in colors_dict.items():
        fig.update_traces(marker=dict(color=color), selector=dict(name=level))
    st.plotly_chart(fig)
    
    rfm_levels = df_RFM_rule['RFM_Level'].unique()
    selected_rfm_level = st.selectbox('Select RFM Level', rfm_levels)

    if selected_rfm_level:
        # Filter data based on selected RFM_Level
        members_in_level = df_RFM_rule[df_RFM_rule['RFM_Level'] == selected_rfm_level]['Member_number']
        filtered_data = df[df['Member_number'].isin(members_in_level)]
        
        if not filtered_data.empty:
            st.subheader(f'Data for RFM Level: {selected_rfm_level}')
            st.dataframe(filtered_data)

            # Sales over time
            st.subheader(f'Sales Over Time for RFM Level: {selected_rfm_level}')
            fig, ax = plt.subplots()
            filtered_data.groupby('Date')['gross_sales'].sum().plot(ax=ax)
            ax.set_ylabel('Gross Sales')
            ax.set_xlabel('Date')
            ax.set_title('Sales Over Time')
            st.pyplot(fig)

            # Sales by product
            st.subheader(f'Top 20 - Sales by Product for RFM Level: {selected_rfm_level}')
            product_sales = filtered_data.groupby('productName')['gross_sales'].sum().sort_values(ascending=True).head(20)
            fig, ax = plt.subplots()
            product_sales.plot(kind='barh', ax=ax)
            ax.set_xlabel('Gross Sales')
            ax.set_ylabel('Product Name')
            ax.set_title('Sales by Product')
            st.pyplot(fig)

            # Number of products bought
            st.subheader(f'Top 20 - Number of Products Bought for RFM Level: {selected_rfm_level}')
            customer_products = filtered_data.groupby('productName')['items'].sum().sort_values(ascending=True).head(20)
            fig, ax = plt.subplots()
            customer_products.plot(kind='barh', ax=ax)
            ax.set_xlabel('Number of Products')
            ax.set_ylabel('Product Name')
            ax.set_title('Number of Products Bought')
            st.pyplot(fig)
        else:
            st.write("No data found for the selected RFM Level.")
        
#----------------------------------------------------------------------------------------------------
elif choice == 'Tra tìm khách hàng':
#----------------------------------------------------------------------------------------------------


    # Phần tra cứu phân cụm khách hàng theo mã khách hàng
    st.header('Tra tìm khách hàng')
        
    default_ID = 1000
    member_number = st.number_input('Nhập Restaurant ID', min_value=df_RFM_rule.Member_number.min(), max_value=df_RFM_rule.Member_number.max(), value=default_ID, step=1)
                


    if member_number:
        # Filter data for the selected member
        user_data = df[df['Member_number']== member_number]
        user_segment = df_RFM_rule[df_RFM_rule['Member_number'] == member_number]
            
        if not user_data.empty:
            st.table(user_segment)
            st.dataframe(user_data)
            
            # Sales over time
            st.subheader(f'Sales Over Time for Member {member_number}')
            fig, ax = plt.subplots()
            user_data.groupby('Date')['gross_sales'].sum().plot(ax=ax)
            ax.set_ylabel('Gross Sales')
            ax.set_xlabel('Date')
            st.pyplot(fig)

            st.subheader(f'Sales by Product for Member {member_number}')
            product_sales = user_data.groupby('productName')['gross_sales'].sum().sort_values(ascending=True)
            fig, ax = plt.subplots()
            product_sales.plot(kind='barh', ax=ax)
            ax.set_xlabel('Gross Sales')
            ax.set_ylabel('Product Name')
            ax.set_title('Sales by Product')
            st.pyplot(fig)
                
            st.subheader(f'Number of Products member {member_number} Bought')
            customer_products = user_data.groupby('productName')['items'].sum().sort_values(ascending=True)
            fig, ax = plt.subplots()
            customer_products.plot(kind='barh', ax=ax)
            ax.set_xlabel('Number of Products')
            ax.set_ylabel('Product Name')
            ax.set_title('Number of Products Customer Bought')
            st.pyplot(fig)

        else:
            st.write("Không tìm thấy mã khách hàng.")

        
    # Add code to search for customer clusters here

    
    # Phần xem dataframe phân cụm của khách
    st.header('Dataframe Phân cụm của khách')
    # Add code to display customer clusters dataframe here

    # Phần biểu đồ các cụm
    st.header('Biểu đồ các cụm')
    # Add code to display cluster plots here


