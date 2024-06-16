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
    return df_filtered

r_labels = range(4, 0, -1)  # Recency: smaller values are better, hence reversed order
f_labels = range(1, 5)      # Frequency: higher values are better
m_labels = range(1, 5)      # Monetary: higher values are better
def assign_rfm_labels(df, quantiles, r_labels, f_labels, m_labels):
    
    def assign_label(value, quantiles, labels):
        if value <= quantiles[0.25]:
            return labels[0]
        elif value <= quantiles[0.50]:
            return labels[1]
        elif value <= quantiles[0.75]:
            return labels[2]
        else:
            return labels[3]
    
    df['R'] = df['Recency'].apply(assign_label, args=(quantiles['Recency'], r_labels))
    df['F'] = df['Frequency'].apply(assign_label, args=(quantiles['Frequency'], f_labels))
    df['M'] = df['Monetary'].apply(assign_label, args=(quantiles['Monetary'], m_labels))
    
    return df


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
menu = ['Tổng quan Project', 'Xây dựng mô hình', 'Phân cụm khách hàng', 'Tra phân cụm khách hàng']
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
                df_filtered = df_RFM.copy()
                for column in ['Recency', 'Frequency', 'Monetary']:
                    df_filtered = remove_outliers_iqr(df_filtered, column)
                quantiles = df_filtered.quantile(q=[0.25,0.5,0.75])
                quantiles = quantiles[['Recency', 'Frequency', 'Monetary']].to_dict()
                df_complete = assign_rfm_labels(df_RFM, quantiles, r_labels, f_labels, m_labels)
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
    df = pd.read_csv('full_trans.csv')
    df_RFM = df_RFM_rule[['Recency', 'Frequency', 'Monetary']]
    quantiles = {'Recency': {0.25: 58.0, 0.5: 142.0, 0.75: 275.0},
                 'Frequency': {0.25: 6.0, 0.5: 8.0, 0.75: 11.0},
                 'Monetary': {0.25: 43.9, 0.5: 73.905, 0.75: 111.3}}

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
    
    st.header('Xây dựng mô hình')
    st.subheader('Data preparation/ Prepare')
    st.subheader('Data Overview')
    
    col1, col2 , col3, col4 = st.columns(4)
    col1.metric(label="Rows", value=products.shape[0])
    col2.metric(label="Columns", value=products.shape[1])
    col3.metric(label="Rows", value=transactions.shape[0])
    col4.metric(label="Columns", value=transactions.shape[1]) 
    left_co, last_co = st.columns(2)
    with left_co:
        st.subheader('Products')
        st.write('Samples of Products data')
        st.dataframe(products.sample(5))
    with last_co:
        st.subheader('Transactions')
        st.write('Samples of Transactions data')
        st.dataframe(transactions.sample(5))
   

#----------------------------------------------------------------------------------------------------    
    st.subheader('Modeling & Evaluation/ Analyze')
    
    st.markdown('Tạo bảng RFM analysis cho mỗi khách hàng')
    st.table(df_RFM_rule[['Recency', 'Frequency', 'Monetary']].sample(5))
    st.image('IMG/RFM_Distribution.png')
    st.image('IMG/RFM_Correlation.png')
    tab1, tab2, tab3, tab4 = st.tabs(["RFM Level Function",
                                      "K-means Clustering sklearn", 
                                      "Hierarchical Clustering",
                                      "K-means Clustering-pySpark"])
#----------------------------------------------------------------------------------------------------
    with tab1:
        st.subheader("RFM Level Function", divider='rainbow')
        st.markdown('''- Tính toán RFM quartiles và tạo cách cột đánh nhãn và RFM score
                    ''')
        st.markdown('''
        - Xây dựng hàm rfm_level để phân loại khách hàng dựa trên điểm RFM (Recency - Gần đây, Frequency - Tần suất, Monetary - Tiền tệ). Dưới đây là giải thích chi tiết cho từng điều kiện trong hàm:

1. CHAMP: Khách hàng mua hàng gần đây, mua thường xuyên và chi tiêu nhiều nhất (điểm RFM là 12).
2. NEW: Khách hàng mới (R = 4, F = 1, M = 1), có nghĩa là họ mới mua hàng gần đây nhưng tần suất và số tiền chi tiêu còn thấp.
3. CANT_LOSE: Khách hàng không thể mất (R = 1, F = 4, M = 4), họ đã chi tiêu nhiều và thường xuyên nhưng đã lâu không mua hàng.
4. LOST: Khách hàng đã mất (điểm RFM là 3), có điểm recency, frequency và monetary thấp nhất.
5. HIBER: Khách hàng đang ngủ đông (R = 1, F = 2, M = 2), họ chi tiêu ít, mua hàng không thường xuyên và đã lâu không mua hàng.
6. ABT_SLEEP: Khách hàng sắp ngủ đông, có điểm recency, frequency và monetary dưới trung bình (R < 3, F < 3, M < 3).
7. RISK: Khách hàng có rủi ro (R = 2, F = 4, M = 4), họ đã chi tiêu nhiều và mua hàng thường xuyên nhưng đã khá lâu rồi.
8. PROMISING: Khách hàng triển vọng (R = 4 và M < 3), họ mới mua hàng gần đây nhưng chưa chi tiêu nhiều.
9. POT_LOYAL: Khách hàng có khả năng trung thành cao (R = 4 và F >1 và M >2), họ mới mua hàng gần đây, đã chi tiêu một số tiền tốt và mua hàng nhiều hơn một lần.
10. LOYAL: Khách hàng trung thành (M = 4), không quan tâm đến điểm R và F.
11. REGULAR: Tất cả các trường hợp khác được coi là khách hàng thường xuyên.
Hàm này giúp doanh nghiệp hiểu rõ hơn về cơ sở khách hàng của mình và phân loại họ dựa trên các hành vi mua sắm gần đây để áp dụng các chiến lược marketing phù hợp.
                    ''')
        st.image("IMG/RFM rule Segments11.png")
        st.image("IMG/Histograms_Segmented_by_RFM_rule_RFM_level.png")
        st.image("IMG/RFM Rule scatter 2d11.png")
        st.image("IMG/RFM Rule scatter 3d11.png")
#----------------------------------------------------------------------------------------------------
    with tab2:
        st.subheader("K-means Clustering sklearn", divider='rainbow')
        st.markdown('''
                    - Chuẩn hóa dữ liệu sử dụng StandardScaler
                    - Tìm kiếm k bằng Elbow Method
                                      
                    ''')
        st.image('IMG/The Elbow Method sklearn.png')
        st.markdown('''- Build model with k=5
                    
| Cluster | RecencyMean | FrequencyMean | MonetaryMean | Count | Percent |
|---------|-------------|---------------|--------------|-------|---------|
| 0       | 120.0       | 5.0           | 42.0         | 902   | 24.26   |
| 1       | 77.0        | 10.0          | 91.0         | 959   | 25.79   |
| 2       | 422.0       | 4.0           | 34.0         | 604   | 16.25   |
| 3       | 299.0       | 9.0           | 87.0         | 566   | 15.22   |
| 4       | 102.0       | 15.0          | 155.0        | 687   | 18.48   |

                    ''')
        st.image("IMG/kmean_sklearn Unsupervised Segments.png")
        st.image("IMG/Histograms Segmented by sklearn kmean.png")
        st.image("IMG/RFM kmean sklearn scatter 2d.png")
        st.image("IMG/RFM kmean sklearn scatter 3d.png")        
#----------------------------------------------------------------------------------------------------
    with tab3:
        st.subheader("Hierarchical Clustering", divider='rainbow')
        st.markdown('''
                    - Chuẩn hóa dữ liệu sử dụng StandardScaler
                    - Tìm kiếm k bằng Vẽ dendrogram 
                                      
                    ''')
        st.image('IMG/Dendrogram for Hierarchical Clustering.png')
        st.markdown('''- Build model AgglomerativeClustering with k=5
                    
| Cluster | RecencyMean | FrequencyMean | MonetaryMean | Count | Percent |
|---------|-------------|---------------|--------------|-------|---------|
| 0       | 123.0       | 14.0          | 149.0        | 818   | 22.00   |
| 1       | 313.0       | 7.0           | 65.0         | 651   | 17.51   |
| 2       | 91.0        | 10.0          | 85.0         | 1235  | 33.22   |
| 3       | 97.0        | 5.0           | 36.0         | 530   | 14.25   |
| 4       | 429.0       | 4.0           | 27.0         | 484   | 13.02   |


                    ''')
        
        st.image("IMG/RFM Hierarchical Cluster Unsupervised Segments.png")
        st.image("IMG/Histograms Segmented by RFM + Hierarchical Clustering.png")
        st.image("IMG/RFM Hierarchical Clustering scatter 2d.png")
        st.image("IMG/RFM Hierarchical Clustering scatter 3d.png")
#----------------------------------------------------------------------------------------------------
    with tab4:
        st.subheader("K-means Clustering-pySpark", divider='rainbow')
        st.markdown('''
                    - Chuẩn hóa dữ liệu sử dụng StandardScaler
                    - Tìm kiếm k bằng Elbow Method và tính Silhouette score
                                      
                    ''')
        st.image('IMG/spark_kmean_wssse.png')
        st.image('IMG/spark_kmean_silhouette.png')
        st.markdown('''- Build model with k=5
                    
| Cluster | RecencyMean | FrequencyMean | MonetaryMean | Count | Percent |
|---------|-------------|---------------|--------------|-------|---------|
| 0       | 421.0       | 4.0           | 34.0         | 598   | 16.08   |
| 1       | 77.0        | 10.0          | 91.0         | 966   | 25.98   |
| 2       | 102.0       | 15.0          | 155.0        | 688   | 18.50   |
| 3       | 120.0       | 6.0           | 42.0         | 901   | 24.23   |
| 4       | 302.0       | 9.0           | 86.0         | 565   | 15.20   |


                    ''')
        
        st.image("IMG/Spark kmean Unsupervised Segments.png")
        st.image("IMG/Histograms Segmented by RFM + KMeans(pySpark).png")
        st.image("IMG/RFM kmean pySpark scatter 2d.png")
        st.image("IMG/RFM kmean pySpark scatter 3d.png")
#----------------------------------------------------------------------------------------------------
elif choice == 'Phân cụm khách hàng':
#----------------------------------------------------------------------------------------------------
      
    # Phần xem tổng quan dữ liệu
    st.header('Phân cụm khách hàng')
    st.subheader('Data Overview')
    col1, col2 , col3, col4 = st.columns(4)
    col1.metric(label="Rows", value=products.shape[0])
    col2.metric(label="Columns", value=products.shape[1])
    col3.metric(label="Rows", value=transactions.shape[0])
    col4.metric(label="Columns", value=transactions.shape[1]) 
    left_co, last_co = st.columns(2)
    with left_co:
        st.subheader('Products')
        st.write('Samples of Products data')
        st.dataframe(products.sample(5))
    with last_co:
        st.subheader('Transactions')
        st.write('Samples of Transactions data')
        st.dataframe(transactions.sample(5))
        
    st.subheader('Kết quả phân vùng của khách hàng theo RFM sau khi tinh chỉnh mô hình')
    st.markdown('''Tính quantiles dựa trên data loại bỏ outlier, dán nhãn trên toàn bộ data, điều chỉnh RRM_Level function với việc loại bỏ những nhóm không có đặc trưng rất nổi bật (gộp vào regular), phân loại thành 9 loại: 'LOYAL', 'CHAMP', 'CANT_LOSE', 'RISK', 'POT_LOYAL', 'REGULAR',
       'PROMISING', 'LOST', 'NEW'
       ''')
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
    
    fig1 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
           hover_name="RFM_Level", size_max=100, title="RFM Rule segments 2D Scatter plot")
    for level, color in colors_dict.items():
        fig1.update_traces(marker=dict(color=color), selector=dict(name=level))
    st.plotly_chart(fig1)
    
    palette = [colors_dict[key] for key in df_RFM_rule['RFM_Level'].unique()]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='Recency', y='Monetary', hue='RFM_Level', palette=palette, data=df_RFM_rule)
    plt.title('RFM Rule Segments 2D Scatter Plot')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    
    fig2 = px.scatter_3d(df_RFM_rule, x='Recency', y='Frequency', z='Monetary',
                    color='RFM_Level', opacity = 0.7, title='RFM Rule  segments 3D Scatter plot')
    for level, color in colors_dict.items():
        fig2.update_traces(marker=dict(color=color), selector=dict(name=level))
    st.plotly_chart(fig2)
    
    # Get unique clusters and features
    clusters = df_RFM_rule['RFM_Level'].unique()
    features = ['Recency', 'Frequency', 'Monetary']

    # Set up subplots with a rectangular layout
    fig3, axes = plt.subplots(3, df_RFM_rule['RFM_Level'].nunique(), figsize=(30, 10))

    # Loop through features and clusters
    for i, feature in enumerate(features):
        for j, cluster in enumerate(clusters):
            # Select data for the current cluster and feature
            data = df_RFM_rule[df_RFM_rule['RFM_Level'] == cluster][feature]
            
            # Plot histogram for the current cluster and feature
            axes[i, j].hist(data, bins=20, color=colors[j], edgecolor='w', alpha=0.7)
            axes[i, j].set_title(f'{cluster} \n {feature}', fontsize=15)
            axes[i, j].set_xlabel('')
            axes[i, j].set_ylabel('')

    # Add a title to the entire plot
    plt.suptitle('Segments RFM Histograms - RFM_level', fontsize=20)

    # Adjusting layout to prevent overlapping
    plt.tight_layout()
    st.pyplot(fig3)

    
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
            fig4, ax = plt.subplots()
            filtered_data.groupby('Date')['gross_sales'].sum().plot(ax=ax)
            ax.set_ylabel('Gross Sales')
            ax.set_xlabel('Date')
            ax.set_title('Sales Over Time')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)

            # Sales by product
            st.subheader(f'Top 20 - Sales by Product for RFM Level: {selected_rfm_level}')
            product_sales = filtered_data.groupby('productName')['gross_sales'].sum().sort_values(ascending=True).head(20)
            fig5, ax = plt.subplots()
            product_sales.plot(kind='barh', ax=ax)
            ax.set_xlabel('Gross Sales')
            ax.set_ylabel('Product Name')
            ax.set_title('Sales by Product')
            st.pyplot(fig5)

            # Number of products bought
            st.subheader(f'Top 20 - Number of Products Bought for RFM Level: {selected_rfm_level}')
            customer_products = filtered_data.groupby('productName')['items'].sum().sort_values(ascending=True).head(20)
            fig6, ax = plt.subplots()
            customer_products.plot(kind='barh', ax=ax)
            ax.set_xlabel('Number of Products')
            ax.set_ylabel('Product Name')
            ax.set_title('Number of Products Bought')
            st.pyplot(fig6)
        else:
            st.write("No data found for the selected RFM Level.")
        
#----------------------------------------------------------------------------------------------------
elif choice == 'Tra phân cụm khách hàng':
#----------------------------------------------------------------------------------------------------

    st.header("Tra phân cụm khách hàng")
    # Phần tra cứu phân cụm khách hàng theo mã khách hàng
    tab1, tab2 = st.tabs(["Tra theo member number",
                          "Tra tìm theo RFM"])
#----------------------------------------------------------------------------------------------------
    with tab1:
        st.subheader('Chọn mã khách hàng có trong dữ liệu', divider='rainbow')
        member_number = st.selectbox('Chọn Member_number', df_RFM_rule.Member_number.unique())
        user_data = df[df['Member_number']== member_number]
        user_segment = df_RFM_rule[df_RFM_rule['Member_number'] == member_number]
                
        
        st.table(user_segment)
        st.dataframe(user_data)
                
                # Sales over time
        st.subheader(f'Sales Over Time for Member {member_number}')
        fig7, ax = plt.subplots()
        user_data.groupby('Date')['gross_sales'].sum().plot(ax=ax, style='o-')
        ax.set_ylabel('Gross Sales')
        ax.set_xlabel('Date')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig7)

        st.subheader(f'Sales by Product for Member {member_number}')
        product_sales = user_data.groupby('productName')['gross_sales'].sum().sort_values(ascending=True)
        fig8, ax = plt.subplots()
        product_sales.plot(kind='barh', ax=ax)
        ax.set_xlabel('Gross Sales')
        ax.set_ylabel('Product Name')
        ax.set_title('Sales by Product')
        st.pyplot(fig8)
                    
        st.subheader(f'Number of Products member {member_number} Bought')
        customer_products = user_data.groupby('productName')['items'].sum().sort_values(ascending=True)
        fig9, ax = plt.subplots()
        customer_products.plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Products')
        ax.set_ylabel('Product Name')
        ax.set_title('Number of Products Customer Bought')
        st.pyplot(fig9)

    with tab2:
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.subheader("Tra tìm theo RFM", divider='rainbow')
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng")
        st.write('''
    Hướng dẫn: Chọn phương thức nhập liệu, nhập liệu và dự đoán
    ''')
        menu2 = ["Nhập nhiều dòng dữ liệu trực tiếp", 
            "Đăng tệp Excel", "Đăng tệp CSV"]
        choice2 = st.selectbox("Phương thức nhập liệu",menu2)

            
        if choice2 == "Nhập nhiều dòng dữ liệu trực tiếp":
            st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
            
             # Tạo dataframe để người dùng nhập thông tin khách hàng
            df_customer = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
            
            number = st.number_input('Nhập số khách hàng cần phân cụm', 1,10,1,1)
            for i in range(number):
                st.write(f"Khách hàng {i+1}")
    
                # Tạo các number_input để nhập giá trị cho cột Recency, Frequency, Monetary
                recency = st.number_input("Recency", min_value=1, max_value=int(df_RFM['Recency'].max()), value=100, key=f"recency_{i}")
                frequency = st.number_input("Frequency", min_value=1, max_value=50, value=5, key=f"frequency_{i}")
                monetary = st.number_input("Monetary", min_value=1.0, max_value=500.0, value=100.0, key=f"monetary_{i}")
    
                # Thêm thông tin khách hàng vừa nhập vào dataframe
            df_customer = df_customer.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)
            # Thực hiện phân cụm khách hàng dựa trên giá trị của 3 cột này
                    # In kết quả ra màn hình
            st.write("Dữ liệu đã nhập:")
            st.write(df_customer)
            st.write("Phân cụm khách hàng...")
            df_customer = assign_rfm_labels(df_customer,  quantiles, r_labels, f_labels, m_labels)
            df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
            df_customer['RFM_Score'] = df_customer[['R','F','M']].sum(axis=1)
            df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
            st.dataframe(df_customer)   
           
       
        elif choice2 == "Đăng tệp Excel":
            st.write('Bạn chọn upload excel')
            uploaded_file = st.file_uploader("Bạn vui lòng chọn file chứa dữ liệu Recency, Frequency, Monetary")
            if uploaded_file is not None:
                # check file type not excel
                if uploaded_file.name.split('.')[-1] != 'xlsx':
                    st.write('File không đúng định dạng, vui lòng chọn file excel')

                elif uploaded_file.name.split('.')[-1] == 'xlsx':

                    # load data excel
                    df_upload = pd.read_excel(uploaded_file, header=0)
                    st.dataframe(df_upload)
                    
                    
                    if 'Recency' not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Recency?')
                        Recency_column = st.selectbox("Chọn cột chứa Recency", df_upload.columns)
                        df_upload = df_upload.rename(columns={Recency_column: 'Recency'})
                    if "Frequency" not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Frequency?')
                        Frequency_column = st.selectbox("Chọn cột chứa Frequency", df_upload.columns)
                        df_upload = df_upload.rename(columns={Frequency_column: 'Frequency'})
                    if "Monetary" not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Monetary?')
                        Monetary_column = st.selectbox("Chọn cột chứa Monetary", df_upload.columns)
                        df_upload = df_upload.rename(columns={Monetary_column: 'Monetary'})
                    
                    
                    st.write("Dữ liệu đã nhập:")
                    df_customer = df_upload[["Recency", "Frequency", "Monetary"]]
                    st.write(df_customer)
                    st.write("Phân cụm khách hàng...")
                    df_customer = assign_rfm_labels(df_customer,  quantiles, r_labels, f_labels, m_labels)
                    df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
                    df_customer['RFM_Score'] = df_customer[['R','F','M']].sum(axis=1)
                    df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
                    st.dataframe(filter_dataframe(df_customer))
                    
                    rfm_agg = df_customer.groupby('RFM_Level').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': ['mean', 'count']}).round(0)

                    rfm_agg.columns = rfm_agg.columns.droplevel()
                    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
                    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

                    # Reset the index
                    rfm_agg = rfm_agg.reset_index()
                    
                    st.subheader('Trực quan dữ liệu')


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
                    
                    fig_treemap, ax_treemap = plt.subplots()  # Tạo đối tượng fig và ax riêng biệt cho biểu đồ Tree Map
                    fig_treemap.set_size_inches(14, 10)

                    squarify.plot(sizes=rfm_agg['Count'],
                                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                                color=colors,
                                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                                        for i in range(0, len(rfm_agg))], alpha=0.5 )


                    plt.title("RFM rule Customers Segments \n",fontsize=26,fontweight="bold")
                    plt.axis('off')


                    ax_treemap.axis('off')
                    st.pyplot(fig_treemap)


                    fig11 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
                        hover_name="RFM_Level", size_max=100, title="RFM Rule segments 2D Scatter plot")
                    for level, color in colors_dict.items():
                        fig11.update_traces(marker=dict(color=color), selector=dict(name=level))
                    st.plotly_chart(fig11)
                    
                    palette = [colors_dict[key] for key in df_customer['RFM_Level'].unique()]
                    plt.figure(figsize=(10, 7))
                    sns.scatterplot(x='Recency', y='Monetary', hue='RFM_Level', palette=palette, data=df_customer)
                    plt.title('RFM Rule Segments 2D Scatter Plot')
                    plt.xlabel('Recency')
                    plt.ylabel('Monetary')

                    # Display the plot in Streamlit
                    st.pyplot(plt.gcf())
                    
                    fig12 = px.scatter_3d(df_customer, x='Recency', y='Frequency', z='Monetary',
                                    color='RFM_Level', opacity = 0.7, title='RFM Rule  segments 3D Scatter plot')
                    for level, color in colors_dict.items():
                        fig12.update_traces(marker=dict(color=color), selector=dict(name=level))
                    st.plotly_chart(fig12)

                    

        elif choice2 == "Đăng tệp CSV" :
            st.write('Bạn chọn upload csv')
            uploaded_file = st.file_uploader("Bạn vui lòng chọn file: ")
            if uploaded_file is not None:
                # check file type if not csv
                if uploaded_file.name.split('.')[-1] != 'csv':
                    st.write('File không đúng định dạng, vui lòng chọn file csv')
                elif uploaded_file.name.split('.')[-1] == 'csv':

                    # load data csv
                    df_upload = pd.read_csv(uploaded_file)

                    # predict sentiment of review
                    # Check if 'Comment' column exists
                    if 'Recency' not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Recency?')
                        Recency_column = st.selectbox("Chọn cột chứa Recency", df_upload.columns)
                        df_upload = df_upload.rename(columns={Recency_column: 'Recency'})
                    if "Frequency" not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Frequency?')
                        Frequency_column = st.selectbox("Chọn cột chứa Frequency", df_upload.columns)
                        df_upload = df_upload.rename(columns={Frequency_column: 'Frequency'})
                    if "Monetary" not in df_upload.columns:
                        st.write('Dữ liệu không hợp lệ, cột nào của bạn chứa Monetary?')
                        Monetary_column = st.selectbox("Chọn cột chứa Monetary", df_upload.columns)
                        df_upload = df_upload.rename(columns={Monetary_column: 'Monetary'})
                   
                    st.write("Dữ liệu đã nhập:")
                    df_customer = df_upload[["Recency", "Frequency", "Monetary"]]
                    st.write(df_upload)
                    st.write("Phân cụm khách hàng...")
                    df_customer = assign_rfm_labels(df_customer,  quantiles, r_labels, f_labels, m_labels)
                    df_customer['RFM_Segment'] = df_customer.apply(join_rfm, axis=1)
                    df_customer['RFM_Score'] = df_customer[['R','F','M']].sum(axis=1)
                    df_customer['RFM_Level'] = df_customer.apply(rfm_level, axis=1)
                    st.dataframe(filter_dataframe(df_customer))   
                    
                    rfm_agg = df_customer.groupby('RFM_Level').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': ['mean', 'count']}).round(0)

                    rfm_agg.columns = rfm_agg.columns.droplevel()
                    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
                    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

                    # Reset the index
                    rfm_agg = rfm_agg.reset_index()
                    
                    st.subheader('Trực quan dữ liệu')

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
                    
                    fig_treemap, ax_treemap = plt.subplots()  # Tạo đối tượng fig và ax riêng biệt cho biểu đồ Tree Map
                    fig_treemap.set_size_inches(14, 10)

                    squarify.plot(sizes=rfm_agg['Count'],
                                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                                color=colors,
                                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                                        for i in range(0, len(rfm_agg))], alpha=0.5 )


                    plt.title("RFM rule Customers Segments \n",fontsize=26,fontweight="bold")
                    plt.axis('off')


                    ax_treemap.axis('off')
                    st.pyplot(fig_treemap)


                    fig13 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
                        hover_name="RFM_Level", size_max=100, title="RFM Rule segments 2D Scatter plot")
                    for level, color in colors_dict.items():
                        fig13.update_traces(marker=dict(color=color), selector=dict(name=level))
                    st.plotly_chart(fig13)
                    
                    palette = [colors_dict[key] for key in df_customer['RFM_Level'].unique()]
                    plt.figure(figsize=(10, 7))
                    sns.scatterplot(x='Recency', y='Monetary', hue='RFM_Level', palette=palette, data=df_customer)
                    plt.title('RFM Rule Segments 2D Scatter Plot')
                    plt.xlabel('Recency')
                    plt.ylabel('Monetary')

                    # Display the plot in Streamlit
                    st.pyplot(plt.gcf())
                    
                    fig14 = px.scatter_3d(df_customer, x='Recency', y='Frequency', z='Monetary',
                                    color='RFM_Level', opacity = 0.7, title='RFM Rule  segments 3D Scatter plot')
                    for level, color in colors_dict.items():
                        fig14.update_traces(marker=dict(color=color), selector=dict(name=level))
                    st.plotly_chart(fig14)          
        # Từ kết quả phân cụm khách hàng, người dùng có thể xem thông tin chi tiết của từng cụm khách hàng, xem biểu đồ, thống kê...
        # hoặc thực hiện các xử lý khác



